# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch

import onnx

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_policy_as_onnx(
    path: str,
    env: ManagerBasedRLEnv,
    policy: object,
    normalizer: object | None = None,
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # Currently only support PerceptiveShadowingEnv export
    policy_exporter = _OnnxPerceptivePolicyExporter(env, policy, normalizer, verbose)
    policy_exporter.export(path)


class _OnnxPerceptivePolicyExporter(_OnnxPolicyExporter):
    """ONNX exporter for PerceptiveShadowingEnv that bakes the reference motion
    into the graph as lookup tables indexed by env ``time_step``.

    The four motion-reference obs (``joint_pos_ref`` / ``joint_vel_ref`` /
    ``position_ref`` / ``rotation_ref``) are regenerated here by replicating the
    CommandTerm update logic directly from the motion_buffer's stored sequence,
    so no env rollout is needed.

    For ``rotation_ref`` we export the raw world-frame reference quaternions
    (10 frames per time_step, wxyz) — the inference side is expected to do the
    ``robot_quat^{-1} ⊗ ref_quat`` and tannorm conversion itself.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        policy,
        normalizer=None,
        verbose=False,
        motion_buffer_name: str | None = None,
        motion_idx: int = 0,
    ):
        super().__init__(policy, normalizer, verbose)
        
        # copy encoders parameters
        if hasattr(policy, "encoders"):
            self.encoders = copy.deepcopy(policy.encoders)
        else:
            raise ValueError("Policy does not have an encoder module.")

        self.num_obs = policy.num_actor_obs

        mref = env.scene["motion_reference"]
        num_frames = mref.num_frames                                  # 10
        frame_interval_s = float(mref.frame_interval_s[0].item())     # 0.1
        env_step_dt = float(env.step_dt)                              # 0.02

        # ---------- locate the motion_buffer and the chosen motion ----------
        if motion_buffer_name is None:
            motion_buffer_name = next(iter(mref._motion_buffers.keys()))
        buf = mref._motion_buffers[motion_buffer_name]
        seqs = buf._all_motion_sequences
        framerate = float(seqs.framerate[motion_idx].item())
        buffer_length = int(seqs.buffer_length[motion_idx].item())    # #frames at framerate

        # Raw per-motion arrays at native framerate (no env_origin applied — it
        # cancels out in the anchor-frame transform and isn't used elsewhere).
        # NOTE: seqs.{joint_pos,joint_vel,base_pos_w,base_quat_w} are
        # ConcatBatchTensor — indexing with a single int returns that motion's
        # own (buffer_length, *data_shape) slice (see ConcatBatchTensor
        # _getitem_from_batch_idx).
        m_joint_pos = seqs.joint_pos[motion_idx].to("cpu")      # (L, J)
        m_joint_vel = seqs.joint_vel[motion_idx].to("cpu")      # (L, J)
        m_base_pos_w = seqs.base_pos_w[motion_idx].to("cpu")    # (L, 3)
        m_base_quat_w = seqs.base_quat_w[motion_idx].to("cpu")  # (L, 4) wxyz

        # ---------- figure out how many env-steps T to support ----------
        # Last valid env-step is when `round((t_env * env_step_dt) * framerate) < buffer_length`
        # (first frame of the window still in range). Cap by episode_length_s.
        T_by_motion = int((buffer_length - 1) / framerate / env_step_dt) + 1
        T_by_episode = int(env.cfg.episode_length_s / env_step_dt) + 1
        T = min(T_by_motion, T_by_episode)
        self.time_step_total = T

        # ---------- build lookup tables: (T, num_frames, ...) ----------
        t_env = torch.arange(T, dtype=torch.float)                 # (T,)
        k = torch.arange(num_frames, dtype=torch.float)            # (F,)
        # sample_time[t, k] = env_step_time(t) + k * frame_interval_s
        sample_time = t_env.unsqueeze(-1) + k.unsqueeze(0) * frame_interval_s / env_step_dt * env_step_dt
        # clamp-to-last-frame behaviour of AMASS fill_motion_data:
        frame_idx = torch.round(sample_time * framerate).long()
        validity = frame_idx < buffer_length                       # (T, F) bool
        frame_idx_clamped = torch.where(validity, frame_idx, torch.full_like(frame_idx, buffer_length - 1))

        # joint_pos_ref: (motion.joint_pos[frame] - default_joint_pos) * validity
        default_joint_pos = env.scene["robot"].data.default_joint_pos[0].cpu()       # (J,) per-env randomised value; use env 0
        # `default_joint_pos_nominal` only exists when the randomize_default_joint_pos
        # event ran (it saves the pre-randomisation value there — see
        # instinctlab/envs/mdp/events/randomization.py). In PLAY mode that event is
        # typically disabled, so fall back to default_joint_pos[0].
        default_joint_pos_nominal = getattr(
            env.scene["robot"].data, "default_joint_pos_nominal", env.scene["robot"].data.default_joint_pos[0]
        ).cpu()
        default_joint_vel = env.scene["robot"].data.default_joint_vel[0].cpu()       # (J,)

        joint_pos_ref_tbl = m_joint_pos[frame_idx_clamped] - default_joint_pos       # (T, F, J)
        joint_vel_ref_tbl = m_joint_vel[frame_idx_clamped] - default_joint_vel       # (T, F, J)
        joint_pos_ref_tbl = joint_pos_ref_tbl * validity.unsqueeze(-1)
        joint_vel_ref_tbl = joint_vel_ref_tbl * validity.unsqueeze(-1)

        # position_ref (anchor_frame="reference"): take frame-0 of the window as
        # the anchor (that's what reference_frame.base_pos_w[:, 0] resolves to),
        # transform the window's positions into that frame.
        window_base_pos_w = m_base_pos_w[frame_idx_clamped]          # (T, F, 3)
        window_base_quat_w = m_base_quat_w[frame_idx_clamped]        # (T, F, 4)

        # anchor = window frame 0 → shape (T, 3) / (T, 4) for transform_points
        anchor_pos_w = window_base_pos_w[:, 0, :]                    # (T, 3)
        anchor_quat_w = window_base_quat_w[:, 0, :]                  # (T, 4)
        anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
            anchor_pos_w, anchor_quat_w
        )
        # transform_points with batch N=T, points (T, F, 3), pos (T, 3), quat (T, 4)
        position_ref_tbl = math_utils.transform_points(
            window_base_pos_w,
            anchor_pos_w_inv,
            anchor_quat_w_inv,
        ) * validity.unsqueeze(-1)                                   # (T, F, 3)

        # rotation_ref: export raw world-frame ref quat (wxyz), board-side does
        # `tan_norm(quat_inv(robot_quat_w) ⊗ ref_quat_w)`.
        rotation_ref_quat_w_tbl = window_base_quat_w * validity.unsqueeze(-1)   # (T, F, 4)

        # ---------- stash constants ----------
        self.joint_pos_ref_tbl = joint_pos_ref_tbl                             # (T, F, J)
        self.joint_vel_ref_tbl = joint_vel_ref_tbl                             # (T, F, J)
        self.position_ref_tbl = position_ref_tbl                               # (T, F, 3)
        self.rotation_ref_quat_w_tbl = rotation_ref_quat_w_tbl                 # (T, F, 4)

        self.default_joint_pos = default_joint_pos_nominal                     # (J,)
        self.action_scale = env.action_manager.get_term("joint_pos")._scale[0].cpu()
        self.joint_stiffness = env.scene["robot"].data.joint_stiffness[0].cpu()
        self.joint_damping = env.scene["robot"].data.joint_damping[0].cpu()

    def forward(self, obs, time_step):
        """
        Args:
            obs:       (1, obs_dim)
            time_step: (1, 1) long/float; clamped to [0, T-1]
        Returns (with leading batch dim = 1):
            actions, joint_pos_ref, joint_vel_ref, position_ref,
            rotation_ref_quat_w,
            default_joint_pos, action_scale, joint_stiffness, joint_damping
        """
        t = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)

        return (
            self.actor(self.encoders(self.normalizer(obs))),
            self.joint_pos_ref_tbl[t],          # (1, F, J)
            self.joint_vel_ref_tbl[t],          # (1, F, J)
            self.position_ref_tbl[t],           # (1, F, 3)
            self.rotation_ref_quat_w_tbl[t],    # (1, F, 4)  wxyz, world frame
            self.default_joint_pos.unsqueeze(0),
            self.action_scale.unsqueeze(0),
            self.joint_stiffness.unsqueeze(0),
            self.joint_damping.unsqueeze(0),
        )

    def export(self, path, filename: str = "policy.onnx"):
        self.to("cpu")
        obs = torch.zeros(1, self.num_obs)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos_ref",
                "joint_vel_ref",
                "position_ref",
                "rotation_ref_quat_w",
                "default_joint_pos",
                "action_scale",
                "joint_stiffness",
                "joint_damping",
            ],
            dynamic_axes={},
        )
