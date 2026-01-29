import argparse
import functools
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import torch
import tqdm

import pytorch_kinematics as pk


def convert_file(src_file, tgt_file):
    """Load from bm source file"""
    motion_data = np.load(src_file)
    # joint order in bm_for_x2t2d5's motion_npz is already the same as IsaacLab
    joint_names = [
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'waist_yaw_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'waist_pitch_joint',
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'waist_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_ankle_pitch_joint',
        'right_ankle_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_ankle_roll_joint',
        'right_ankle_roll_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_elbow_joint',
        'right_elbow_joint',
        'left_wrist_yaw_joint',
        'right_wrist_yaw_joint',
        'left_wrist_pitch_joint',
        'right_wrist_pitch_joint',
        'left_wrist_roll_joint',
        'right_wrist_roll_joint'
    ]
    
    # pack the file and store
    np.savez(
        tgt_file,
        framerate=motion_data["fps"],
        joint_names=joint_names,
        joint_pos=motion_data["joint_pos"],
        base_pos_w=motion_data["body_pos_w"][:, 0],
        base_quat_w=motion_data["body_quat_w"][:, 0],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input file")
    parser.add_argument("--tgt", type=str, default=None, help="Target file")
    
    args = parser.parse_args()
    if args.tgt is None:
        args.tgt = args.src.replace(".npz", "_instinct_retargetted.npz")
    
    convert_file(args.src, args.tgt)


if __name__ == "__main__":
    main()
