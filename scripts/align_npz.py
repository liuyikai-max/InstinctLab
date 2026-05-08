import argparse
import numpy as np


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conj(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_rotate(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def align(a_path, b_path, m, n, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0):
    A = dict(np.load(a_path, allow_pickle=True))
    B = dict(np.load(b_path, allow_pickle=True))

    a_pos = A["base_pos_w"].astype(np.float64)
    a_quat = A["base_quat_w"].astype(np.float64)
    b_pos_ref = B["base_pos_w"][n].astype(np.float64)
    b_quat_ref = B["base_quat_w"][n].astype(np.float64)

    # dq such that dq * a_quat[m] = b_quat_ref
    dq = quat_mul(b_quat_ref, quat_conj(a_quat[m]))
    dq = dq / np.linalg.norm(dq)

    # dp such that dq rotates a_pos[m] then +dp = b_pos_ref
    dp = b_pos_ref - quat_rotate(dq, a_pos[m])

    new_pos = quat_rotate(np.broadcast_to(dq, a_quat.shape), a_pos) + dp
    new_quat = quat_mul(np.broadcast_to(dq, a_quat.shape), a_quat)
    new_quat = new_quat / np.linalg.norm(new_quat, axis=-1, keepdims=True)

    # manual fine-tune: rotate by dyaw around reference point b_pos_ref (z-axis),
    # then translate by (dx, dy, dz)
    if dyaw != 0.0 or dx != 0.0 or dy != 0.0 or dz != 0.0:
        half = 0.5 * dyaw
        yaw_q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        pivot = b_pos_ref
        rel = new_pos - pivot
        rel_rot = quat_rotate(np.broadcast_to(yaw_q, new_quat.shape), rel)
        new_pos = rel_rot + pivot + np.array([dx, dy, dz])
        new_quat = quat_mul(np.broadcast_to(yaw_q, new_quat.shape), new_quat)
        new_quat = new_quat / np.linalg.norm(new_quat, axis=-1, keepdims=True)

    A["base_pos_w"] = new_pos.astype(A["base_pos_w"].dtype)
    A["base_quat_w"] = new_quat.astype(A["base_quat_w"].dtype)

    np.savez(a_path, **A)
    print(f"Aligned {a_path}[{m}] to {b_path}[{n}] (dx={dx}, dy={dy}, dz={dz}, dyaw={dyaw})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="npz file A (will be overwritten)")
    parser.add_argument("b", help="npz file B (reference)")
    parser.add_argument("-m", type=int, default=0, help="frame index in A to align")
    parser.add_argument("-n", type=int, default=0, help="frame index in B to align to")
    parser.add_argument("--dx", type=float, default=0.0, help="extra x offset (m)")
    parser.add_argument("--dy", type=float, default=0.0, help="extra y offset (m)")
    parser.add_argument("--dz", type=float, default=0.0, help="extra z offset (m)")
    parser.add_argument("--dyaw", type=float, default=0.0, help="extra yaw (rad) around reference point")
    args = parser.parse_args()
    align(args.a, args.b, args.m, args.n, args.dx, args.dy, args.dz, args.dyaw)