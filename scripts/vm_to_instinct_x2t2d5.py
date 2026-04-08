"""This is a tool that convert robot base pose to another frame on given robot chain"""

import argparse
import functools
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import torch
import tqdm
import h5py

import pytorch_kinematics as pk


def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        path: The current path in the HDF5 file.
    
    Returns:
        A nested dictionary with the data.
    """
    result = {}
    
    # Get the current group
    if path == "/":
        current_group = h5file
    else:
        current_group = h5file[path]
    
    # Load datasets and groups
    for key in current_group.keys():
        if path == "/":
            key_path = key
        else:
            key_path = f"{path}/{key}"
            
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path)
        else:
            result[key] = h5file[key_path][:]
    
    # Load attributes of the current group
    for attr_key, attr_value in current_group.attrs.items():
        result[attr_key] = attr_value

    return result


def convert_file(
    src_tgt_pairs,
):
    src_file, tgt_file = src_tgt_pairs

    with h5py.File(src_file, "r") as f:
        motion_data = load_dict_from_hdf5(f)
    
    if "joint_names" in motion_data:
        joint_names = motion_data["joint_names"].tolist()
    elif "/joint_names" in motion_data:
        joint_names = motion_data["/joint_names"].tolist()
    else:
        raise ValueError(f"joint_names not found in {src_file}")
    
    joint_names = [
        name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
        for name in joint_names
    ]
    
    if "fps" in motion_data:
        framerate = motion_data["fps"]
    elif "/fps" in motion_data:
        framerate = motion_data["/fps"]
    else:
        raise ValueError(f"fps not found in {src_file}")
    
    if len(joint_names) == 23:
        # add missing wrist joints if needed
        joint_names = joint_names[:19] \
            + ["left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"] \
            + joint_names[19:] \
            + ["right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
        joint_pos = np.zeros((motion_data["joints"].shape[0], 29), dtype=np.float32)
        joint_pos[:, :19] = motion_data["joints"][:, :19]
        joint_pos[:, 22:26] = motion_data["joints"][:, 19:]
    else:
        joint_pos = motion_data["joints"]  # (N, 29)

    # pack the file and store
    np.savez(
        tgt_file,
        framerate=framerate,
        joint_names=joint_names,
        joint_pos=joint_pos,  # (N, 29)
        base_pos_w=motion_data["root_pos"],  # (N, 3)
        base_quat_w=motion_data["root_quat"][..., [3, 0, 1, 2]],  # (N, 4), wxyz order
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Input file or folder")
    parser.add_argument("--tgt", type=str, help="Target file or folder. Structure preserved if folder")
    parser.add_argument("--num_cpus", default=10)

    args = parser.parse_args()

    # walk through the source folder and make folders in target folder if needed
    src_tgt_pairs = []
    if os.path.isfile(args.src):
        src_tgt_pairs.append((args.src, args.tgt))
    else:
        if not os.path.exists(args.tgt):
            os.makedirs(args.tgt, exist_ok=True)
        for root, _, filenames in os.walk(args.src):
            target_dirpath = os.path.join(args.tgt, os.path.relpath(root, args.src))
            os.makedirs(target_dirpath, exist_ok=True)
            for filename in filenames:
                if not filename.endswith(".h5"):
                    continue
                src_tgt_pairs.append(
                    (
                        os.path.join(root, filename),
                        os.path.join(target_dirpath, filename.replace(".h5", "_retargeted.npz")),
                    )
                )

    with mp.Pool(args.num_cpus) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    functools.partial(
                        convert_file,
                    ),
                    src_tgt_pairs,
                ),
                total=len(src_tgt_pairs),
            )
        )


if __name__ == "__main__":
    main()
