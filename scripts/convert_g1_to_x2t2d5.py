"""Convert a retargeted motion NPZ into the x2t2d5 variant by flipping elbow joints."""

import argparse
import os

import numpy as np


def _derive_output_path(src_path: str) -> str:
	"""Create same-directory output path with the x2t2d5 suffix."""
	base = os.path.basename(src_path)
	if base.endswith("-retargeted.npz"):
		prefix = base[: -len("-retargeted.npz")]
		tgt_path = os.path.join(os.path.dirname(src_path), f"{prefix}-x2t2d5-retargeted.npz")
	elif base.endswith("_retargetted.npz"):
		prefix = base[: -len("_retargetted.npz")]
		tgt_path = os.path.join(os.path.dirname(src_path), f"{prefix}_x2t2d5_retargetted.npz")
	else:
		raise ValueError(f"Expected input file to end with -retargeted.npz or _retargetted.npz, got: {src_path}")

	return tgt_path


def convert_file(src_file: str) -> str:
	data = np.load(src_file, allow_pickle=True)
	out = {key: data[key] for key in data.files}

	joint_names_raw = out["joint_names"]
	joint_names = [
		name.decode("utf-8") if isinstance(name, (bytes, bytearray, np.bytes_)) else str(name)
		for name in joint_names_raw.tolist()
	]
	joints = np.array(out["joint_pos"], copy=True)
	left_elbow_idx = joint_names.index("left_elbow_joint")
	right_elbow_idx = joint_names.index("right_elbow_joint")
	joints[:, left_elbow_idx] *= -1
	joints[:, right_elbow_idx] *= -1
	out["joint_pos"] = joints
	out["joint_names"] = joint_names_raw

	tgt_file = _derive_output_path(src_file)
	np.savez(tgt_file, **out)
	return tgt_file


def main():
	parser = argparse.ArgumentParser(description="Flip elbow joints for x2t2d5 retargeted motions.")
	parser.add_argument("--src", required=True, type=str, help="Input NPZ file, must end with retargeted.npz")
	args = parser.parse_args()

	tgt_file = convert_file(args.src)
	print(tgt_file)


if __name__ == "__main__":
	main()
