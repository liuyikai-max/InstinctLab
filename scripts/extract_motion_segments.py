import os
import subprocess
import joblib
import yaml


def extract_pkl_segment(pkl_path, segments_sec, output_dir):
    """按秒范围从pkl截取片段。segments_sec: [(start_s, end_s), ...]"""
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    data = joblib.load(pkl_path)

    for key, motion in data.items():
        fps = motion["fps"]
        print(f"Processing pkl key: {key}, fps: {fps}")

        for i, (start_s, end_s) in enumerate(segments_sec):
            start = int(round(start_s * fps))
            end = int(round(end_s * fps))

            total_len = motion["root_trans_offset"].shape[0]
            if start < 0 or end > total_len or start >= end:
                print(f"⚠️  跳过非法片段 ({start}, {end})，总长度 {total_len}")
                continue

            seg_data = {
                "root_trans_offset": motion["root_trans_offset"][start:end].copy(),
                "pose_aa": motion["pose_aa"][start:end].copy(),
                "dof": motion["dof"][start:end].copy(),
                "root_rot": motion["root_rot"][start:end].copy(),
                "fps": fps,
            }

            out_key = f"{key}_seg{i}_frames{start_s}-{end_s}"
            out_path = os.path.join(output_dir, f"{base_name}_seg{i}_frames{start_s}-{end_s}.pkl")
            joblib.dump({out_key: seg_data}, out_path)
            print(f"✅ pkl: {out_path}")


def extract_mp4_segment(mp4_path, segments_sec, output_dir):
    """用ffmpeg按秒范围截取mp4片段。"""
    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    for i, (start_s, end_s) in enumerate(segments_sec):
        out_name = f"{base_name}_seg{i}_frames{start_s}-{end_s}.mp4"
        out_path = os.path.join(output_dir, out_name)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_s), "-to", str(end_s),
            "-i", mp4_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"✅ mp4: {out_path}")


def process_yaml(yaml_path, output_dir):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    src_dir = cfg["dir"]
    os.makedirs(output_dir, exist_ok=True)

    for entry in cfg["data"]:
        mp4_name = entry["file"]
        base_name = os.path.splitext(mp4_name)[0]
        segments = [tuple(r) for r in entry["range"]]

        mp4_path = os.path.join(src_dir, mp4_name)
        pkl_path = os.path.join(src_dir, base_name + ".pkl")

        print(f"\n=== {base_name} ===")
        if os.path.exists(pkl_path):
            extract_pkl_segment(pkl_path, segments, output_dir)
        else:
            print(f"⚠️  缺少pkl: {pkl_path}")

        if os.path.exists(mp4_path):
            extract_mp4_segment(mp4_path, segments, output_dir)
        else:
            print(f"⚠️  缺少mp4: {mp4_path}")


if __name__ == "__main__":
    yaml_path = "/home/agiuser/projects/Instinct/InstinctLab/data/parkour_motion_x2/parkour_motion_x2.yaml"
    output_dir = "/home/agiuser/projects/Instinct/InstinctLab/data/parkour_motion_x2/segments"
    process_yaml(yaml_path, output_dir)
