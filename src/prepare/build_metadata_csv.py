import os
import re
import glob
import pandas as pd

DATA_ROOT = "/home/saslab01/Desktop/replay_pad/data"
OUTPUT_CSV = "/home/saslab01/Desktop/replay_pad/metadata/replayattack_metadata.csv"

VIDEO_EXTENSIONS = ("*.mov", "*.mp4", "*.avi", "*.mkv")


def find_video_files(root_dir):
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(video_files)


def parse_client_id(filename):
    """
    Example:
    attack_highdef_client003_session01_highdef_photo_fixed.mov
    real_client005_session01_webcam_authentication_adverse.mp4
    """
    m = re.search(r"client(\d+)", filename.lower())
    if m:
        return f"client{m.group(1)}"
    return "unknown"


def parse_environment(filename):
    filename = filename.lower()
    if "adverse" in filename:
        return "adverse"
    if "controlled" in filename:
        return "controlled"
    return "unknown"


def parse_attack_type(filename, label):
    """
    Replay-Attack에서는 파일명에 photo / video 가 들어가는 경우가 많음.
    real은 attack_type을 real로 둔다.
    """
    filename = filename.lower()

    if label == 0:
        return "real"

    if "photo" in filename:
        return "photo"
    if "video" in filename:
        return "video"

    return "unknown"


def build_video_id(split, label_name, support_type, filename_no_ext):
    safe_name = filename_no_ext.replace(" ", "_")
    return f"{split}__{label_name}__{support_type}__{safe_name}"


def infer_label_and_support(video_path):
    """
    path examples:
    /home/.../data/train/real/xxx.mov
    /home/.../data/train/attack/fixed/xxx.mov
    /home/.../data/train/attack/hand/xxx.mov
    """
    norm = video_path.replace("\\", "/").lower()

    if "/real/" in norm:
        return 0, "real", "real"
    elif "/attack/fixed/" in norm:
        return 1, "attack", "fixed"
    elif "/attack/hand/" in norm:
        return 1, "attack", "hand"
    else:
        raise ValueError(f"[ERROR] Cannot infer label/support_type from path: {video_path}")


def infer_split(video_path):
    norm = video_path.replace("\\", "/").lower()

    if "/train/" in norm:
        return "train"
    elif "/devel/" in norm:
        return "devel"
    elif "/test/" in norm:
        return "test"
    else:
        raise ValueError(f"[ERROR] Cannot infer split from path: {video_path}")


def main():
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"[ERROR] DATA_ROOT does not exist: {DATA_ROOT}")

    all_video_files = find_video_files(DATA_ROOT)

    if len(all_video_files) == 0:
        raise ValueError(f"[ERROR] No video files found under: {DATA_ROOT}")

    rows = []

    for video_path in all_video_files:
        filename = os.path.basename(video_path)
        filename_no_ext = os.path.splitext(filename)[0]

        split = infer_split(video_path)
        label, label_name, support_type = infer_label_and_support(video_path)
        client_id = parse_client_id(filename)
        environment = parse_environment(filename)
        attack_type = parse_attack_type(filename, label)

        video_id = build_video_id(split, label_name, support_type, filename_no_ext)

        rows.append({
            "video_id": video_id,
            "video_path": video_path,
            "split": split,
            "label": label,
            "label_name": label_name,
            "attack_type": attack_type,
            "support_type": support_type,
            "environment": environment,
            "client_id": client_id,
            "dataset_name": "replay-attack",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["split", "label", "support_type", "video_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saved metadata CSV -> {OUTPUT_CSV}")
    print()
    print("[INFO] Overall counts")
    print(df.groupby(["split", "label_name", "support_type"]).size())
    print()
    print("[INFO] Total videos by split")
    print(df.groupby(["split"]).size())
    print()
    print("[INFO] Sample rows")
    print(df.head(10))


if __name__ == "__main__":
    main()