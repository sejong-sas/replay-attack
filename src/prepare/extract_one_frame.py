from pathlib import Path
import cv2
import pandas as pd

META_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/replay_pad_metadata.csv")
FRAME_ROOT = Path("/Users/youbin/Desktop/replay_pad/frames_1frame")
OUTPUT_CSV = Path("/Users/youbin/Desktop/replay_pad/frame_index/replay_pad_1frame.csv")


def extract_middle_frame(video_path: str, save_path: Path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, -1

    frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, -1

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), frame)
    if not ok:
        return False, -1

    return True, frame_idx


def main():
    df = pd.read_csv(META_CSV)
    rows = []

    for _, row in df.iterrows():
        split = row["split"]
        label = row["label"]
        attack_type = row["attack_type"]
        video_id = row["video_id"]
        video_path = row["video_path"]

        save_dir = FRAME_ROOT / split / label / attack_type
        save_path = save_dir / f"{video_id}.jpg"

        success, frame_idx = extract_middle_frame(video_path, save_path)

        if not success:
            print(f"[WARN] failed: {video_path}")
            continue

        rows.append({
            "frame_path": str(save_path.resolve()),
            "video_id": video_id,
            "label": label,
            "attack_type": attack_type,
            "split": split,
            "frame_idx": frame_idx,
        })

    out_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] frame index saved -> {OUTPUT_CSV}")
    print(f"[INFO] total extracted frames: {len(out_df)}")
    print("\n[INFO] split counts")
    print(out_df["split"].value_counts())
    print("\n[INFO] label counts")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()