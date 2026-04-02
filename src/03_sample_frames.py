from pathlib import Path
import cv2
import pandas as pd

META_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/replayattack_metadata.csv")
OUT_ROOT = Path("/Users/youbin/Desktop/replay_pad/interim/sampled_frames")

NUM_SAMPLED_FRAMES = 8

def sample_indices(num_frames, num_samples):
    if num_frames <= 0:
        return []
    if num_frames < num_samples:
        return list(range(num_frames))
    step = (num_frames - 1) / (num_samples - 1)
    return [round(i * step) for i in range(num_samples)]

def get_video_id(video_path: str):
    return Path(video_path).stem

def save_sampled_frames(video_path, out_dir, num_samples=8):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total_frames, num_samples)

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    current_idx = 0
    target_set = set(indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            filename = out_dir / f"frame_{current_idx:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1

        current_idx += 1

    cap.release()
    return total_frames, saved

def main():
    df = pd.read_csv(META_CSV)

    sampled_dirs = []
    total_videos = len(df)

    for i, row in df.iterrows():
        split = row["split"]
        label = row["label_binary"]
        attack_type = row["attack_type"]
        video_path = Path(row["video_path"])
        video_id = get_video_id(row["video_path"])

        out_dir = OUT_ROOT / split / label / attack_type / video_id
        total_frames, saved = save_sampled_frames(video_path, out_dir, NUM_SAMPLED_FRAMES)

        sampled_dirs.append(str(out_dir))

        if (i + 1) % 50 == 0 or i == total_videos - 1:
            print(f"[{i+1}/{total_videos}] done")

    df["sampled_frame_dir"] = sampled_dirs
    out_csv = META_CSV.parent / "baseline_frames.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved updated metadata to: {out_csv}")

if __name__ == "__main__":
    main()