import os
import cv2
import pandas as pd

METADATA_CSV = "/home/saslab01/Desktop/replay_pad/metadata/replayattack_metadata.csv"
FRAME_OUTPUT_ROOT = "/home/saslab01/Desktop/replay_pad/frames/replay_attack"
FRAME_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv"

# 각 비디오에서 몇 장을 추출할지
FRAMES_PER_VIDEO = 20


def sample_frame_indices(total_frames, num_samples):
    """
    video 전체 구간에서 균등하게 num_samples개 인덱스를 뽑는다.
    """
    if total_frames <= 0:
        return []

    if total_frames <= num_samples:
        return list(range(total_frames))

    indices = []
    for i in range(num_samples):
        idx = round(i * (total_frames - 1) / (num_samples - 1))
        indices.append(idx)
    return sorted(list(set(indices)))


def extract_sampled_frames(video_path, save_dir, num_samples=20):
    """
    비디오를 열어서 균등 샘플링한 frame들을 저장한다.
    return:
        saved_frame_paths, sampled_indices
    """
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, num_samples)

    saved_paths = []
    saved_indices = []

    current_idx = 0
    target_set = set(frame_indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in target_set:
            out_name = f"frame_{current_idx:05d}.jpg"
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, frame)

            saved_paths.append(out_path)
            saved_indices.append(current_idx)

        current_idx += 1

    cap.release()
    return saved_paths, saved_indices


def main():
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"[ERROR] Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    rows = []

    print(f"[INFO] Loaded metadata: {METADATA_CSV}")
    print(f"[INFO] Total videos: {len(df)}")

    for i, row in df.iterrows():
        video_id = row["video_id"]
        video_path = row["video_path"]
        split = row["split"]
        label = row["label"]
        label_name = row["label_name"]
        attack_type = row["attack_type"]
        support_type = row["support_type"]
        environment = row["environment"]
        client_id = row["client_id"]
        dataset_name = row["dataset_name"]

        save_dir = os.path.join(FRAME_OUTPUT_ROOT, split, video_id)

        frame_paths, frame_indices = extract_sampled_frames(
            video_path=video_path,
            save_dir=save_dir,
            num_samples=FRAMES_PER_VIDEO
        )

        if len(frame_paths) == 0:
            print(f"[WARN] No frames extracted: {video_id}")
            continue

        for fp, frame_idx in zip(frame_paths, frame_indices):
            rows.append({
                "frame_path": fp,
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "attack_type": attack_type,
                "split": split,
                "frame_idx": frame_idx,
                "environment": environment,
                "support_type": support_type,
                "client_id": client_id,
                "dataset_name": dataset_name,
            })

        if (i + 1) % 50 == 0:
            print(f"[INFO] Processed {i + 1}/{len(df)} videos")

    frame_df = pd.DataFrame(rows)
    frame_df = frame_df.sort_values(by=["split", "video_id", "frame_idx"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(FRAME_INDEX_CSV), exist_ok=True)
    frame_df.to_csv(FRAME_INDEX_CSV, index=False)

    print()
    print(f"[INFO] Saved frame index CSV -> {FRAME_INDEX_CSV}")
    print(f"[INFO] Total frames indexed: {len(frame_df)}")
    print()
    print("[INFO] Frames per split")
    print(frame_df.groupby("split").size())
    print()
    print("[INFO] Sample rows")
    print(frame_df.head(10))


if __name__ == "__main__":
    main()