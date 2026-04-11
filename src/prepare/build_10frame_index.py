import os
import pandas as pd

FRAME_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv"
OUTPUT_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_10frame_index.csv"

WINDOW_SIZE = 10
STRIDE = 1


def build_sequence_index(df, window_size=10, stride=1):
    rows = []

    grouped = df.groupby("video_id")

    for video_id, g in grouped:
        g = g.sort_values("frame_idx").reset_index(drop=True)

        if len(g) < window_size:
            continue

        for start in range(0, len(g) - window_size + 1, stride):
            sub = g.iloc[start:start + window_size]

            frame_paths = sub["frame_path"].tolist()
            frame_indices = sub["frame_idx"].tolist()

            rows.append({
                "clip_id": f"{video_id}__seq{start:03d}",
                "video_id": video_id,
                "label": int(sub["label"].iloc[0]),
                "label_name": sub["label_name"].iloc[0],
                "attack_type": sub["attack_type"].iloc[0],
                "split": sub["split"].iloc[0],
                "frame_paths": "|".join(frame_paths),
                "num_frames": window_size,
                "start_frame_idx": int(frame_indices[0]),
                "end_frame_idx": int(frame_indices[-1]),
                "environment": sub["environment"].iloc[0],
                "support_type": sub["support_type"].iloc[0],
                "client_id": sub["client_id"].iloc[0],
                "dataset_name": sub["dataset_name"].iloc[0],
            })

    return pd.DataFrame(rows)


def main():
    if not os.path.exists(FRAME_INDEX_CSV):
        raise FileNotFoundError(f"[ERROR] Not found: {FRAME_INDEX_CSV}")

    df = pd.read_csv(FRAME_INDEX_CSV)
    out_df = build_sequence_index(df, window_size=WINDOW_SIZE, stride=STRIDE)

    if len(out_df) == 0:
        raise ValueError("[ERROR] No 10-frame sequences were created")

    out_df = out_df.sort_values(["split", "video_id", "start_frame_idx"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saved 10-frame index -> {OUTPUT_CSV}")
    print(f"[INFO] Total sequences: {len(out_df)}")
    print()
    print("[INFO] Sequences by split")
    print(out_df.groupby("split").size())
    print()
    print("[INFO] Sample rows")
    print(out_df.head(10))


if __name__ == "__main__":
    main()