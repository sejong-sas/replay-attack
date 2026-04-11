import os
import pandas as pd

FRAME_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv"
OUTPUT_CSV = "/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip10_index.csv"

CLIP_LEN = 10
STRIDE = 1


def main():
    df = pd.read_csv(FRAME_INDEX_CSV)
    rows = []

    for video_id, g in df.groupby("video_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        if len(g) < CLIP_LEN:
            continue

        for start in range(0, len(g) - CLIP_LEN + 1, STRIDE):
            sub = g.iloc[start:start + CLIP_LEN]

            rows.append({
                "clip_id": f"{video_id}__clip{start:03d}",
                "video_id": video_id,
                "label": int(sub["label"].iloc[0]),
                "label_name": sub["label_name"].iloc[0],
                "attack_type": sub["attack_type"].iloc[0],
                "split": sub["split"].iloc[0],
                "frame_paths": "|".join(sub["frame_path"].tolist()),
                "num_frames": CLIP_LEN,
                "start_frame_idx": int(sub["frame_idx"].iloc[0]),
                "end_frame_idx": int(sub["frame_idx"].iloc[-1]),
                "environment": sub["environment"].iloc[0],
                "support_type": sub["support_type"].iloc[0],
                "client_id": sub["client_id"].iloc[0],
                "dataset_name": sub["dataset_name"].iloc[0],
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["split", "video_id", "start_frame_idx"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saved -> {OUTPUT_CSV}")
    print(out_df.groupby("split").size())
    print(out_df.head())


if __name__ == "__main__":
    main()