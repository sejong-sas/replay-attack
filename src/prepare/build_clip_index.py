import os
import json
import argparse
from typing import List, Dict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build clip index CSV from unified frame index CSV.")
    parser.add_argument(
        "--frame_csv",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv",
        help="Unified frame index CSV path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/clip_index",
        help="Directory to save split-wise clip index CSVs",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=20,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="replay-attack",
        help="Dataset name",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, csv_path: str):
    required_cols = [
        "frame_path",
        "video_id",
        "label",
        "attack_type",
        "split",
        "frame_idx",
        "environment",
        "support_type",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Missing required columns in {csv_path}: {missing}\n"
            f"Required columns: {required_cols}"
        )


def build_clip_rows_for_video(
    video_df: pd.DataFrame,
    clip_len: int,
    stride: int,
    dataset_name: str,
) -> List[Dict]:
    video_df = video_df.sort_values("frame_idx").reset_index(drop=True)
    num_frames = len(video_df)

    if num_frames < clip_len:
        return []

    starts = list(range(0, num_frames - clip_len + 1, stride))
    last_possible_start = num_frames - clip_len
    if starts[-1] != last_possible_start:
        starts.append(last_possible_start)

    rows = []

    first = video_df.iloc[0]
    video_id = first["video_id"]
    split = first["split"]
    label = first["label"]
    attack_type = first["attack_type"]
    environment = first["environment"]
    support_type = first["support_type"]

    for clip_idx, start in enumerate(starts):
        clip_df = video_df.iloc[start : start + clip_len]
        frame_paths = clip_df["frame_path"].tolist()
        start_frame_idx = int(clip_df.iloc[0]["frame_idx"])
        end_frame_idx = int(clip_df.iloc[-1]["frame_idx"])

        clip_id = f"{video_id}__clip{clip_len}__start{start_frame_idx:04d}__idx{clip_idx:03d}"

        rows.append(
            {
                "clip_id": clip_id,
                "video_id": video_id,
                "label": label,
                "attack_type": attack_type,
                "split": split,
                "frame_paths": json.dumps(frame_paths, ensure_ascii=False),
                "num_frames": len(frame_paths),
                "start_frame_idx": start_frame_idx,
                "end_frame_idx": end_frame_idx,
                "environment": environment,
                "support_type": support_type,
                "dataset_name": dataset_name,
            }
        )

    return rows


def build_clip_index_for_split(
    split_df: pd.DataFrame,
    clip_len: int,
    stride: int,
    dataset_name: str,
) -> pd.DataFrame:
    all_rows = []

    grouped = split_df.groupby("video_id", sort=False)
    for _, video_df in grouped:
        clip_rows = build_clip_rows_for_video(
            video_df=video_df,
            clip_len=clip_len,
            stride=stride,
            dataset_name=dataset_name,
        )
        all_rows.extend(clip_rows)

    if len(all_rows) == 0:
        raise ValueError(
            f"[ERROR] No clip rows were created for split={split_df['split'].iloc[0]} "
            f"with clip_len={clip_len}. Check number of frames per video."
        )

    clip_df = pd.DataFrame(all_rows)

    expected_cols = [
        "clip_id",
        "video_id",
        "label",
        "attack_type",
        "split",
        "frame_paths",
        "num_frames",
        "start_frame_idx",
        "end_frame_idx",
        "environment",
        "support_type",
        "dataset_name",
    ]
    return clip_df[expected_cols]


def main():
    args = parse_args()

    if not os.path.exists(args.frame_csv):
        raise FileNotFoundError(f"[ERROR] Frame index CSV not found: {args.frame_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.frame_csv)
    validate_columns(df, args.frame_csv)

    print("=" * 80)
    print("[INFO] Build clip index from unified frame CSV")
    print(f"[INFO] frame_csv   : {args.frame_csv}")
    print(f"[INFO] clip_len    : {args.clip_len}")
    print(f"[INFO] stride      : {args.stride}")
    print(f"[INFO] total rows   : {len(df)}")
    print(f"[INFO] total videos : {df['video_id'].nunique()}")
    print("=" * 80)

    for split_name in ["train", "devel", "test"]:
        split_df = df[df["split"] == split_name].copy()

        if len(split_df) == 0:
            raise ValueError(f"[ERROR] No rows found for split={split_name}")

        clip_df = build_clip_index_for_split(
            split_df=split_df,
            clip_len=args.clip_len,
            stride=args.stride,
            dataset_name=args.dataset_name,
        )

        out_path = os.path.join(
            args.output_dir,
            f"replayattack_clip{args.clip_len}_{split_name}.csv"
        )
        clip_df.to_csv(out_path, index=False)

        print(f"[INFO] Saved {split_name} clip index -> {out_path}")
        print(f"[INFO] {split_name} clips  : {len(clip_df)}")
        print(f"[INFO] {split_name} videos : {clip_df['video_id'].nunique()}")
        print(f"[INFO] {split_name} label counts:")
        print(clip_df["label"].value_counts(dropna=False))
        print("-" * 80)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()