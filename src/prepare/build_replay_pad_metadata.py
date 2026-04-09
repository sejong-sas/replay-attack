from pathlib import Path
import pandas as pd

DATA_ROOT = Path("/Users/youbin/Desktop/replay_pad/data")
OUTPUT_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/replay_pad_metadata.csv")

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".m4v"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def collect_videos():
    rows = []

    for split in ["train", "devel", "test"]:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] split folder not found: {split_dir}")
            continue

        # real
        real_dir = split_dir / "real"
        if real_dir.exists():
            for path in sorted(real_dir.rglob("*")):
                if not is_video_file(path):
                    continue

                rows.append({
                    "video_id": path.stem,
                    "label": "real",
                    "attack_type": "none",
                    "split": split,
                    "video_path": str(path.resolve()),
                })

        # fixed
        fixed_dir = split_dir / "attack" / "fixed"
        if fixed_dir.exists():
            for path in sorted(fixed_dir.rglob("*")):
                if not is_video_file(path):
                    continue

                rows.append({
                    "video_id": path.stem,
                    "label": "attack",
                    "attack_type": "fixed",
                    "split": split,
                    "video_path": str(path.resolve()),
                })

        # hand
        hand_dir = split_dir / "attack" / "hand"
        if hand_dir.exists():
            for path in sorted(hand_dir.rglob("*")):
                if not is_video_file(path):
                    continue

                rows.append({
                    "video_id": path.stem,
                    "label": "attack",
                    "attack_type": "hand",
                    "split": split,
                    "video_path": str(path.resolve()),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No video files found. Check dataset path and file extensions."
        )

    df = df.sort_values(["split", "label", "attack_type", "video_id"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] metadata saved -> {OUTPUT_CSV}")
    print(f"[INFO] total videos: {len(df)}")
    print("\n[INFO] split counts")
    print(df["split"].value_counts())
    print("\n[INFO] label counts")
    print(df["label"].value_counts())
    print("\n[INFO] split x attack_type")
    print(df.groupby(["split", "attack_type"]).size())


if __name__ == "__main__":
    collect_videos()