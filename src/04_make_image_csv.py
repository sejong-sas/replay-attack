from pathlib import Path
import pandas as pd
import re

INPUT_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/baseline_frames.csv")
OUTPUT_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/baseline_image_labels.csv")

def extract_subject_id_from_path(path_str):
    match = re.search(r"(client\d+)", path_str.lower())
    if match:
        return match.group(1)
    return "unknown"

def main():
    df = pd.read_csv(INPUT_CSV)
    rows = []

    for _, row in df.iterrows():
        frame_dir = Path(row["sampled_frame_dir"])
        split = row["split"]
        label_binary = row["label_binary"]
        attack_type = row["attack_type"]
        video_path = row["video_path"]

        subject_id = extract_subject_id_from_path(video_path)

        if frame_dir.exists():
            for img_path in sorted(frame_dir.glob("*.jpg")):
                rows.append({
                    "split": split,
                    "subject_id": subject_id,
                    "image_path": str(img_path),
                    "label_binary": label_binary,
                    "label": 0 if label_binary == "live" else 1,
                    "attack_type": attack_type,
                    "video_path": video_path,
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Total images: {len(out_df)}")
    print(out_df.head())

if __name__ == "__main__":
    main()