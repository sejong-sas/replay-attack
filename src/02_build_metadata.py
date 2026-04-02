from pathlib import Path
import csv

DATA_ROOT = Path("/Users/youbin/Desktop/replay_pad/data")
OUT_CSV = Path("/Users/youbin/Desktop/replay_pad/metadata/replayattack_metadata.csv")
VIDEO_EXTS = {".mov"}

def infer_label_and_attack(path_parts):
    text = "/".join(path_parts).lower()

    # binary label
    if "real" in text:
        label_binary = "live"
    else:
        label_binary = "spoof"

    # attack type
    if "print" in text:
        attack_type = "print"
    elif "replay" in text:
        attack_type = "replay"
    elif "real" in text:
        attack_type = "real"
    else:
        attack_type = "unknown"

    return label_binary, attack_type

import re
def extract_subject_id(path_parts):
    text = "/".join(path_parts).lower()
    match = re.search(r"(client\d+)", text)
    if match:
        return match.group(1)
    return "unknown"

def main():
    rows = []

    for path in DATA_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            rel = path.relative_to(DATA_ROOT)
            parts = rel.parts

            split = parts[0] if len(parts) > 0 else "unknown"
            label_binary, attack_type = infer_label_and_attack(parts)
            subject_id = extract_subject_id(parts)

            rows.append({
                "split": split,
                "subject_id": subject_id,
                "video_path": str(path),
                "label_binary": label_binary,
                "attack_type": attack_type,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "subject_id",
                "video_path",
                "label_binary",
                "attack_type",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metadata to: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")

if __name__ == "__main__":
    main()