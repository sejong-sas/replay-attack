from pathlib import Path
from collections import Counter

DATA_ROOT = Path("/Users/youbin/Desktop/replay_pad/data")
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}

def main():
    all_videos = []
    split_counter = Counter()
    ext_counter = Counter()

    for path in DATA_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            all_videos.append(path)
            ext_counter[path.suffix.lower()] += 1

            parts = path.relative_to(DATA_ROOT).parts
            if len(parts) > 0:
                split_counter[parts[0]] += 1

    print(f"Total videos: {len(all_videos)}")
    print("Videos by split:")
    for k, v in split_counter.items():
        print(f"  {k}: {v}")

    print("Extensions:")
    for k, v in ext_counter.items():
        print(f"  {k}: {v}")

    print("\nSample paths:")
    for p in all_videos[:20]:
        print(p)

if __name__ == "__main__":
    main()