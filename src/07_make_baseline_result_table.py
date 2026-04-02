from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

VIDEO_CSV = Path("/Users/youbin/Desktop/replay_pad/outputs/predictions/baseline_test_video_preds.csv")
OUT_CSV = Path("/Users/youbin/Desktop/replay_pad/outputs/tables/baseline_results.csv")

def main():
    df = pd.read_csv(VIDEO_CSV)

    acc = accuracy_score(df["label"], df["pred"])
    cm = confusion_matrix(df["label"], df["pred"])

    result_df = pd.DataFrame([{
        "model": "ResNet18_frame_baseline",
        "test_video_accuracy": round(acc, 4),
        "tn": int(cm[0,0]),
        "fp": int(cm[0,1]),
        "fn": int(cm[1,0]),
        "tp": int(cm[1,1]),
    }])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(result_df)

if __name__ == "__main__":
    main()