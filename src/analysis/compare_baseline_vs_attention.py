import json
import pandas as pd

BASELINE_JSON = "/home/saslab01/Desktop/replay_pad/outputs/results/cnn_lstm_clip10_eval_results.json"
ATTN_JSON = "/home/saslab01/Desktop/replay_pad/outputs/results/cnn_lstm_attention_clip10_eval_results.json"
OUT_CSV = "/home/saslab01/Desktop/replay_pad/outputs/results/cnn_lstm_baseline_vs_attention_comparison.csv"


def load_result(path):
    with open(path, "r") as f:
        data = json.load(f)

    metrics = data["test_video_metrics"]
    return {
        "model": data["model"],
        "input_type": data.get("input_type", ""),
        "added_feature": data.get("added_feature", "None"),
        "threshold": data["threshold"],
        "accuracy": metrics["accuracy"],
        "apcer": metrics["apcer"],
        "bpcer": metrics["bpcer"],
        "acer": metrics["acer"],
        "hter": metrics["hter"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }


def main():
    rows = [
        load_result(BASELINE_JSON),
        load_result(ATTN_JSON),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(df)
    print(f"\n[INFO] Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()