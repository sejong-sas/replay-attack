import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier
from src.evaluation.video_level_metrics import (
    search_best_threshold,
    apply_threshold_and_compute_metrics,
)

SEQ_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_5frame_index.csv"
CHECKPOINT_PATH = "/home/saslab01/Desktop/replay_pad/outputs/checkpoints/mobilenetv3_small_1frame_random_best.pth"

OUTPUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


def load_model():
    model = MobileNetV3SmallBinaryClassifier(
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def predict_one_image(model, image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        score = torch.softmax(logits, dim=1)[:, 1].item()
    return score


def evaluate_sequence_scores(model, split):
    df = pd.read_csv(SEQ_INDEX_CSV)
    df = df[df["split"] == split].reset_index(drop=True)

    transform = get_transform()
    seq_scores = []

    for _, row in df.iterrows():
        frame_paths = row["frame_paths"].split("|")
        frame_scores = [predict_one_image(model, p, transform) for p in frame_paths]
        seq_score = sum(frame_scores) / len(frame_scores)
        seq_scores.append(seq_score)

    df["score"] = seq_scores

    out_csv = os.path.join(PRED_DIR, f"image_5frame_avg_{split}_sequence_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv} ({len(df)} rows)")
    return df, out_csv


def aggregate_sequence_to_video(seq_df, out_csv):
    video_df = (
        seq_df.groupby("video_id", as_index=False)
        .agg({
            "label": "first",
            "split": "first",
            "attack_type": "first",
            "support_type": "first",
            "environment": "first",
            "client_id": "first",
            "dataset_name": "first",
            "score": "mean",
        })
    )
    video_df.to_csv(out_csv, index=False)
    return video_df


def main():
    print(f"[INFO] Device: {DEVICE}")
    model = load_model()
    print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")

    devel_seq_df, _ = evaluate_sequence_scores(model, "devel")
    test_seq_df, _ = evaluate_sequence_scores(model, "test")

    devel_video_csv = os.path.join(PRED_DIR, "image_5frame_avg_devel_video_predictions.csv")
    test_video_csv = os.path.join(PRED_DIR, "image_5frame_avg_test_video_predictions.csv")

    devel_video_df = aggregate_sequence_to_video(devel_seq_df, devel_video_csv)
    test_video_df = aggregate_sequence_to_video(test_seq_df, test_video_csv)

    print(f"[INFO] Saved: {devel_video_csv}")
    print(f"[INFO] Saved: {test_video_csv}")
    print(f"[INFO] Devel videos: {len(devel_video_df)}")
    print(f"[INFO] Test videos: {len(test_video_df)}")

    best_threshold, devel_metrics = search_best_threshold(devel_video_df, step=0.001)
    test_metrics = apply_threshold_and_compute_metrics(test_video_df, best_threshold)

    result = {
        "model": "MobileNetV3-Small",
        "input_type": "5-frame average",
        "initialization": "random",
        "threshold_selected_on": "devel",
        "threshold": round(float(best_threshold), 6),
        "devel_threshold_search_result": devel_metrics,
        "test_video_metrics": test_metrics,
    }

    out_json = os.path.join(RESULT_DIR, "mobilenetv3_small_5frame_avg_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("[INFO] Final result")
    print(json.dumps(result, indent=2))
    print()
    print(f"[INFO] Saved result JSON -> {out_json}")


if __name__ == "__main__":
    main()