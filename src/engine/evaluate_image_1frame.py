import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_frame_dataset import ReplayPADFrameDataset
from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier
from src.evaluation.video_level_metrics import (
    aggregate_frame_to_video,
    search_best_threshold,
    apply_threshold_and_compute_metrics,
)

FRAME_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv"
CHECKPOINT_PATH = "/home/saslab01/Desktop/replay_pad/outputs/checkpoints/mobilenetv3_small_1frame_random_best.pth"

OUTPUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4


def build_loader(split):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = ReplayPADFrameDataset(
        csv_path=FRAME_INDEX_CSV,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return dataset, loader


def inference_and_save_frame_predictions(model, split):
    _, loader = build_loader(split)

    split_df = pd.read_csv(FRAME_INDEX_CSV)
    split_df = split_df[split_df["split"] == split].reset_index(drop=True)

    all_scores = []
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # spoof=1 score
            all_scores.extend(probs.detach().cpu().numpy().tolist())

    split_df["score"] = all_scores

    out_csv = os.path.join(PRED_DIR, f"image_1frame_random_{split}_frame_predictions.csv")
    split_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv} ({len(split_df)} rows)")

    pred_label_05 = (split_df["score"] >= 0.5).astype(int)
    frame_acc_05 = (pred_label_05 == split_df["label"]).mean()
    print(f"[INFO] {split} frame-level acc@0.5 = {frame_acc_05:.4f}")

    return out_csv


def main():
    print(f"[INFO] Device: {DEVICE}")

    model = MobileNetV3SmallBinaryClassifier(
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"[INFO] Loaded checkpoint: {CHECKPOINT_PATH}")

    devel_frame_csv = inference_and_save_frame_predictions(model, "devel")
    test_frame_csv = inference_and_save_frame_predictions(model, "test")

    devel_video_csv = os.path.join(PRED_DIR, "image_1frame_random_devel_video_predictions.csv")
    test_video_csv = os.path.join(PRED_DIR, "image_1frame_random_test_video_predictions.csv")

    devel_video_df = aggregate_frame_to_video(devel_frame_csv, devel_video_csv)
    test_video_df = aggregate_frame_to_video(test_frame_csv, test_video_csv)

    print(f"[INFO] Saved: {devel_video_csv}")
    print(f"[INFO] Saved: {test_video_csv}")
    print(f"[INFO] Devel videos: {len(devel_video_df)}")
    print(f"[INFO] Test videos: {len(test_video_df)}")

    best_threshold, devel_metrics = search_best_threshold(devel_video_df, step=0.001)
    test_metrics = apply_threshold_and_compute_metrics(test_video_df, best_threshold)

    result = {
        "model": "MobileNetV3-Small",
        "initialization": "random",
        "threshold_selected_on": "devel",
        "threshold": round(float(best_threshold), 6),
        "devel_threshold_search_result": devel_metrics,
        "test_video_metrics": test_metrics,
    }

    out_json = os.path.join(RESULT_DIR, "mobilenetv3_small_1frame_random_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("[INFO] Final result")
    print(json.dumps(result, indent=2))
    print()
    print(f"[INFO] Saved result JSON -> {out_json}")


if __name__ == "__main__":
    main()