from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_frame_dataset import ReplayPadFrameDataset
from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier
from src.engine.pad_metrics import (
    search_best_threshold,
    compute_pad_metrics_from_labels,
    aggregate_video_scores,
    compute_subgroup_metrics,
)

ROOT = Path("/Users/youbin/Desktop/replay_pad")
FRAME_CSV = ROOT / "frame_index" / "replay_pad_1frame.csv"
CKPT_PATH = ROOT / "outputs" / "checkpoints" / "mobilenetv3_small_1frame_best.pth"

DEVEL_PRED_CSV = ROOT / "outputs" / "predictions" / "mobilenetv3_small_devel_frame_predictions.csv"
TEST_PRED_CSV = ROOT / "outputs" / "predictions" / "mobilenetv3_small_test_frame_predictions.csv"
DEVEL_VIDEO_CSV = ROOT / "outputs" / "predictions" / "mobilenetv3_small_devel_video_predictions.csv"
TEST_VIDEO_CSV = ROOT / "outputs" / "predictions" / "mobilenetv3_small_test_video_predictions.csv"

RESULT_JSON = ROOT / "outputs" / "results" / "mobilenetv3_small_eval_results.json"
FRAME_SUBGROUP_CSV = ROOT / "outputs" / "results" / "mobilenetv3_small_test_frame_subgroups.csv"
VIDEO_SUBGROUP_CSV = ROOT / "outputs" / "results" / "mobilenetv3_small_test_video_subgroups.csv"

BATCH_SIZE = 32
IMAGE_SIZE = 224
PRETRAINED = False


def build_loader(split):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = ReplayPadFrameDataset(FRAME_CSV, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader


def collect_predictions(model, loader, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i in range(len(probs)):
                rows.append({
                    "frame_path": batch["frame_path"][i],
                    "video_id": batch["video_id"][i],
                    "label": batch["label"][i].item() if hasattr(batch["label"], "__len__") else int(batch["label"]),
                    "attack_type": batch["attack_type"][i],
                    "split": batch["split"][i],
                    "score": float(probs[i]),
                })

    df = pd.DataFrame(rows)
    df["label"] = df["label"].map({0: "real", 1: "attack"})
    df["label_binary"] = df["label"].map({"real": 0, "attack": 1})
    return df


def apply_threshold(df, threshold):
    df = df.copy()
    df["pred"] = (df["score"] >= threshold).astype(int)
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = MobileNetV3SmallBinaryClassifier(pretrained=PRETRAINED).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] checkpoint loaded -> {CKPT_PATH}")

    devel_loader = build_loader("devel")
    test_loader = build_loader("test")

    devel_df = collect_predictions(model, devel_loader, device)
    test_df = collect_predictions(model, test_loader, device)

    DEVEL_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    devel_df.to_csv(DEVEL_PRED_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PRED_CSV, index=False, encoding="utf-8-sig")

    best = search_best_threshold(devel_df, score_col="score", label_col="label_binary")
    threshold = best["threshold"]
    print(f"[INFO] selected threshold on devel: {threshold:.4f}")

    test_frame_df = apply_threshold(test_df, threshold)
    test_frame_metrics = compute_pad_metrics_from_labels(
        test_frame_df["label_binary"], test_frame_df["pred"]
    )

    devel_video_df = aggregate_video_scores(devel_df, score_col="score")
    test_video_df = aggregate_video_scores(test_df, score_col="score")

    devel_video_df = apply_threshold(devel_video_df, threshold)
    test_video_df = apply_threshold(test_video_df, threshold)

    DEVEL_VIDEO_CSV.parent.mkdir(parents=True, exist_ok=True)
    devel_video_df.to_csv(DEVEL_VIDEO_CSV, index=False, encoding="utf-8-sig")
    test_video_df.to_csv(TEST_VIDEO_CSV, index=False, encoding="utf-8-sig")

    test_video_metrics = compute_pad_metrics_from_labels(
        test_video_df["label_binary"], test_video_df["pred"]
    )

    frame_subgroups = compute_subgroup_metrics(test_frame_df, pred_col="pred", label_col="label_binary")
    video_subgroups = compute_subgroup_metrics(test_video_df, pred_col="pred", label_col="label_binary")

    FRAME_SUBGROUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame_subgroups.to_csv(FRAME_SUBGROUP_CSV, index=False, encoding="utf-8-sig")
    video_subgroups.to_csv(VIDEO_SUBGROUP_CSV, index=False, encoding="utf-8-sig")

    results = {
        "model": "MobileNetV3-Small",
        "initialization": "pretrained" if PRETRAINED else "random",
        "threshold_selected_on": "devel",
        "threshold": threshold,
        "devel_threshold_search_result": best,
        "test_frame_metrics": test_frame_metrics,
        "test_video_metrics": test_video_metrics,
    }

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[TEST FRAME-LEVEL]")
    for k, v in test_frame_metrics.items():
        print(f"{k}: {v}")

    print("\n[TEST VIDEO-LEVEL]")
    for k, v in test_video_metrics.items():
        print(f"{k}: {v}")

    print(f"\n[INFO] result json saved -> {RESULT_JSON}")


if __name__ == "__main__":
    main()