import pandas as pd
from src.evaluation.metrics_pad import compute_pad_metrics_from_counts


def aggregate_frame_to_video(frame_pred_csv, out_video_csv=None):
    df = pd.read_csv(frame_pred_csv)

    video_df = (
        df.groupby("video_id", as_index=False)
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

    if out_video_csv is not None:
        video_df.to_csv(out_video_csv, index=False)

    return video_df


def apply_threshold_and_compute_metrics(video_df, threshold):
    pred = (video_df["score"] >= threshold).astype(int)

    label = video_df["label"].astype(int)

    tp = int(((pred == 1) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())

    return compute_pad_metrics_from_counts(tp, tn, fp, fn)


def search_best_threshold(video_df, step=0.001):
    best_result = None
    best_threshold = None
    best_acer = float("inf")

    threshold = 0.0
    while threshold <= 1.0:
        metrics = apply_threshold_and_compute_metrics(video_df, threshold)

        if metrics["acer"] < best_acer:
            best_acer = metrics["acer"]
            best_threshold = threshold
            best_result = metrics

        threshold += step

    return best_threshold, best_result