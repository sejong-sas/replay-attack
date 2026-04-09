import numpy as np
import pandas as pd


def compute_confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def compute_pad_metrics_from_labels(y_true, y_pred):
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    # spoof = 1, bona fide(real) = 0
    num_attack = tp + fn
    num_real = tn + fp

    apcer = fn / num_attack if num_attack > 0 else 0.0
    bpcer = fp / num_real if num_real > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    hter = (apcer + bpcer) / 2.0

    return {
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "hter": hter,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def search_best_threshold(df, score_col="score", label_col="label_binary"):
    thresholds = np.linspace(0.0, 1.0, 1001)

    best = None
    best_acer = float("inf")

    y_true = df[label_col].values

    for th in thresholds:
        y_pred = (df[score_col].values >= th).astype(int)
        metrics = compute_pad_metrics_from_labels(y_true, y_pred)
        acer = metrics["acer"]

        if acer < best_acer:
            best_acer = acer
            best = {
                "threshold": float(th),
                **metrics
            }

    return best


def aggregate_video_scores(df, score_col="score"):
    agg = (
        df.groupby(["video_id", "label", "attack_type", "split"], as_index=False)
        .agg(score=(score_col, "mean"))
    )
    agg["label_binary"] = agg["label"].map({"real": 0, "attack": 1})
    return agg


def compute_subgroup_metrics(df, pred_col="pred", label_col="label_binary"):
    rows = []
    for subgroup in ["real", "fixed", "hand"]:
        if subgroup == "real":
            part = df[df["label"] == "real"].copy()
        else:
            part = df[df["attack_type"] == subgroup].copy()

        if len(part) == 0:
            continue

        metrics = compute_pad_metrics_from_labels(part[label_col], part[pred_col])
        rows.append({
            "subgroup_value": subgroup,
            **metrics
        })
    return pd.DataFrame(rows)