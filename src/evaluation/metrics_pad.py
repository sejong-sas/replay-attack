def compute_pad_metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # attack=1, real=0
    # APCER = attack를 real로 잘못 본 비율 = FN / (TP + FN)
    apcer_den = tp + fn
    apcer = fn / apcer_den if apcer_den > 0 else 0.0

    # BPCER = real을 attack으로 잘못 본 비율 = FP / (TN + FP)
    bpcer_den = tn + fp
    bpcer = fp / bpcer_den if bpcer_den > 0 else 0.0

    acer = (apcer + bpcer) / 2.0
    hter = acer

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