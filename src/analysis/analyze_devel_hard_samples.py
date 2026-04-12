import os
import pandas as pd

PRED_CSV = "/home/saslab01/Desktop/replay_pad/outputs/predictions/cnn_lstm_clip10_devel_video_predictions_annotated.csv"
OUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs/analysis/devel_errors"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(PRED_CSV)

    if "threshold" not in df.columns:
        raise ValueError("[ERROR] threshold column not found in annotated prediction csv")

    threshold = float(df["threshold"].iloc[0])

    # threshold와의 절대 거리
    df["margin_to_threshold"] = (df["score"] - threshold).abs()

    # 클래스 기준 confidence margin
    # bona fide(label=0): score가 threshold보다 낮을수록 안전
    # spoof(label=1): score가 threshold보다 높을수록 안전
    df["class_confidence_margin"] = df.apply(
        lambda row: (threshold - row["score"]) if row["label"] == 0 else (row["score"] - threshold),
        axis=1
    )

    # decision boundary 근처일수록 hard sample
    df_sorted = df.sort_values("margin_to_threshold", ascending=True).reset_index(drop=True)

    top_n = 100
    hard_df = df_sorted.head(top_n).copy()

    hard_csv = os.path.join(OUT_DIR, "devel_hard_samples_top100.csv")
    hard_df.to_csv(hard_csv, index=False)

    print(f"[INFO] Threshold: {threshold:.6f}")
    print(f"[INFO] Saved hard samples -> {hard_csv} ({len(hard_df)} rows)")

    print("\n[INFO] Top-100 hard samples by support_type")
    print(hard_df["support_type"].value_counts(dropna=False))

    print("\n[INFO] Top-100 hard samples by environment")
    print(hard_df["environment"].value_counts(dropna=False))

    print("\n[INFO] Top-100 hard samples by attack_type")
    print(hard_df["attack_type"].value_counts(dropna=False))

    print("\n[INFO] Top-100 hard samples by label")
    print(hard_df["label"].value_counts(dropna=False))

    # subgroup summary 저장
    support_summary = hard_df.groupby(["support_type", "label"]).size().reset_index(name="count")
    env_summary = hard_df.groupby(["environment", "label"]).size().reset_index(name="count")
    attack_summary = hard_df.groupby(["attack_type", "label"]).size().reset_index(name="count")

    support_csv = os.path.join(OUT_DIR, "devel_hard_samples_support_summary.csv")
    env_csv = os.path.join(OUT_DIR, "devel_hard_samples_environment_summary.csv")
    attack_csv = os.path.join(OUT_DIR, "devel_hard_samples_attacktype_summary.csv")

    support_summary.to_csv(support_csv, index=False)
    env_summary.to_csv(env_csv, index=False)
    attack_summary.to_csv(attack_csv, index=False)

    print(f"\n[INFO] Saved support summary -> {support_csv}")
    print(f"[INFO] Saved environment summary -> {env_csv}")
    print(f"[INFO] Saved attack type summary -> {attack_csv}")

    # real hard sample top-20 별도 저장
    real_hard_df = hard_df[hard_df["label"] == 0].head(20).copy()
    real_hard_csv = os.path.join(OUT_DIR, "devel_real_hard_samples_top20.csv")
    real_hard_df.to_csv(real_hard_csv, index=False)
    print(f"[INFO] Saved real hard samples -> {real_hard_csv} ({len(real_hard_df)} rows)")

    # spoof hard sample top-20도 같이 저장
    spoof_hard_df = hard_df[hard_df["label"] == 1].head(20).copy()
    spoof_hard_csv = os.path.join(OUT_DIR, "devel_spoof_hard_samples_top20.csv")
    spoof_hard_df.to_csv(spoof_hard_csv, index=False)
    print(f"[INFO] Saved spoof hard samples -> {spoof_hard_csv} ({len(spoof_hard_df)} rows)")


if __name__ == "__main__":
    main()