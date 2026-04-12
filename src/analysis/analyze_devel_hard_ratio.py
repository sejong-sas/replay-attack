import os
import pandas as pd

DEVEL_PRED_CSV = "/home/saslab01/Desktop/replay_pad/outputs/predictions/cnn_lstm_clip10_devel_video_predictions_annotated.csv"
HARD_CSV = "/home/saslab01/Desktop/replay_pad/outputs/analysis/devel_errors/devel_hard_samples_top100.csv"

OUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs/analysis/devel_errors"
os.makedirs(OUT_DIR, exist_ok=True)


def make_ratio_rows_for_column(df_all, df_hard, subgroup_type, column_name, valid_values=None):
    rows = []

    if valid_values is None:
        values = sorted(df_all[column_name].dropna().unique().tolist())
    else:
        values = valid_values

    for value in values:
        total_count = int((df_all[column_name] == value).sum())
        hard_count = int((df_hard[column_name] == value).sum())
        hard_ratio = (hard_count / total_count) if total_count > 0 else 0.0

        rows.append({
            "subgroup_type": subgroup_type,
            "subgroup": value,
            "total_count": total_count,
            "hard_count": hard_count,
            "hard_ratio": round(hard_ratio, 6),
        })

    return rows


def make_ratio_rows_for_filtered_column(
    df_all, df_hard,
    subgroup_type,
    filter_col, filter_val,
    target_col, valid_values=None
):
    rows = []

    df_all_f = df_all[df_all[filter_col] == filter_val].copy()
    df_hard_f = df_hard[df_hard[filter_col] == filter_val].copy()

    if valid_values is None:
        values = sorted(df_all_f[target_col].dropna().unique().tolist())
    else:
        values = valid_values

    for value in values:
        total_count = int((df_all_f[target_col] == value).sum())
        hard_count = int((df_hard_f[target_col] == value).sum())
        hard_ratio = (hard_count / total_count) if total_count > 0 else 0.0

        rows.append({
            "subgroup_type": subgroup_type,
            "subgroup": value,
            "total_count": total_count,
            "hard_count": hard_count,
            "hard_ratio": round(hard_ratio, 6),
        })

    return rows


def main():
    df_all = pd.read_csv(DEVEL_PRED_CSV)
    df_hard = pd.read_csv(HARD_CSV)

    print(f"[INFO] Loaded all devel predictions: {len(df_all)}")
    print(f"[INFO] Loaded hard samples: {len(df_hard)}")

    ratio_rows = []

    # 1) label 기준
    ratio_rows += make_ratio_rows_for_column(
        df_all, df_hard,
        subgroup_type="label",
        column_name="label",
        valid_values=[0, 1],
    )

    # 2) environment 기준 (전체)
    ratio_rows += make_ratio_rows_for_column(
        df_all, df_hard,
        subgroup_type="environment",
        column_name="environment",
        valid_values=["controlled", "adverse"],
    )

    # 3) support_type 기준 (spoof only)
    # real은 support_type이 real이라서 fixed/hand 비교는 spoof에서만 해야 함
    ratio_rows += make_ratio_rows_for_filtered_column(
        df_all, df_hard,
        subgroup_type="support_type_spoof_only",
        filter_col="label",
        filter_val=1,
        target_col="support_type",
        valid_values=["fixed", "hand"],
    )

    # 4) attack_type 기준 (spoof only)
    # real attack_type=real이라 photo/video 비교는 spoof에서만 해야 함
    ratio_rows += make_ratio_rows_for_filtered_column(
        df_all, df_hard,
        subgroup_type="attack_type_spoof_only",
        filter_col="label",
        filter_val=1,
        target_col="attack_type",
        valid_values=["photo", "video"],
    )

    # 5) environment 기준 (spoof only)
    ratio_rows += make_ratio_rows_for_filtered_column(
        df_all, df_hard,
        subgroup_type="environment_spoof_only",
        filter_col="label",
        filter_val=1,
        target_col="environment",
        valid_values=["controlled", "adverse"],
    )

    # 6) environment 기준 (real only)
    ratio_rows += make_ratio_rows_for_filtered_column(
        df_all, df_hard,
        subgroup_type="environment_real_only",
        filter_col="label",
        filter_val=0,
        target_col="environment",
        valid_values=["controlled", "adverse"],
    )

    ratio_df = pd.DataFrame(ratio_rows)

    out_csv = os.path.join(OUT_DIR, "devel_hard_ratio_summary.csv")
    ratio_df.to_csv(out_csv, index=False)

    print(f"[INFO] Saved hard ratio summary -> {out_csv}")
    print()
    print(ratio_df.to_string(index=False))

    # ----------------------------
    # visualization candidate lists
    # ----------------------------

    # real hard top-10
    real_top10 = df_hard[df_hard["label"] == 0].sort_values("margin_to_threshold", ascending=True).head(10).copy()
    real_top10_csv = os.path.join(OUT_DIR, "devel_real_hard_top10_for_visual_check.csv")
    real_top10.to_csv(real_top10_csv, index=False)

    # spoof hard top-10
    spoof_top10 = df_hard[df_hard["label"] == 1].sort_values("margin_to_threshold", ascending=True).head(10).copy()
    spoof_top10_csv = os.path.join(OUT_DIR, "devel_spoof_hard_top10_for_visual_check.csv")
    spoof_top10.to_csv(spoof_top10_csv, index=False)

    # hand spoof hard top-5
    hand_spoof_top5 = df_hard[
        (df_hard["label"] == 1) &
        (df_hard["support_type"] == "hand")
    ].sort_values("margin_to_threshold", ascending=True).head(5).copy()
    hand_spoof_top5_csv = os.path.join(OUT_DIR, "devel_hand_spoof_hard_top5_for_visual_check.csv")
    hand_spoof_top5.to_csv(hand_spoof_top5_csv, index=False)

    # adverse spoof hard top-5
    adverse_spoof_top5 = df_hard[
        (df_hard["label"] == 1) &
        (df_hard["environment"] == "adverse")
    ].sort_values("margin_to_threshold", ascending=True).head(5).copy()
    adverse_spoof_top5_csv = os.path.join(OUT_DIR, "devel_adverse_spoof_hard_top5_for_visual_check.csv")
    adverse_spoof_top5.to_csv(adverse_spoof_top5_csv, index=False)

    # photo spoof hard top-5
    photo_spoof_top5 = df_hard[
        (df_hard["label"] == 1) &
        (df_hard["attack_type"] == "photo")
    ].sort_values("margin_to_threshold", ascending=True).head(5).copy()
    photo_spoof_top5_csv = os.path.join(OUT_DIR, "devel_photo_spoof_hard_top5_for_visual_check.csv")
    photo_spoof_top5.to_csv(photo_spoof_top5_csv, index=False)

    print()
    print("[INFO] Saved visualization candidate csv files")
    print(f"[INFO] {real_top10_csv}")
    print(f"[INFO] {spoof_top10_csv}")
    print(f"[INFO] {hand_spoof_top5_csv}")
    print(f"[INFO] {adverse_spoof_top5_csv}")
    print(f"[INFO] {photo_spoof_top5_csv}")


if __name__ == "__main__":
    main()