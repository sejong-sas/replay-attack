from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FRAME_ROOT = ROOT / "frames" / "replay_attack" / "test"
OUT_DIR = ROOT / "outputs" / "analysis" / "clip5_error_analysis"

THRESHOLD = 0.103

CASES = [
    {
        "name": "Adverse 1",
        "video": "test__real__real__client014_session01_webcam_authenticate_adverse_1",
        "frame": "frame_00118.jpg",
        "score": 0.852429,
        "prediction": "attack",
        "result": "False positive",
        "border": "#b9252d",
        "fill": "#fff1f1",
    },
    {
        "name": "Adverse 2",
        "video": "test__real__real__client014_session01_webcam_authenticate_adverse_2",
        "frame": "frame_00118.jpg",
        "score": 0.974687,
        "prediction": "attack",
        "result": "False positive",
        "border": "#b9252d",
        "fill": "#fff1f1",
    },
    {
        "name": "Controlled 1",
        "video": "test__real__real__client014_session01_webcam_authenticate_controlled_1",
        "frame": "frame_00118.jpg",
        "score": 0.000476,
        "prediction": "bona fide",
        "result": "Correct",
        "border": "#1d7f45",
        "fill": "#eef8f1",
    },
    {
        "name": "Controlled 2",
        "video": "test__real__real__client014_session01_webcam_authenticate_controlled_2",
        "frame": "frame_00118.jpg",
        "score": 0.000483,
        "prediction": "bona fide",
        "result": "Correct",
        "border": "#1d7f45",
        "fill": "#eef8f1",
    },
]


def load_frame(case: dict[str, object]) -> Image.Image:
    path = FRAME_ROOT / str(case["video"]) / str(case["frame"])
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def draw_case(ax: plt.Axes, case: dict[str, object]) -> None:
    image = load_frame(case)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    border = str(case["border"])
    fill = str(case["fill"])
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color(border)

    ax.set_title(
        (
            f"{case['name']} - {case['result']}\n"
            f"GT: bona fide | Pred: {case['prediction']}\n"
            f"Attack score={case['score']:.3f}"
        ),
        fontsize=8.8,
        weight="bold",
        color="#1a1a1a",
        pad=7,
        bbox={"facecolor": fill, "edgecolor": border, "boxstyle": "round,pad=0.35"},
    )


def draw_score_panel(ax: plt.Axes) -> None:
    names = [str(case["name"]) for case in CASES]
    scores = [float(case["score"]) for case in CASES]
    colors = [str(case["border"]) for case in CASES]

    ax.barh(names, scores, color=colors, alpha=0.9)
    ax.axvline(THRESHOLD, color="#222222", linewidth=1.6, linestyle="--")
    ax.text(
        THRESHOLD + 0.015,
        -0.45,
        f"threshold = {THRESHOLD:.3f}",
        fontsize=9,
        color="#222222",
        va="center",
    )

    for y, score in enumerate(scores):
        x = min(score + 0.025, 0.92) if score > 0.05 else 0.025
        label = f"{score:.3f}" if score >= 0.001 else "<0.001"
        ax.text(x, y, label, va="center", fontsize=10, color="#1a1a1a")

    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Attack score")
    ax.set_title("Video-level decision scores", fontsize=12, weight="bold", pad=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.0, 7.0), dpi=220)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.9],
        height_ratios=[0.18, 1.0, 1.0],
        wspace=0.22,
        hspace=0.62,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.78,
        "Bona fide client014: adverse illumination increases attack scores",
        fontsize=17,
        weight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0,
        0.22,
        "All four videos are bona fide. Red panels are false positives: only adverse-illumination trials cross the attack threshold.",
        fontsize=11,
        color="#444444",
        ha="left",
        va="center",
    )

    image_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    for ax, case in zip(image_axes, CASES):
        draw_case(ax, case)

    score_ax = fig.add_subplot(gs[1:, 2])
    draw_score_panel(score_ax)

    fig.add_artist(
        Rectangle(
            (0.03, 0.04),
            0.94,
            0.90,
            transform=fig.transFigure,
            fill=False,
            linewidth=0.8,
            edgecolor="#d0d0d0",
        )
    )

    png_path = OUT_DIR / "client014_illumination_score_case_figure.png"
    pdf_path = OUT_DIR / "client014_illumination_score_case_figure.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
