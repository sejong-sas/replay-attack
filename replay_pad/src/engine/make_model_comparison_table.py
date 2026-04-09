from pathlib import Path
import json
import pandas as pd

ROOT = Path("/Users/youbin/Desktop/replay_pad")
OUT_CSV = ROOT / "outputs" / "results" / "model_comparison_video_level.csv"

files = [
    ROOT / "outputs" / "results" / "resnet18_eval_results.json",
    ROOT / "outputs" / "results" / "mobilenetv3_small_eval_results.json",
]

rows = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data["test_video_metrics"]
    rows.append({
        "Model": data["model"],
        "Initialization": data["initialization"],
        "Accuracy": m["accuracy"],
        "APCER": m["apcer"],
        "BPCER": m["bpcer"],
        "ACER": m["acer"],
        "HTER": m["hter"],
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(df)
print(f"\n[INFO] saved -> {OUT_CSV}")