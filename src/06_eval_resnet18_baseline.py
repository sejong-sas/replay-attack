from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

CSV_PATH = Path("/Users/youbin/Desktop/replay_pad/metadata/baseline_image_labels.csv")
CKPT_PATH = Path("/Users/youbin/Desktop/replay_pad/outputs/checkpoints/resnet18_baseline.pt")
OUT_PRED_CSV = Path("/Users/youbin/Desktop/replay_pad/outputs/predictions/baseline_test_frame_preds.csv")
OUT_VIDEO_CSV = Path("/Users/youbin/Desktop/replay_pad/outputs/predictions/baseline_test_video_preds.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

class ReplayImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label, row["image_path"], row["video_path"]

def main():
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].copy()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_ds = ReplayImageDataset(test_df, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    rows = []

    with torch.no_grad():
        for images, labels, image_paths, video_paths in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]   # spoof probability
            preds = (probs >= 0.5).long().cpu().numpy()

            for i in range(len(image_paths)):
                rows.append({
                    "image_path": image_paths[i],
                    "video_path": video_paths[i],
                    "label": int(labels[i]),
                    "pred": int(preds[i]),
                    "spoof_prob": float(probs[i].cpu().item()),
                })

    pred_df = pd.DataFrame(rows)
    OUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUT_PRED_CSV, index=False)

    frame_acc = accuracy_score(pred_df["label"], pred_df["pred"])
    print(f"Frame-level Test Accuracy: {frame_acc:.4f}")

    video_df = pred_df.groupby("video_path").agg({
        "label": "first",
        "spoof_prob": "mean"
    }).reset_index()

    video_df["pred"] = (video_df["spoof_prob"] >= 0.5).astype(int)
    video_df.to_csv(OUT_VIDEO_CSV, index=False)

    video_acc = accuracy_score(video_df["label"], video_df["pred"])
    print(f"Video-level Test Accuracy: {video_acc:.4f}")
    print("\nVideo-level classification report:")
    print(classification_report(video_df["label"], video_df["pred"], digits=4))
    print("Video-level confusion matrix:")
    print(confusion_matrix(video_df["label"], video_df["pred"]))

if __name__ == "__main__":
    main()