from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

CSV_PATH = Path("/Users/youbin/Desktop/replay_pad/metadata/baseline_image_labels.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4

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

        return image, label

def get_loaders():
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "devel"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = ReplayImageDataset(train_df, transform=transform)
    val_ds = ReplayImageDataset(val_df, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    return accuracy_score(labels, preds)

def main():
    train_loader, val_loader = get_loaders()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Devel Acc: {val_acc:.4f}")

    ckpt_dir = Path("/Users/youbin/Desktop/replay_pad/outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "resnet18_baseline.pt")
    print("Saved checkpoint.")

if __name__ == "__main__":
    main()