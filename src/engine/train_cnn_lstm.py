import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier

CLIP_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip10_index.csv"
CHECKPOINT_DIR = "/home/saslab01/Desktop/replay_pad/outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for clips, labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(clips)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += clips.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = ReplayPADClipDataset(CLIP_INDEX_CSV, split="train", transform=transform)
    devel_dataset = ReplayPADClipDataset(CLIP_INDEX_CSV, split="devel", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    devel_loader = DataLoader(devel_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = CNNLSTMBinaryClassifier(
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_devel_acc = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "cnn_lstm_clip10_random_best.pth")

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Train clips: {len(train_dataset)}")
    print(f"[INFO] Devel clips: {len(devel_dataset)}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        devel_loss, devel_acc = run_one_epoch(model, devel_loader, criterion)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}"
        )

        if devel_acc > best_devel_acc:
            best_devel_acc = devel_acc
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved best checkpoint -> {best_path}")

    print(f"[INFO] Training finished. Best devel acc = {best_devel_acc:.4f}")


if __name__ == "__main__":
    main()