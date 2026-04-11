import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_frame_dataset import ReplayPADFrameDataset
from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier

FRAME_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/frame_index/replayattack_frame_index.csv"
CHECKPOINT_DIR = "/home/saslab01/Desktop/replay_pad/outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
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

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def main():
    set_seed(SEED)

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = ReplayPADFrameDataset(
        csv_path=FRAME_INDEX_CSV,
        split="train",
        transform=transform_train,
    )
    devel_dataset = ReplayPADFrameDataset(
        csv_path=FRAME_INDEX_CSV,
        split="devel",
        transform=transform_eval,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    devel_loader = DataLoader(
        devel_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = MobileNetV3SmallBinaryClassifier(
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_devel_acc = 0.0
    best_path = os.path.join(
        CHECKPOINT_DIR,
        "mobilenetv3_small_1frame_random_best.pth"
    )

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Initialization: random")
    print(f"[INFO] Train frames: {len(train_dataset)}")
    print(f"[INFO] Devel frames: {len(devel_dataset)}")

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