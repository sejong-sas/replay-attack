from pathlib import Path
import copy
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_frame_dataset import ReplayPadFrameDataset
from src.models.mobilenetv3_small_baseline import MobileNetV3SmallBinaryClassifier


ROOT = Path("/Users/youbin/Desktop/replay_pad")
FRAME_CSV = ROOT / "frame_index" / "replay_pad_1frame.csv"
CKPT_PATH = ROOT / "outputs" / "checkpoints" / "mobilenetv3_small_1frame_best.pth"
META_PATH = ROOT / "outputs" / "results" / "mobilenetv3_small_1frame_train_log.json"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
PRETRAINED = False


def build_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ReplayPadFrameDataset(FRAME_CSV, split="train", transform=train_tf)
    devel_ds = ReplayPadFrameDataset(FRAME_CSV, split="devel", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    devel_loader = DataLoader(devel_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, devel_loader


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_loader, devel_loader = build_loaders()
    print(f"[INFO] train frames: {len(train_loader.dataset)}")
    print(f"[INFO] devel frames: {len(devel_loader.dataset)}")

    model = MobileNetV3SmallBinaryClassifier(pretrained=PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_state = None
    best_devel_loss = float("inf")
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        devel_loss, devel_acc = run_one_epoch(model, devel_loader, criterion, optimizer, device, train=False)

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"[Epoch {epoch+1}] devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "devel_loss": devel_loss,
            "devel_acc": devel_acc,
        })

        if devel_loss < best_devel_loss:
            best_devel_loss = devel_loss
            best_state = copy.deepcopy(model.state_dict())
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, CKPT_PATH)
            print(f"[INFO] best model saved -> {CKPT_PATH}")

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MobileNetV3-Small",
            "initialization": "pretrained" if PRETRAINED else "random",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lr": LR,
            "image_size": IMAGE_SIZE,
            "best_devel_loss": best_devel_loss,
            "history": history,
            "checkpoint_path": str(CKPT_PATH),
        }, f, ensure_ascii=False, indent=2)

    print("\n[INFO] training finished")
    print(f"[INFO] checkpoint saved at: {CKPT_PATH}")
    print(f"[INFO] log saved at: {META_PATH}")


if __name__ == "__main__":
    main()