import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.cnn_lstm_attention import CNNLSTMAttentionBinaryClassifier

CLIP_INDEX_CSV = "/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip10_index.csv"
OUTPUT_DIR = "/home/saslab01/Desktop/replay_pad/outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4


def build_loader(split, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = ReplayPADClipDataset(
        csv_path=CLIP_INDEX_CSV,
        split=split,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return dataset, loader


def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for clips, labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(clips)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += clips.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def main():
    print(f"[INFO] Device: {DEVICE}")

    _, train_loader = build_loader("train", shuffle=True)
    _, devel_loader = build_loader("devel", shuffle=False)

    model = CNNLSTMAttentionBinaryClassifier(
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        pretrained=False,
        dropout=0.2,
        freeze_backbone=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_devel_acc = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        devel_loss, devel_acc = run_one_epoch(model, devel_loader, criterion, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "devel_loss": devel_loss,
            "devel_acc": devel_acc,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}"
        )

        if devel_acc > best_devel_acc:
            best_devel_acc = devel_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    ckpt_path = os.path.join(CHECKPOINT_DIR, "cnn_lstm_attention_clip10_random_best.pth")
    torch.save(best_state, ckpt_path)
    print(f"[INFO] Saved best checkpoint -> {ckpt_path}")

    log_path = os.path.join(LOG_DIR, "cnn_lstm_attention_clip10_train_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_devel_acc": best_devel_acc,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved training log -> {log_path}")


if __name__ == "__main__":
    main()