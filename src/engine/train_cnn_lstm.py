import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.replay_pad_clip_dataset import ReplayPADClipDataset
from src.models.cnn_lstm_baseline import CNNLSTMBinaryClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip20_train.csv",
    )
    parser.add_argument(
        "--devel_csv",
        type=str,
        default="/home/saslab01/Desktop/replay_pad/clip_index/replayattack_clip20_devel.csv",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="cnn_lstm_clip20_random_best.pth",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser.parse_args()


CHECKPOINT_DIR = "/home/saslab01/Desktop/replay_pad/outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
    args = parse_args()
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    # split별 CSV를 직접 사용
    train_dataset = ReplayPADClipDataset(
        csv_path=args.train_csv,
        split="train",
        transform=transform
    )
    devel_dataset = ReplayPADClipDataset(
        csv_path=args.devel_csv,
        split="devel",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    devel_loader = DataLoader(
        devel_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = CNNLSTMBinaryClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        pretrained=False,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_devel_acc = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, args.save_name)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Train CSV: {args.train_csv}")
    print(f"[INFO] Devel CSV: {args.devel_csv}")
    print(f"[INFO] Train clips: {len(train_dataset)}")
    print(f"[INFO] Devel clips: {len(devel_dataset)}")

    for epoch in range(1, args.epochs + 1):
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