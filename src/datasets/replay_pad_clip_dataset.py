import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ReplayPADClipDataset(Dataset):
    def __init__(self, csv_path, split="train", transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_paths = row["frame_paths"].split("|")
        label = int(row["label"])

        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        # [T, C, H, W]
        frames = torch.stack(frames, dim=0)
        return frames, label