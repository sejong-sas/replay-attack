from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


LABEL_MAP = {
    "real": 0,     # bona fide
    "attack": 1,   # spoof
}


class ReplayPadFrameDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["frame_path"]).convert("RGB")
        label = LABEL_MAP[row["label"]]

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": label,
            "video_id": row["video_id"],
            "attack_type": row["attack_type"],
            "split": row["split"],
            "frame_path": row["frame_path"],
        }
        return sample