import os
import imghdr
import random
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Optional: fix random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transform(is_train: bool, args):
    """Build image augmentation/preprocessing pipeline.
    
    Expected args fields:
      - input_size: int
      - resize_shorter: Optional[int], shorter side for val/test (optional)
    """
    input_size = args.input_size
        
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    crop_size = int(args.input_size / crop_pct)

    if is_train:
        # Common training augmentations: random resized crop + hflip + color jitter
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(0.64, 1.0),
                ratio=(3.0/4.0, 4.0/3.0),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.01),
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Validation/Test: resize shorter side then center crop
        transform = transforms.Compose([
            transforms.Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform


class SurvDataSet(Dataset):
    """Image dataset for survival analysis.
    
    Expected directory structure:
      data_path/
        ├─ images/
        │    ├─ xxx.jpg/png/...
        ├─ train.csv
        └─ val.csv

    CSV must contain at least:
      - image: filename relative to images/ folder
      - tte:   time-to-event (float)
      - event: outcome indicator (0/1)

    Recommended args fields:
      - data_path: root data directory
      - input_size: input image size
      - resize_shorter: (optional) shorter side for val/test
    """

    def __init__(self, args, is_train: bool = True):
        super().__init__()
        self.is_train = bool(is_train)
        self.data_path = args.data_path

        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"data_path not found: {self.data_path}")

        csv_name = "train.csv" if self.is_train else "val.csv"
        meta_file_path = os.path.join(self.data_path, csv_name)
        if not os.path.isfile(meta_file_path):
            raise FileNotFoundError(f"CSV file not found: {meta_file_path}")

        df = pd.read_csv(meta_file_path)
        if df.empty:
            raise ValueError(f"Empty CSV: {meta_file_path}")

        required_cols = {"image", "tte", "event"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

        # Read columns
        images = df["image"].astype(str).tolist()
        tte_np = df["tte"].astype(float).to_numpy()
        event_np = df["event"].astype(int).to_numpy()

        # Build transform
        self.transform = build_transform(self.is_train, args)

        # Prepare samples and check files
        img_root = os.path.join(self.data_path, "images")
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"Images folder not found: {img_root}")

        self.data_list: List[Tuple[str, float, int]] = []
        missing_files: List[str] = []

        for fname, tte, evt in zip(images, tte_np, event_np):
            image_path = os.path.join(img_root, fname)
            if not os.path.isfile(image_path):
                missing_files.append(fname)
                continue
            # Optional: filter non-image files
            try:
                if imghdr.what(image_path) is None:
                    missing_files.append(fname)
                    continue
            except Exception:
                missing_files.append(fname)
                continue
            self.data_list.append((image_path, float(tte), int(evt)))

        if len(self.data_list) == 0:
            raise ValueError(
                f"No valid samples found. Missing/invalid files: {len(missing_files)} of {len(images)}."
            )
        if missing_files:
            print(f"[SurvDataSet] Warning: skipped {len(missing_files)} invalid/missing images. "
                  f"Examples: {missing_files[:5]}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        image_path, tte, evt = self.data_list[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        # Return torch.tensors with proper dtypes
        tte_tensor = torch.tensor(tte, dtype=torch.float32)
        evt_tensor = torch.tensor(evt, dtype=torch.long)

        return img, tte_tensor, evt_tensor