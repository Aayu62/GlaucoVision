"""
PyTorch Dataset and DataLoader module for CA-Net.
Implements on-the-fly data augmentation:
- Random rotation
- Random scaling / zoom
- Horizontal flip
- Vertical flip
- Resize to 224x224
- ImageNet normalization
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Standard ImageNet normalization used for CNN backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(is_train=True, img_size=224):
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_train=False, img_size=224):
        if isinstance(csv_file, str):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = csv_file
        self.transform = transform or get_transforms(is_train=is_train, img_size=img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def build_dataloaders(splits_dir="c:/Users/KIIT0001/GlaucoVision/data_splits",
                      dataset_type="lmg",
                      batch_size=16,
                      num_workers=0):
    train_csv = os.path.join(splits_dir, f"{dataset_type}_train.csv")
    val_csv = os.path.join(splits_dir, f"{dataset_type}_val.csv")
    test_csv = os.path.join(splits_dir, f"{dataset_type}_test.csv")

    train_ds = GlaucomaDataset(train_csv, is_train=True)
    val_ds = GlaucomaDataset(val_csv, is_train=False)
    test_ds = GlaucomaDataset(test_csv, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
