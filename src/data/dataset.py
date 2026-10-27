"""Load the frozen dataset_v1 splits and serve RDD2022 images for image-level D40 classification."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data/manifests"
DATASET_ROOT = ROOT / "data/raw/RDD2022"
SPLITS = ("train", "val", "test_id", "test_ood", "test_perspective")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def release_info(manifests=MANIFESTS):
    """Return dataset version and splits.csv SHA256, failing if splits.csv differs from the lock."""
    lock = json.loads((manifests / "dataset_v1.lock.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((manifests / "splits.csv").read_bytes()).hexdigest()
    if sha != lock["sha256"]["data/manifests/splits.csv"]:
        raise ValueError("splits.csv does not match dataset_v1.lock.json; use the official splits")
    return {"dataset_version": lock["version"], "splits_sha256": sha}


def load_split_frame(splits=None, manifests=MANIFESTS):
    """Merge images.csv with splits.csv one-to-one and keep the requested official splits."""
    release_info(manifests)
    images = pd.read_csv(manifests / "images.csv", low_memory=False)
    frame = images.merge(pd.read_csv(manifests / "splits.csv"), on="sample_id", validate="one_to_one")
    assert set(frame.split) <= set(SPLITS), "Unknown split name"
    if splits is not None:
        frame = frame[frame.split.isin(splits)]
    return frame.reset_index(drop=True)


def label_columns(aux_classes=()):
    """D40 is always column 0 (the primary target); auxiliary damage classes follow."""
    return ["has_D40"] + [f"has_{name}" for name in aux_classes]


def compute_pos_weight(frame, column="has_D40"):
    """negatives / positives computed on the train split only."""
    train = frame[frame.split == "train"]
    positives = int(train[column].sum())
    return (len(train) - positives) / positives


def build_transforms(train=False, augment=False, size=224):
    """Deterministic resize for evaluation; augmentation only when training with augment=True."""
    if train and augment:
        steps = [T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                 T.RandomHorizontalFlip(),
                 T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                 T.RandomRotation(5)]
    else:
        steps = [T.Resize((size, size))]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PotholeDataset(Dataset):
    """Yields (image tensor, label vector [has_D40, *aux], sample_id); image_path is relative to RDD2022/."""

    def __init__(self, frame, transform, dataset_root=DATASET_ROOT, aux_classes=()):
        self.paths = [Path(dataset_root) / p for p in frame.image_path]
        self.labels = frame[label_columns(aux_classes)].astype("float32").to_numpy()
        self.ids = frame.sample_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.labels[index]), self.ids[index]
