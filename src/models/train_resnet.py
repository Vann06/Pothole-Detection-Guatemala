"""Train ResNet18 for D40 presence on dataset_v1, select on val and evaluate every held-out split.

With aux_classes (e.g. [D00, D10, D20]) the network also predicts those damages as auxiliary tasks;
D40 stays the primary target for checkpoint selection, threshold and reported metrics.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from tqdm import tqdm

from src.data.dataset import (DATASET_ROOT, MANIFESTS, PotholeDataset, build_transforms, compute_pos_weight,
                              label_columns, load_split_frame, make_loader, release_info)
from src.evaluation.latency import measure_latency
from src.evaluation.metrics import best_f1_threshold, prediction_frame, split_metrics
from src.models.resnet import build_resnet18
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/experiments/resnet"
CHECKPOINTS = ROOT / "checkpoints/resnet"
EVAL_SPLITS = ("val", "test_id", "test_ood", "test_perspective")
DEFAULTS = dict(pretrained=False, augment=False, image_size=224, batch_size=32, epochs=20, lr=1e-4,
                weight_decay=1e-4, patience=5, num_workers=4, seed=42, amp=True, latency_images=200,
                aux_classes=[], aux_weight=0.0)


class AuxiliaryLoss(nn.Module):
    """BCE(D40) + aux_weight * mean BCE(auxiliary classes), each with its own train pos_weight."""

    def __init__(self, pos_weights, aux_weight):
        super().__init__()
        self.register_buffer("pos_weights", torch.as_tensor(pos_weights, dtype=torch.float32))
        self.aux_weight = aux_weight

    def forward(self, logits, labels):
        per_class = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels, pos_weight=self.pos_weights, reduction="none").mean(0)
        if len(per_class) == 1:
            return per_class[0]
        return per_class[0] + self.aux_weight * per_class[1:].mean()


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    total = 0.0
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, criterion, device, amp):
    """Return mean loss, sample ids, label matrix and probability matrix (column 0 = D40)."""
    model.eval()
    total, ids, labels, scores = 0.0, [], [], []
    for images, y, sample_ids in tqdm(loader, desc="eval", leave=False):
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images).float()
        total += criterion(logits, y).item() * len(y)
        ids += list(sample_ids)
        labels.append(y.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return total / len(loader.dataset), ids, np.concatenate(labels).astype(int), np.concatenate(scores)


def train(config, manifests=MANIFESTS, dataset_root=DATASET_ROOT, reports=REPORTS,
          checkpoints=CHECKPOINTS, limit=None, device=None):
    config = {**DEFAULTS, **config}
    experiment_id = config["experiment_id"] + (f"_limit{limit}" if limit else "")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = config["amp"] and device.type == "cuda"
    set_seed(config["seed"])

    frame = load_split_frame(manifests=manifests)
    if limit:
        frame = frame.sample(frac=1, random_state=config["seed"]).groupby("split").head(limit)
    columns = label_columns(config["aux_classes"])
    pos_weights = {column: compute_pos_weight(frame, column) for column in columns}
    size = config["image_size"]

    def loader_for(split_frame, transform, shuffle):
        return make_loader(split_frame, transform, dataset_root, config["batch_size"], config["num_workers"],
                           config["seed"], shuffle, config["aux_classes"])

    eval_transform = build_transforms(train=False, size=size)
    train_loader = loader_for(frame[frame.split == "train"], build_transforms(True, config["augment"], size),
                              shuffle=True)
    val_loader = loader_for(frame[frame.split == "val"], eval_transform, shuffle=False)
    print(f"{experiment_id}: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val, "
          f"pos_weight={ {k: round(v, 3) for k, v in pos_weights.items()} }, device={device}")

    model = build_resnet18(config["pretrained"], outputs=len(columns)).to(device)
    criterion = AuxiliaryLoss(list(pos_weights.values()), config["aux_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    out_dir, ckpt_dir = Path(reports) / experiment_id, Path(checkpoints) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history, best_auc, best_epoch = [], -1.0, 0
    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp)
        val_loss, _, y_val, s_val = predict(model, val_loader, criterion, device, amp)
        val_auc = float(average_precision_score(y_val[:, 0], s_val[:, 0]))  # selection uses D40 only
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_pr_auc=val_auc,
                            lr=lr, seconds=time.time() - start))
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_pr_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "best.pt")
        elif epoch - best_epoch >= config["patience"]:
            print(f"early stopping: no val PR-AUC improvement since epoch {best_epoch}")
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, lineterminator="\n")

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)["model"])
    release = release_info(manifests)
    results, rows, threshold = {}, [], None
    for split in EVAL_SPLITS:
        loader = val_loader if split == "val" else loader_for(frame[frame.split == split], eval_transform,
                                                             shuffle=False)
        _, ids, y_true, y_score = predict(model, loader, criterion, device, amp)
        if threshold is None:  # val comes first: the threshold is fixed there and reused on every test split
            threshold = best_f1_threshold(y_true[:, 0], y_score[:, 0])
        results[split] = split_metrics(y_true[:, 0], y_score[:, 0], threshold)
        table = prediction_frame(ids, split, y_true[:, 0], y_score[:, 0], threshold, experiment_id,
                                 release["dataset_version"])
        if config["aux_classes"]:
            results[split]["auxiliary"] = {}
        for index, name in enumerate(config["aux_classes"], start=1):
            table[f"y_true_{name}"], table[f"y_score_{name}"] = y_true[:, index], y_score[:, index]
            results[split]["auxiliary"][name] = dict(
                positives=int(y_true[:, index].sum()), prevalence=float(y_true[:, index].mean()),
                pr_auc=float(average_precision_score(y_true[:, index], y_score[:, index])))
        rows.append(table)
        print(f"{split}: " + ", ".join(f"{k}={results[split][k]:.4f}" for k in ("precision", "recall", "f1", "pr_auc")))
    pd.concat(rows).to_csv(out_dir / "predictions.csv", index=False, lineterminator="\n")

    latency_set = PotholeDataset(frame[frame.split == "test_id"], eval_transform, dataset_root)
    latency = {"batch_size": 1, "image_size": size, "includes": "forward pass only, preprocessed input"}
    if device.type == "cuda":
        latency["cuda_ms"] = measure_latency(model, latency_set, device, config["latency_images"])
    latency["cpu_ms"] = measure_latency(model, latency_set, torch.device("cpu"), config["latency_images"] // 4)
    metrics = dict(experiment_id=experiment_id, threshold=threshold, best_epoch=best_epoch,
                   splits=results, latency=latency)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    resolved = {**config, "experiment_id": experiment_id, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
                "loss": "BCEWithLogitsLoss(D40) + aux_weight * mean BCE(aux)" if config["aux_classes"]
                else "BCEWithLogitsLoss", "pos_weight": pos_weights[columns[0]], "pos_weights": pos_weights,
                "best_epoch": best_epoch, "selection": "max val PR-AUC (D40); threshold = max val F1 (D40)",
                "limit": limit, "device": str(device), "torch": str(torch.__version__), **release}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print(f"results in {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Use at most N images per split (smoke runs)")
    parser.add_argument("--epochs", type=int, help="Override epochs from the config")
    parser.add_argument("--device", help="cuda or cpu; defaults to cuda when available")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs:
        config["epochs"] = args.epochs
    train(config, limit=args.limit, device=args.device)


if __name__ == "__main__":
    main()
