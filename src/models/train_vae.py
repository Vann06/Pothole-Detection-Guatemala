"""Train the convolutional VAE on dataset_v1 train images without annotated damage and evaluate every split.

Checkpoint: lowest val loss on no-damage images at beta=1 (unsupervised), only once the beta warmup ended.
Threshold: max val F1 for D40, per score (global / patch); the primary score is the one with higher val PR-AUC.
Test splits never influence the checkpoint, the threshold or the primary score.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.data.cache_images import cache_root, cached_frame
from src.data.dataset import (DATASET_ROOT, MANIFESTS, PotholeDataset, build_transforms, load_split_frame,
                              make_loader, release_info)
from src.evaluation.breakdown import damage_group, group_report
from src.evaluation.latency import measure_latency
from src.evaluation.metrics import best_f1_threshold, prediction_frame, split_metrics
from src.models.vae import ConvVAE, anomaly_scores, vae_loss, vae_terms
from src.utils.seed import safe_experiment_id, set_seed

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/experiments/vae"
CHECKPOINTS = ROOT / "checkpoints/vae"
EVAL_SPLITS = ("val", "test_id", "test_ood", "test_perspective")
SCORES = ("global", "patch")
DEFAULTS = dict(latent=128, image_size=128, batch_size=64, epochs=50, lr=1e-3, patience=8, warmup_epochs=5,
                patch=16, num_workers=2, seed=42, amp=True, use_cache=True, latency_images=200)
SCORE_SCALE = ("squared error per pixel in [0, 1] RGB space, averaged over channels; global = image mean, "
               "patch = max of the patch-averaged error; higher = more anomalous")


def beta_at(epoch, warmup):
    """Linear KL warmup: beta reaches 1 at epoch == warmup (epochs start at 1)."""
    return 1.0 if warmup <= 0 else min(1.0, epoch / warmup)


def select_frames(frame, limit, seed):
    """Train = no annotated damage only; --limit applies after that filter and per evaluation split."""
    train = frame[(frame.split == "train") & (frame.has_any_damage == 0)]
    evaluation = frame[frame.split.isin(EVAL_SPLITS)]
    if limit:
        train = train.sample(n=min(limit, len(train)), random_state=seed)
        evaluation = evaluation.sample(frac=1, random_state=seed).groupby("split").head(limit)
    if train.empty:
        raise ValueError("no train images without annotated damage")
    if not ((evaluation.split == "val") & (evaluation.has_any_damage == 0)).any():
        raise ValueError("no val images without annotated damage: checkpoint selection needs them")
    return train, evaluation


def image_source(config, dataset_root, frames):
    """Use the resized PNG cache when asked for it, failing early if any image is missing."""
    if not config["use_cache"]:
        return Path(dataset_root), frames
    root = cache_root(config["image_size"], dataset_root)
    frames = [cached_frame(frame) for frame in frames]
    missing = sum(not (root / path).exists() for frame in frames for path in frame.image_path)
    if missing:
        raise FileNotFoundError(f"{missing} cached images missing in {root.name}; run "
                                f"python -m src.data.cache_images --size {config['image_size']}")
    return root, frames


def train_epoch(model, loader, optimizer, scaler, device, amp, beta):
    model.train()
    totals, count = {"loss": 0.0, "recon": 0.0, "kl": 0.0}, 0
    for images, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            recon, mu, logvar = model(images)
        loss, reconstruction, kl = vae_loss(recon, images, mu, logvar, beta)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for key, value in zip(totals, (loss, reconstruction, kl)):
            totals[key] += value.item() * len(images)
        count += len(images)
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def score_images(model, loader, device, amp, patch):
    """Per image: sample_id, has_D40, reconstruction and KL terms, and both anomaly scores (z = mu)."""
    model.eval()
    parts = []
    for images, labels, sample_ids in tqdm(loader, desc="score", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            recon, mu, logvar = model(images)
        reconstruction, kl = vae_terms(recon, images, mu, logvar)
        global_score, patch_score = anomaly_scores(images, recon, patch, patch)
        parts.append(pd.DataFrame(dict(sample_id=list(sample_ids), y_true=labels[:, 0].int().numpy(),
                                       recon=reconstruction.cpu().numpy(), kl=kl.cpu().numpy(),
                                       global_score=global_score.cpu().numpy(),
                                       patch_score=patch_score.cpu().numpy())))
    return pd.concat(parts, ignore_index=True)


def val_summary(scored, normal_ids):
    """Loss at beta=1 on the no-damage val images, plus D40 PR-AUC of each score (logged only)."""
    normal = scored[scored.sample_id.isin(normal_ids)]
    summary = dict(val_loss=float((normal.recon + normal.kl).mean()), val_recon=float(normal.recon.mean()),
                   val_kl=float(normal.kl.mean()))
    for name in SCORES:
        summary[f"val_pr_auc_{name}"] = float(average_precision_score(scored.y_true, scored[f"{name}_score"]))
    return summary


def train(config, manifests=MANIFESTS, dataset_root=DATASET_ROOT, reports=REPORTS,
          checkpoints=CHECKPOINTS, limit=None, device=None):
    config = {**DEFAULTS, **config}
    if config["warmup_epochs"] > config["epochs"]:
        raise ValueError("warmup_epochs must not exceed epochs, otherwise beta never reaches 1")
    experiment_id = safe_experiment_id(config["experiment_id"]) + (f"_limit{limit}" if limit else "")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = config["amp"] and device.type == "cuda"
    set_seed(config["seed"])
    # sin autotune de cudnn para que las corridas se parezcan lo mas posible
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

    release = release_info(manifests)
    frame = load_split_frame(manifests=manifests)
    train_frame, eval_frame = select_frames(frame, limit, config["seed"])
    root, (train_images, eval_images) = image_source(config, dataset_root, [train_frame, eval_frame])
    transform = build_transforms(size=config["image_size"], normalize=False)

    def loader_for(split_frame, shuffle):
        return make_loader(split_frame, transform, root, config["batch_size"], config["num_workers"],
                           config["seed"], shuffle)

    train_loader = loader_for(train_images, shuffle=True)
    val_loader = loader_for(eval_images[eval_images.split == "val"], shuffle=False)
    val_normals = set(eval_frame.sample_id[(eval_frame.split == "val") & (eval_frame.has_any_damage == 0)])
    print(f"{experiment_id}: {len(train_loader.dataset)} train (no damage) / {len(val_loader.dataset)} val "
          f"({len(val_normals)} no damage), device={device}")

    model = ConvVAE(latent=config["latent"], image_size=config["image_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    out_dir, ckpt_dir = Path(reports) / experiment_id, Path(checkpoints) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch = [], float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        start, beta = time.time(), beta_at(epoch, config["warmup_epochs"])
        train_stats = train_epoch(model, train_loader, optimizer, scaler, device, amp, beta)
        val_stats = val_summary(score_images(model, val_loader, device, amp, config["patch"]), val_normals)
        history.append(dict(epoch=epoch, beta=beta, **{f"train_{k}": v for k, v in train_stats.items()},
                            **val_stats, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - start))
        print(f"epoch {epoch:02d} beta={beta:.2f} train_loss={train_stats['loss']:.2f} "
              f"val_loss={val_stats['val_loss']:.2f} val_pr_auc global={val_stats['val_pr_auc_global']:.4f} "
              f"patch={val_stats['val_pr_auc_patch']:.4f}")
        if epoch < config["warmup_epochs"]:
            continue  # con beta < 1 la perdida no es comparable, todavia no se elige checkpoint
        if val_stats["val_loss"] < best_loss:
            best_loss, best_epoch = val_stats["val_loss"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, ckpt_dir / "best.pt")
        elif epoch - best_epoch >= config["patience"]:
            print(f"early stopping: no val loss improvement since epoch {best_epoch}")
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, lineterminator="\n")
    peak_mb = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)["model"])
    scored = {}
    for split in EVAL_SPLITS:
        loader = val_loader if split == "val" else loader_for(eval_images[eval_images.split == split], shuffle=False)
        scored[split] = score_images(model, loader, device, amp, config["patch"]).merge(
            eval_frame, on="sample_id", validate="one_to_one")

    # todo lo que se elige sale de val
    val = scored["val"]
    thresholds = {name: best_f1_threshold(val.y_true, val[f"{name}_score"]) for name in SCORES}
    val_pr_auc = {name: float(average_precision_score(val.y_true, val[f"{name}_score"])) for name in SCORES}
    primary = max(SCORES, key=lambda name: val_pr_auc[name])

    results, breakdown, rows = {}, {}, []
    for split, table in scored.items():
        results[split] = {name: split_metrics(table.y_true.to_numpy(), table[f"{name}_score"].to_numpy(),
                                              thresholds[name]) for name in SCORES}
        breakdown[split] = {name: group_report(table, table[f"{name}_score"], thresholds[name]) for name in SCORES}
        predictions = prediction_frame(table.sample_id, split, table.y_true, table[f"{primary}_score"],
                                       thresholds[primary], experiment_id, release["dataset_version"])
        rows.append(predictions.assign(y_score_global=table.global_score, y_score_patch=table.patch_score,
                                       group=damage_group(table), country=table.country))
        print(f"{split}: " + ", ".join(f"{name} pr_auc={results[split][name]['pr_auc']:.4f} "
                                       f"f1={results[split][name]['f1']:.4f}" for name in SCORES))
    pd.concat(rows).to_csv(out_dir / "predictions.csv", index=False, lineterminator="\n")

    latency_set = PotholeDataset(eval_images[eval_images.split == "test_id"], transform, root)
    latency = {"batch_size": 1, "image_size": config["image_size"],
               "includes": "forward pass only (z = mu), preprocessed input"}
    if device.type == "cuda":
        latency["cuda_ms"] = measure_latency(model, latency_set, device, config["latency_images"])
    latency["cpu_ms"] = measure_latency(model, latency_set, torch.device("cpu"), config["latency_images"] // 4)

    metrics = dict(experiment_id=experiment_id, primary_score=primary, thresholds=thresholds,
                   val_pr_auc=val_pr_auc, best_epoch=best_epoch, best_val_loss=best_loss, splits=results,
                   breakdown=breakdown, latency=latency, cuda_peak_mb=peak_mb)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    resolved = {**config, "experiment_id": experiment_id, "optimizer": "Adam", "scheduler": None,
                "loss": "per-image sum of squared error + beta * KL, averaged over the batch",
                "beta_schedule": f"linear 0 -> 1 over {config['warmup_epochs']} epochs",
                "selection": "min val loss (beta=1) on val images without annotated damage, after warmup",
                "threshold": "max val F1 (D40) per score; primary score = higher val PR-AUC (D40)",
                "score_scale": SCORE_SCALE, "image_source": root.name, "train_images": len(train_frame),
                "val_no_damage_images": len(val_normals), "best_epoch": best_epoch, "limit": limit,
                "device": str(device), "torch": str(torch.__version__), **release}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    print(f"primary score: {primary}; results in {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Use at most N images per split (smoke runs)")
    parser.add_argument("--epochs", type=int, help="Override epochs from the config")
    parser.add_argument("--warmup-epochs", type=int, help="Override the beta warmup (needed when --epochs is small)")
    parser.add_argument("--device", help="cuda or cpu; defaults to cuda when available")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs:
        config["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        config["warmup_epochs"] = args.warmup_epochs
    train(config, limit=args.limit, device=args.device)


if __name__ == "__main__":
    main()
