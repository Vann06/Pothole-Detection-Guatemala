"""Load audited checkpoints, benchmark the complete score and evaluate reviewed Guatemala data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import DATASET_ROOT, PotholeDataset, build_transforms
from src.evaluation.cross_domain import frozen_frame, load_experiment, sha256, validate_predictions, metric_row
from src.evaluation.latency import benchmark_latency
from src.evaluation.metrics import prediction_frame
from src.models.classifiers import CLASSIFIER_ARCHITECTURES, build_classifier
from src.models.vae import anomaly_scores, build_vae


def load_checkpoint(path, config, family):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("experiment_id", "image_size", "aux_classes", "latent", "patch_size"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Checkpoint/config mismatch: {key}")
    if checkpoint["epoch"] != config["best_epoch"]:
        raise ValueError("Checkpoint is not the selected epoch")
    if family in CLASSIFIER_ARCHITECTURES:
        for source in (config, checkpoint["config"]):
            # Old ResNet checkpoints predate explicit family/architecture metadata.
            if source.get("family", "resnet") != family or source.get(
                    "architecture", "resnet18") != CLASSIFIER_ARCHITECTURES[family]:
                raise ValueError("Checkpoint/config classifier architecture disagrees with experiment family")
        model = build_classifier(family, outputs=1+len(config.get("aux_classes", [])))
    elif family == "vae":
        model = build_vae(config["latent"], config["image_size"], config.get("patch_size"))
    else:
        raise ValueError("Unknown model family")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def score_batch(model, images, family, config, metadata):
    if family in CLASSIFIER_ARCHITECTURES:
        return model(images)[:, 0].sigmoid()
    recon = model(images)[0]  # eval mode uses z = mu, never random sampling.
    global_score, patch_score, grid = anomaly_scores(images, recon, config["patch"], config["patch"])
    primary = metadata["primary_score"]
    if primary == "global":
        return global_score
    if primary == "patch":
        return patch_score
    if primary != "patch_norm":
        raise ValueError("Unknown VAE score")
    stats = metadata["position_stats"]
    # MPS does not implement float64; keep the operation on-device, without hidden transfers.
    dtype = torch.float32 if images.device.type == "mps" else torch.float64
    mean = torch.tensor(stats["mean"], dtype=dtype, device=images.device)
    std = torch.tensor(stats["std_eff"], dtype=dtype, device=images.device)
    return ((grid.to(dtype)-mean)/std).amax(1)


def load_guatemala(path, official):
    """No implicit labels, invented splits, or accepting unreviewed external samples."""
    frame = pd.read_csv(path)
    required = {"sample_id", "image_path", "has_D40", "has_any_damage", "source", "capture_session",
                "road_segment", "camera", "lighting", "weather", "labeler", "reviewer", "review_status", "notes"}
    if required - set(frame.columns):
        raise ValueError(f"Missing Guatemala protocol fields: {sorted(required-set(frame.columns))}")
    if frame.empty or not frame.sample_id.is_unique or frame.sample_id.isna().any():
        raise ValueError("Guatemala needs unique labeled images")
    if set(frame.sample_id) & set(official.sample_id):
        raise ValueError("Guatemala IDs overlap RDD2022")
    if not frame.review_status.eq("approved").all():
        raise ValueError("Every external label needs review_status=approved")
    for column in ("source", "labeler", "reviewer", "capture_session", "road_segment"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing external provenance: {column}")
    for column in ("has_D40", "has_any_damage"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be a reviewed binary label")
    if (frame.has_D40 > frame.has_any_damage).any():
        raise ValueError("D40 cannot be labeled as no annotated damage")
    for value in frame.image_path:
        p = Path(str(value))
        if pd.isna(value) or p.is_absolute() or ".." in p.parts or "\\" in str(value) or ":" in str(value):
            raise ValueError("External image paths must be relative to the supplied image root")
    return frame.assign(split="test_guatemala", country="Guatemala")


@torch.no_grad()
def predict_frame(model, frame, dataset_root, family, config, metadata, device):
    dataset = PotholeDataset(frame, build_transforms(size=config["image_size"],
                            normalize=family in CLASSIFIER_ARCHITECTURES), dataset_root)
    model.to(device).eval()
    scores = [float(score_batch(model, dataset[i][0][None].to(device), family, config, metadata).cpu()[0])
              for i in range(len(dataset))]
    return np.array(scores), dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--guatemala-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    official, release = frozen_frame()
    predictions, config, metadata = load_experiment(args.experiment, official, release)
    family = args.experiment.parent.name
    model = load_checkpoint(args.checkpoint, config, family)
    threshold = float(predictions.threshold.iloc[0])
    frame = (load_guatemala(args.guatemala_manifest, official) if args.guatemala_manifest
             else official[official.split == "test_id"].sort_values("sample_id").head(args.count))
    # External evaluation scores EVERY external sample; ID benchmark only needs its fixed subset.
    scores, dataset = predict_frame(model, frame, args.dataset_root, family, config, metadata, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.guatemala_manifest:
        exported = prediction_frame(frame.sample_id, "test_guatemala", frame.has_D40, scores, threshold,
                                    config["experiment_id"], release["dataset_version"])
        joined = validate_predictions(exported, frame, config["experiment_id"], release["dataset_version"], threshold)
        exported.to_csv(args.output / "predictions.csv", index=False)
        (args.output / "metrics.json").write_text(json.dumps(metric_row(joined), indent=2, allow_nan=False)+"\n")
    else:
        reference = predictions.set_index("sample_id").loc[frame.sample_id, "y_score"].to_numpy()
        if not np.allclose(scores, reference, rtol=.02, atol=.002):
            raise ValueError("Checkpoint does not reproduce archived scores; check preprocessing/AMP/source images")
    latency = benchmark_latency(model, dataset, torch.device(args.device), args.count, args.warmup,
                                operation=lambda x: score_batch(model, x, family, config, metadata))
    latency.update(experiment_id=config["experiment_id"], family=family, scope="forward float32 + D40 scoring",
                   normalized_score_precision=("float32" if args.device == "mps" else "float64")
                   if family == "vae" and metadata["primary_score"] == "patch_norm" else None,
                   checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.experiment / "config.yaml"),
                   sample_ids=frame.sample_id.tolist(), dataset_version=release["dataset_version"],
                   external_manifest_sha256=sha256(args.guatemala_manifest) if args.guatemala_manifest else None)
    (args.output / "latency.json").write_text(json.dumps(latency, indent=2)+"\n")
    print(json.dumps({k: latency[k] for k in ("experiment_id", "mean_ms", "median_ms", "p95_ms", "device", "scope")}, indent=2))


if __name__ == "__main__":
    main()
