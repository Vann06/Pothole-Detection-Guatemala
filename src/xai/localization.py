"""Grad-CAM localization on the SAME D40-positive test images for several classifiers.

`src.xai.gradcam` samples a few TP/TN/FP/FN cases per model, so two models are explained on different
photos. Here every model is explained on every official D40 image of the requested splits, and the
alignment of the map with the union of D40 boxes is aggregated. This compares *where* models look;
it is a diagnostic (not IoU, detection accuracy or causal proof).

    python -m src.xai.localization --experiments reports/experiments/resnet/exp2_pretrained_fp32 \
        reports/experiments/resnet/exp5_res384 --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.data.dataset import DATASET_ROOT, MANIFESTS, build_transforms
from src.evaluation.cross_domain import frozen_frame, load_experiment, sha256
from src.evaluation.inference import load_checkpoint
from src.xai.gradcam import bbox_alignment, gradcam

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/xai_evaluation/localization"
SPLITS = ("test_id", "test_ood", "test_perspective")


def pick_examples(frame, splits, per_split, seed=42):
    """Uniform seeded sample of D40-positive images per split; never chosen by model outcome."""
    positives = frame[(frame.has_D40 == 1) & frame.split.isin(splits)].sort_values("sample_id")
    return (positives.groupby("split", sort=False).sample(per_split, random_state=seed)
            .set_index("split").loc[list(splits)].reset_index()[["split", "sample_id", "image_path"]])


def render_examples(examples, maps, boxes, dataset_root, path):
    """Rows = the same images; columns = image with D40 boxes, then each model's Grad-CAM overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(maps)
    fig, axes = plt.subplots(len(examples), 1 + len(models), figsize=(3.2 * (1 + len(models)), 3 * len(examples)),
                             squeeze=False)
    for r, row in enumerate(examples.itertuples()):
        with Image.open(Path(dataset_root) / row.image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        for c in range(1 + len(models)):
            ax = axes[r, c]
            ax.imshow(rgb)
            for x1, y1, x2, y2 in boxes[row.sample_id]:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=1.2))
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"{row.split}\n{row.sample_id.split('__')[-1]}", fontsize=8)
        for c, model in enumerate(models, start=1):
            item = maps[model][row.sample_id]
            # Stretch back to the original aspect, matching the anisotropic resize used at evaluation.
            heat = np.asarray(Image.fromarray((item["heatmap"] * 255).astype(np.uint8)).resize((width, height)))
            axes[r, c].imshow(heat, cmap="jet", alpha=.45, vmin=0, vmax=255)
            outcome = "TP" if item["detected"] else "FN"
            axes[r, c].set_title(f"{model}\np={item['probability']:.3f} · {outcome} · "
                                 f"masa en cajas {item['mass_in_boxes'] or 0:.0%}", fontsize=7)
    axes[0, 0].set_title("Imagen y cajas D40 oficiales", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def explain_positives(experiment, checkpoint, frame, release, boxes, splits, dataset_root, device, keep=()):
    """One row per D40-positive image: probability, outcome and Grad-CAM alignment with D40 boxes.

    Heatmaps of the sample_ids in `keep` are also returned for the side-by-side figure.
    """
    family = experiment.parent.name
    predictions, config, _ = load_experiment(experiment, frame, release)
    model = load_checkpoint(checkpoint, config, family).to(device).eval()
    layer = model.layer4 if family == "resnet" else model.features[-1]
    transform = build_transforms(size=config["image_size"])
    positives = predictions[(predictions.y_true == 1) & predictions.split.isin(splits)]
    rows, kept = [], {}
    for row in tqdm(positives.itertuples(), total=len(positives), desc=config["experiment_id"], leave=False):
        with Image.open(Path(dataset_root) / row.image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = transform(image).to(device)
        result = gradcam(model, tensor, layer)
        if abs(result["probability"] - row.y_score) > .002:
            raise ValueError(f"{config['experiment_id']}: checkpoint does not reproduce {row.sample_id}")
        alignment = bbox_alignment(result["heatmap"], boxes[row.sample_id], width, height)
        rows.append(dict(experiment_id=config["experiment_id"], image_size=config["image_size"],
                         sample_id=row.sample_id, split=row.split,
                         outcome="TP" if row.y_score >= row.threshold else "FN",
                         probability=row.y_score, constant_map=result["constant"], **alignment))
        if row.sample_id in keep:
            kept[row.sample_id] = dict(heatmap=result["heatmap"], probability=row.y_score,
                                       detected=row.y_score >= row.threshold, mass_in_boxes=alignment["mass_in_boxes"])
    return pd.DataFrame(rows), dict(experiment_id=config["experiment_id"], family=family,
                                    checkpoint_sha256=sha256(checkpoint)), kept


def summarize(detail):
    """Per model/split: detection rate and how much Grad-CAM mass falls on the D40 boxes."""
    groups = detail.groupby(["experiment_id", "split"], sort=False)
    summary = groups.apply(lambda g: pd.Series(dict(
        positives=len(g),
        recall=(g.outcome == "TP").mean(),
        box_area_fraction=g.box_area_fraction.median(),
        mass_in_boxes=g.mass_in_boxes.median(),
        mass_over_area=g.mass_over_area.median(),
        peak_in_box=g.peak_in_box.astype(float).mean(),
        peak_in_box_TP=g.loc[g.outcome == "TP", "peak_in_box"].astype(float).mean(),
        peak_in_box_FN=g.loc[g.outcome == "FN", "peak_in_box"].astype(float).mean(),
    )), include_groups=False)
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=2, help="Random D40 images per split for the figure")
    args = parser.parse_args()

    frame, release = frozen_frame()
    annotations = pd.read_csv(MANIFESTS / "annotations.csv")
    d40 = annotations[annotations["class"] == "D40"]
    boxes = {sid: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy() for sid, g in d40.groupby("sample_id")}
    examples = pick_examples(frame, args.splits, args.examples)
    details, provenance, maps = [], [], {}
    for experiment in args.experiments:
        checkpoint = args.checkpoints / experiment.parent.name / experiment.name / "best.pt"
        detail, info, kept = explain_positives(experiment, checkpoint, frame, release, boxes, args.splits,
                                               args.dataset_root, torch.device(args.device),
                                               keep=set(examples.sample_id))
        details.append(detail)
        provenance.append(info)
        maps[info["experiment_id"]] = kept
        print(f"{info['experiment_id']}: {len(detail)} D40 images explained")
    detail = pd.concat(details, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output / "per_image.csv", index=False)
    if len(examples):
        render_examples(examples, maps, boxes, args.dataset_root, args.output / "same_images.png")
        examples.to_csv(args.output / "same_images.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        dataset_version=release["dataset_version"], splits=args.splits, target="D40 logit (output 0)",
        images="every official D40-positive image of the splits, identical for all models",
        metrics="mass_in_boxes = Grad-CAM mass inside the union of D40 boxes; mass_over_area > 1 means "
                "more mass than a uniform map; peak_in_box = maximum lies inside a D40 box",
        models=provenance), indent=2) + "\n")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
