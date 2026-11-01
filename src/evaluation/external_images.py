"""Score a small folder of external photos with trained ResNet checkpoints (qualitative check, not a test split).

Labels come only from the folder name (``con_bache/`` = has a pothole, ``sin_bache/`` = no pothole). Each model
uses its own frozen validation threshold and the same evaluation preprocessing as the protocol; nothing is tuned
here. This does NOT replace the formal Guatemala evaluation (``src.evaluation.inference --guatemala-manifest``),
which requires a reviewed manifest with provenance; no AP/F1 is reported because the sample is tiny and unreviewed.

    python -m src.evaluation.external_images --images data/external/sample_photos --device cuda
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.data.dataset import build_transforms
from src.evaluation.inference import load_checkpoint
from src.xai.gradcam import gradcam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_FOLDERS = {"con_bache": 1, "sin_bache": 0}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EXPERIMENTS = ["exp5_res384_seed43", "exp5_res384", "exp5_res384_seed44", "exp3_augment_fp32",
                       "exp2_pretrained_fp32"]


def collect_images(root):
    """Return (relative path, label) rows; only the two label folders are accepted."""
    root = Path(root)
    unknown = [p.name for p in root.iterdir() if p.is_dir() and p.name not in LABEL_FOLDERS]
    if unknown:
        raise ValueError(f"Unknown label folders {unknown}; use {sorted(LABEL_FOLDERS)}")
    rows = [dict(image=path.relative_to(root).as_posix(), label=label)
            for folder, label in LABEL_FOLDERS.items() if (root / folder).is_dir()
            for path in sorted((root / folder).iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not rows:
        raise ValueError(f"No images under {root}/{{{','.join(LABEL_FOLDERS)}}}")
    return pd.DataFrame(rows)


def outcome_counts(predictions):
    """TP/FN/FP/TN per experiment from boolean predictions and 0/1 labels."""
    p, y = predictions.predicted.astype(bool), predictions.label.astype(bool)
    outcome = np.select([p & y, ~p & y, p & ~y], ["TP", "FN", "FP"], "TN")
    table = pd.crosstab(predictions.experiment_id, outcome).reindex(columns=["TP", "FN", "FP", "TN"], fill_value=0)
    return table.rename_axis(index="experiment_id", columns=None)


def load_run(experiment, project_root=PROJECT_ROOT):
    run_dir = project_root / "reports/experiments/resnet" / experiment
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    threshold = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["threshold"]
    model = load_checkpoint(project_root / "checkpoints/resnet" / experiment / "best.pt", config, "resnet")
    return model, config, threshold


def score(images, root, experiments, selected, device, project_root=PROJECT_ROOT):
    rows, heatmaps = [], {}
    for experiment in experiments:
        model, config, threshold = load_run(experiment, project_root)
        model = model.to(device)
        transform = build_transforms(size=config["image_size"])
        for item in images.itertuples():
            x = transform(Image.open(Path(root) / item.image).convert("RGB")).to(device)
            if experiment == selected:
                cam = gradcam(model, x)
                heatmaps[item.image], probability = cam["heatmap"], cam["probability"]
            else:
                with torch.no_grad():
                    probability = float(model(x[None])[0, 0].sigmoid())
            rows.append(dict(image=item.image, label=item.label, experiment_id=experiment, probability=probability,
                             threshold=threshold, predicted=probability >= threshold))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), heatmaps


def render(predictions, heatmaps, root, selected, label, path):
    sel = predictions[(predictions.experiment_id == selected) & (predictions.label == label)].reset_index(drop=True)
    fig, axes = plt.subplots(2, len(sel), figsize=(3.2 * len(sel), 9), squeeze=False)
    for col, row in sel.iterrows():
        image = Image.open(Path(root) / row.image).convert("RGB")
        heat = np.asarray(Image.fromarray((heatmaps[row.image] * 255).astype(np.uint8)).resize(
            image.size, Image.BILINEAR)) / 255  # back to the photo's own aspect ratio for display
        for r, ax in enumerate(axes[:, col]):
            ax.imshow(image)
            if r == 1:
                ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.axis("off")
        correct = bool(row.predicted) == bool(label)
        axes[0, col].set_title(f"{Path(row.image).stem}\np={row.probability:.3f} -> "
                               f"{'bache' if row.predicted else 'sin bache'}", fontsize=10,
                               color="green" if correct else "red")
    kind = "con bache" if label else "sin bache"
    fig.suptitle(f"{selected} · fotos {kind} · umbral val {sel.threshold.iloc[0]:.3f} · "
                 f"verde = correcto · fila inferior: Grad-CAM layer4", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90, pil_kwargs=dict(quality=85))  # JPEG: these are photos, PNG would be ~4x larger
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/external/sample_photos")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/external_images")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--selected", help="Model explained with Grad-CAM; default: selected_resnet in status.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    selected = args.selected or json.loads(
        (PROJECT_ROOT / "reports/xai_evaluation/status.json").read_text(encoding="utf-8"))["selected_resnet"]
    experiments = list(dict.fromkeys([selected, *args.experiments]))
    images = collect_images(args.images)
    predictions, heatmaps = score(images, args.images, experiments, selected, args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    counts = outcome_counts(predictions)
    counts.to_csv(args.output / "outcomes.csv")
    for label in sorted(images.label.unique(), reverse=True):
        render(predictions, heatmaps, args.images, selected, label,
               args.output / f"gradcam_{'con' if label else 'sin'}_bache.jpg")
    (args.output / "status.json").write_text(json.dumps(dict(
        selected=selected, experiments=experiments, images=len(images),
        positives=int(images.label.sum()), negatives=int((images.label == 0).sum()),
        note="Qualitative check on unreviewed external photos; not a test split, no threshold tuning."),
        indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(predictions.pivot(index="image", columns="experiment_id", values="probability")[experiments].round(3))
    print(counts)
    print("results in", args.output)


if __name__ == "__main__":
    main()
