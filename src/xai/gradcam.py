"""Manual Grad-CAM for the D40 logit of ResNet18 or EfficientNet-B0, including false negatives."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image

from src.data.dataset import DATASET_ROOT, MANIFESTS, build_transforms
from src.evaluation.cross_domain import frozen_frame, load_experiment, select_cases, sha256, validate_predictions
from src.evaluation.inference import load_checkpoint, load_guatemala
from src.models.classifiers import CLASSIFIER_ARCHITECTURES


def gradcam(model, image, target_layer=None):
    """Return input-size and native maps for logit 0; layer4 is the ResNet-compatible default.

    alpha_k = mean_ij(d logit_D40 / d A_kij); L = ReLU(sum_k alpha_k A_k).
    Autograd reads activation derivatives without accumulating parameter .grad. Input gradients
    are enabled explicitly, so fully frozen encoders work too. Model must already be in eval mode.
    """
    if model.training or image.ndim != 3:
        raise ValueError("Grad-CAM requires an eval model and a CHW input image")
    capture = {}
    layer = model.layer4 if target_layer is None else target_layer
    handle = layer.register_forward_hook(lambda module, inputs, output: capture.update(activation=output))
    try:
        with torch.enable_grad():
            logits = model(image.detach().unsqueeze(0).requires_grad_(True))
            activation = capture["activation"]
            gradient = torch.autograd.grad(logits[0, 0], activation)[0]
            weights = gradient.mean((2, 3), keepdim=True)
            raw = (weights*activation).sum(1)[0]
            up = F.interpolate(raw.relu()[None, None], image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            span = up.max()-up.min()
            heatmap = (up-up.min())/span if span > 0 else torch.zeros_like(up)
            return dict(heatmap=heatmap.detach().cpu().numpy(), raw=raw.detach().cpu().numpy(),
                        logit=float(logits[0, 0].detach()), probability=float(logits[0, 0].detach().sigmoid()),
                        constant=bool(span == 0))
    finally:
        handle.remove()


def bbox_alignment(heatmap, boxes, original_width, original_height):
    """Mass inside UNION of D40 rectangles after the same anisotropic resize as evaluation.

    Uses the manifest's continuous xmin/ymin/xmax/ymax convention (no invented pixel offset).
    This is a diagnostic, not segmentation IoU or causal fidelity. Zero maps are undefined.
    """
    height, width = heatmap.shape
    mask = np.zeros_like(heatmap, dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        if not (0 <= x1 < x2 <= original_width and 0 <= y1 < y2 <= original_height):
            raise ValueError("D40 bounding box is outside the original image")
        left, right = int(np.floor(x1*width/original_width)), int(np.ceil(x2*width/original_width))
        top, bottom = int(np.floor(y1*height/original_height)), int(np.ceil(y2*height/original_height))
        mask[top:bottom, left:right] = True
    mass = float(heatmap.sum())
    valid = mass > 0 and mask.any()
    fraction = float(heatmap[mask].sum()/mass) if valid else None
    return dict(box_area_fraction=float(mask.mean()), mass_in_boxes=fraction,
                mass_over_area=fraction/float(mask.mean()) if valid else None,
                peak_in_box=bool(mask.flat[int(heatmap.argmax())]) if valid else None)


def render_case(rgb, result, boxes, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(rgb); axes[0].set_title("Imagen y cajas D40")
    for x1,y1,x2,y2 in boxes:
        axes[0].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color="lime", linewidth=1.5))
    axes[1].imshow(rgb); axes[1].imshow(result["heatmap"], cmap="jet", alpha=.45, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM del logit D40")
    axes[2].imshow(np.maximum(result["raw"], 0), cmap="jet")
    axes[2].set_title("Mapa nativo" + (" · constante" if result["constant"] else ""))
    for ax in axes: ax.axis("off")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guatemala-manifest", type=Path)
    parser.add_argument("--guatemala-predictions", type=Path)
    parser.add_argument("--per-kind", type=int, default=3)
    args = parser.parse_args()
    family = args.experiment.parent.name
    if family not in CLASSIFIER_ARCHITECTURES:
        parser.error("Grad-CAM explains supervised D40 logits, not VAE reconstruction error")
    torch.set_num_threads(4)
    frame, release = frozen_frame()
    predictions, config, _ = load_experiment(args.experiment, frame, release)
    model = load_checkpoint(args.checkpoint, config, family)
    layer_name = "layer4" if family == "resnet" else "features.8"
    target_layer = model.layer4 if family == "resnet" else model.features[-1]
    if bool(args.guatemala_manifest) != bool(args.guatemala_predictions):
        parser.error("Both Guatemala manifest and predictions are required")
    external = args.guatemala_manifest is not None
    if external:
        gt = load_guatemala(args.guatemala_manifest, frame)
        predictions = validate_predictions(pd.read_csv(args.guatemala_predictions), gt,
                                           config["experiment_id"], release["dataset_version"],
                                           predictions.threshold.iloc[0])
    cases = select_cases(predictions, args.per_kind)
    annotations = pd.read_csv(MANIFESTS / "annotations.csv")
    transform = build_transforms(size=config["image_size"])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in cases.itertuples():
        with Image.open(args.dataset_root / row.image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = transform(image)
            rgb = build_transforms(size=config["image_size"], normalize=False)(image).permute(1, 2, 0).numpy()
        result = gradcam(model, tensor, target_layer)
        if abs(result["probability"]-row.y_score) > .002:
            raise ValueError(f"Checkpoint/preprocessing fails to reproduce probability for {row.sample_id}")
        boxes = (np.empty((0, 4)) if external else annotations.loc[
            (annotations.sample_id == row.sample_id) & (annotations['class'] == "D40"),
            ["xmin", "ymin", "xmax", "ymax"]].to_numpy())
        alignment = bbox_alignment(result["heatmap"], boxes, width, height)
        scaled = boxes * np.array([config["image_size"]/width, config["image_size"]/height]*2)
        # Filename depends on trusted sequence index, not arbitrary externally supplied IDs.
        filename = f"case_{len(records):03d}.png"
        render_case(rgb, result, scaled, f"{row.sample_id} · {row.split} · {row.outcome} · p={row.y_score:.4f}", args.output / filename)
        np.save(args.output / filename.replace(".png", ".npy"), result["heatmap"])
        records.append(dict(sample_id=row.sample_id, split=row.split, outcome=row.outcome,
                            probability=result["probability"], threshold=row.threshold, figure=filename,
                            constant_map=result["constant"], bbox_status="not provided" if external else "official manifest",
                            **alignment, interpretation="pending visual review"))
    pd.DataFrame(records).to_csv(args.output / "cases.csv", index=False)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        experiment_id=config["experiment_id"], family=family, target_layer=layer_name,
        checkpoint_sha256=sha256(args.checkpoint), seed=42,
        target="D40 logit (output 0), including FN", selection="uniform per split/outcome",
        unavailable_groups="see case_coverage.csv; never duplicated or manufactured",
        external_manifest_sha256=sha256(args.guatemala_manifest) if external else None), indent=2)+"\n")
    print(f"Generated {len(records)} cases; inspect images before interpreting them.")


if __name__ == "__main__":
    main()
