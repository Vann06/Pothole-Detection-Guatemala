"""Audit prediction contracts and consolidate frozen-threshold cross-domain evaluation.

No training, threshold fitting or model selection on test data occurs here. The validation
threshold is recomputed only to AUDIT the submitted threshold, never to replace it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

from src.data.dataset import MANIFESTS, ROOT, load_split_frame, release_info
from src.evaluation.breakdown import damage_group, group_report, safe_score
from src.evaluation.metrics import PREDICTION_COLUMNS, best_f1_threshold, split_metrics
from src.models.classifiers import CLASSIFIER_ARCHITECTURES
from sklearn.metrics import average_precision_score

EVAL_SPLITS = ("val", "test_id", "test_ood", "test_perspective")
OUTCOMES = ("TP", "TN", "FP", "FN")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_frame(manifests=MANIFESTS):
    """Check labels AND annotations, not only split membership, against the release lock."""
    manifests = Path(manifests)
    release = release_info(manifests)
    lock = json.loads((manifests / "dataset_v1.lock.json").read_text())
    for name in ("images.csv", "annotations.csv"):
        if sha256(manifests / name) != lock["sha256"][f"data/manifests/{name}"]:
            raise ValueError(f"{name} differs from the frozen release")
    return load_split_frame(manifests=manifests), release


def validate_predictions(predictions, frame, experiment_id, dataset_version, threshold=None):
    """Reject missing/duplicated cases, mislabeled domains, nonfinite scores and test retuning."""
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty or predictions.sample_id.isna().any() or not predictions.sample_id.is_unique:
        raise ValueError("Predictions must contain unique, nonempty sample IDs")
    if not frame.sample_id.is_unique or set(predictions.sample_id) != set(frame.sample_id):
        raise ValueError("Prediction coverage does not exactly match the evaluation manifest")
    if not predictions.experiment_id.eq(experiment_id).all():
        raise ValueError("Mixed/wrong experiment_id")
    if not predictions.dataset_version.eq(dataset_version).all():
        raise ValueError("Wrong dataset_version")
    if not predictions.y_true.isin([0, 1]).all():
        raise ValueError("y_true must be binary")
    if not np.isfinite(predictions[["y_score", "threshold"]].to_numpy(dtype=float)).all():
        raise ValueError("Scores and thresholds must be finite")
    if predictions.threshold.nunique() != 1:
        raise ValueError("Threshold must remain fixed across all domains")
    if threshold is not None and not np.isclose(predictions.threshold.iloc[0], threshold, rtol=1e-7, atol=1e-12):
        raise ValueError("External threshold differs from the validation threshold")
    # Only trust metadata from the frozen manifest, never duplicated CSV metadata.
    joined = predictions[list(PREDICTION_COLUMNS)].merge(
        frame, on="sample_id", suffixes=("", "_official"), validate="one_to_one")
    if not joined.split.eq(joined.split_official).all() or not joined.y_true.eq(joined.has_D40).all():
        raise ValueError("Predicted labels/splits disagree with the manifest")
    joined = joined.drop(columns="split_official")
    joined["y_pred"] = (joined.y_score >= joined.threshold).astype(int)
    joined["outcome"] = np.select(
        [(joined.y_true == 1) & (joined.y_pred == 1),
         (joined.y_true == 0) & (joined.y_pred == 0), joined.y_pred == 1],
        ["TP", "TN", "FP"], default="FN")
    joined["group"] = damage_group(joined)
    return joined


def load_experiment(directory, frame, release):
    """Use only complete runs with validation evidence and matching split hashes."""
    directory = Path(directory)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    if config.get("limit") is not None:
        raise ValueError("Smoke/limited runs cannot enter the final comparison")
    for key, value in release.items():
        if config.get(key) != value:
            raise ValueError(f"Experiment {key} does not match the release")
    identifier = config["experiment_id"]
    if identifier != directory.name:
        raise ValueError("Experiment directory and config ID disagree")
    predictions = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
    table = validate_predictions(predictions, frame[frame.split.isin(EVAL_SPLITS)],
                                 identifier, release["dataset_version"])
    val = table[table.split == "val"]
    if val.y_true.nunique() != 2:
        raise ValueError("Validation needs both classes for threshold/model selection")
    expected = best_f1_threshold(val.y_true.to_numpy(), val.y_score.to_numpy())
    if not np.isclose(expected, table.threshold.iloc[0], rtol=1e-7, atol=1e-12):
        raise ValueError("Threshold does not match the documented max-val-F1 rule")
    family = directory.parent.name
    if family not in (*CLASSIFIER_ARCHITECTURES, "vae"):
        raise ValueError("Expected experiment under resnet/, efficientnet/ or vae/")
    if family in CLASSIFIER_ARCHITECTURES:
        if not table.y_score.between(0, 1).all():
            raise ValueError("Classifier scores must be D40 probabilities")
        if config.get("family", "resnet") != family or config.get(
                "architecture", "resnet18") != CLASSIFIER_ARCHITECTURES[family]:
            raise ValueError("Classifier architecture disagrees with experiment family")
    metadata = json.loads((directory / "metrics.json").read_text())
    if directory.parent.name == "vae":
        primary = metadata["primary_score"]
        # Audit that the exported score was selected by val AP, not by external performance.
        aps = {name: average_precision_score(predictions.loc[predictions.split == "val", "y_true"],
                                             predictions.loc[predictions.split == "val", f"y_score_{name}"])
               for name in metadata["val_pr_auc"]}
        if aps[primary] < max(aps.values()) - 1e-12:
            raise ValueError("VAE primary score was not selected on validation AP")
        if not np.allclose(predictions.y_score, predictions[f"y_score_{primary}"], rtol=1e-7, atol=1e-12):
            raise ValueError("VAE exported y_score does not match its primary score")
    return table, config, metadata


def metric_row(table):
    scores = table.y_score.to_numpy()
    labels = table.y_true.to_numpy()
    metrics = split_metrics(labels, scores, float(table.threshold.iloc[0]))
    metrics.update(metrics.pop("confusion_matrix"))
    metrics["pr_auc"] = safe_score(average_precision_score, labels, scores)
    # Explicit denominators avoid interpreting perfect recall with almost every image flagged as success.
    metrics["fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) else None
    metrics["flagged_rate"] = float(table.y_pred.mean())
    metrics["threshold"] = float(table.threshold.iloc[0])
    return metrics


def domain_deltas(metrics):
    """Positive delta = degradation from internal test to external domain (not relative percent)."""
    rows = []
    for (family, experiment), group in metrics.groupby(["family", "experiment_id"]):
        internal = group[group.split == "test_id"].iloc[0]
        for external in group[~group.split.isin(["val", "test_id"])].itertuples():
            rows.append(dict(family=family, experiment_id=experiment, domain=external.split,
                             f1_id=internal.f1, f1_external=external.f1,
                             delta_f1=internal.f1-external.f1,
                             recall_id=internal.recall, recall_external=external.recall,
                             delta_recall=internal.recall-external.recall,
                             prevalence_id=internal.prevalence, prevalence_external=external.prevalence))
    return pd.DataFrame(rows)


def select_cases(table, per_kind=3, seed=42):
    """Uniform reproducible examples per domain/outcome; stable under CSV reordering; no cherry-picking."""
    if per_kind < 1:
        raise ValueError("per_kind must be positive")
    rows = []
    for _, group in table[table.split != "val"].groupby(["split", "outcome"], sort=True):
        rows.append(group.sort_values("sample_id").sample(n=min(per_kind, len(group)), random_state=seed))
    return pd.concat(rows, ignore_index=True) if rows else table.iloc[:0].copy()


def plot_experiment(table, directory):
    """Counts and precision-recall curves use identical fixed predictions, with per-domain prevalence."""
    tests = [(name, group) for name, group in table.groupby("split", sort=True) if name != "val"]
    fig, axes = plt.subplots(1, len(tests), figsize=(4.1*len(tests), 3.6), squeeze=False)
    for ax, (split, group) in zip(axes[0], tests):
        m = metric_row(group)
        matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max()/2 else "black", fontsize=12)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no-D40", "D40"],
               yticklabels=["no-D40", "D40"], xlabel="Predicción", ylabel="Etiqueta real",
               title=f"{split}\nN={len(group)}, F1={m['f1']:.3f}")
    fig.tight_layout(); fig.savefig(directory / "confusion.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, len(tests), figsize=(4.1*len(tests), 3.4), squeeze=False)
    for ax, (split, group) in zip(axes[0], tests):
        if group.y_true.nunique() == 2:
            precision, recall, _ = precision_recall_curve(group.y_true, group.y_score)
            ap = average_precision_score(group.y_true, group.y_score)
            ax.plot(recall, precision, label=f"AP={ap:.3f}")
        ax.axhline(group.y_true.mean(), ls="--", color="gray", label="Prevalencia")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precisión", title=split)
        ax.legend()
    fig.tight_layout(); fig.savefig(directory / "precision_recall.png", dpi=150); plt.close(fig)


def run(experiments=ROOT / "reports/experiments", output=ROOT / "reports/xai_evaluation", manifests=MANIFESTS,
        guatemala_manifest=None, guatemala_results=None):
    output, experiments = Path(output), Path(experiments)
    frame, release = frozen_frame(manifests)
    if bool(guatemala_manifest) != bool(guatemala_results):
        raise ValueError("Provide both Guatemala manifest and results directory")
    external = None
    if guatemala_manifest:
        from src.evaluation.inference import load_guatemala
        external = load_guatemala(guatemala_manifest, frame)
    records, sources, latency, cases, counts, countries, audit = [], [], [], [], [], [], []
    identifiers = set()
    for family in (*CLASSIFIER_ARCHITECTURES, "vae"):
        for path in sorted((experiments / family).glob("*/predictions.csv")):
            config = yaml.safe_load(path.with_name("config.yaml").read_text())
            if config.get("limit") is not None:
                continue
            table, config, metadata = load_experiment(path.parent, frame, release)
            identifier = config["experiment_id"]
            if identifier in identifiers:
                raise ValueError("experiment_id must be unique across families")
            identifiers.add(identifier)
            external_path = Path(guatemala_results) / identifier / "predictions.csv" if external is not None else None
            if external_path is not None and external_path.exists():
                evaluated = validate_predictions(pd.read_csv(external_path), external, identifier,
                                                 release["dataset_version"], table.threshold.iloc[0])
                table = pd.concat([table, evaluated], ignore_index=True)
            folder = output / identifier
            folder.mkdir(parents=True, exist_ok=True)
            for split, group in table.groupby("split", sort=True):
                recomputed = metric_row(group)
                records.append(dict(family=family, experiment_id=identifier, split=split, **recomputed))
                if split in metadata["splits"]:
                    archived = metadata["splits"][split]
                    if family == "vae":
                        archived = archived[metadata["primary_score"]]
                    archived = {**archived, **archived["confusion_matrix"]}
                    for key in ("tp", "fp", "tn", "fn", "f1", "precision", "recall", "pr_auc"):
                        if not np.isclose(recomputed[key], archived[key], rtol=1e-9, atol=1e-12):
                            audit.append(dict(experiment_id=identifier, split=split, metric=key,
                                              archived=archived[key], recomputed=recomputed[key]))
                for country, sub in group.groupby("country"):
                    countries.append(dict(experiment_id=identifier, split=split, country=country, **metric_row(sub)))
                for outcome in OUTCOMES:
                    counts.append(dict(experiment_id=identifier, split=split, outcome=outcome,
                                       available=int(group.outcome.eq(outcome).sum())))
            details = {}
            for split, g in table.groupby("split"):
                if split == "test_guatemala":
                    # External protocol has no required per-damage-class labels. Do not invent them.
                    details[split] = {column: {
                        str(value): metric_row(sub) for value, sub in g.groupby(column, dropna=False)}
                        for column in ("group", "lighting", "weather", "camera", "capture_session", "road_segment")}
                else:
                    details[split] = group_report(g, g.y_score, g.threshold.iloc[0])
            (folder / "error_groups.json").write_text(json.dumps(details, indent=2, allow_nan=False)+"\n")
            chosen = select_cases(table)
            chosen["visual_interpretation"] = ""  # Requires the original images, never inferred from a score.
            cases.append(chosen)
            plot_experiment(table, folder)
            sources.append(dict(family=family, experiment_id=identifier,
                                predictions_sha256=sha256(path), config_sha256=sha256(path.with_name("config.yaml")),
                                external_predictions_sha256=sha256(external_path) if external_path and external_path.exists() else None))
            for device in ("cpu", "cuda"):
                ms = metadata.get("latency", {}).get(f"{device}_ms")
                if ms is not None:
                    latency.append(dict(experiment_id=identifier, device=device, reported_median_ms=ms,
                                        image_size=config["image_size"],
                                        scope=metadata["latency"].get("includes", "forward only; see training pipeline"),
                                        comparability="unverified: hardware/warmup/sample count not recorded"))
    if not records:
        raise ValueError("No complete experiments; cannot manufacture a comparison")
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "metrics.csv", index=False)
    deltas = domain_deltas(metrics); deltas.to_csv(output / "domain_shift.csv", index=False)
    pd.DataFrame(countries).to_csv(output / "by_country.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "case_coverage.csv", index=False)
    pd.DataFrame(audit, columns=["experiment_id", "split", "metric", "archived", "recomputed"]).to_csv(
        output / "archive_discrepancies.csv", index=False)
    pd.concat(cases).to_csv(output / "cases.csv", index=False)
    pd.DataFrame(latency, columns=["experiment_id", "device", "reported_median_ms", "image_size",
                                  "scope", "comparability"]).to_csv(output / "latency_reported.csv", index=False)
    resnet = metrics[(metrics.family == "resnet") & (metrics.split == "val")]
    selected = None if resnet.empty else resnet.sort_values(["pr_auc", "experiment_id"], ascending=[False, True]).iloc[0].experiment_id
    supervised = metrics[metrics.family.isin(CLASSIFIER_ARCHITECTURES) & (metrics.split == "val")]
    best = None if supervised.empty else supervised.sort_values(
        ["pr_auc", "experiment_id"], ascending=[False, True]).iloc[0]
    status = dict(**release, sources=sources, selection="Classifiers: maximum validation AP; lexical ID tie-break",
                  selected_resnet=selected,
                  selected_classifier=None if best is None else dict(family=best.family, experiment_id=best.experiment_id),
                  guatemala_evaluated=metrics.loc[metrics.split == "test_guatemala", "experiment_id"].tolist(),
                  guatemala_pending=sorted(identifiers-set(metrics.loc[metrics.split == "test_guatemala", "experiment_id"])),
                  external_manifest_sha256=sha256(guatemala_manifest) if guatemala_manifest else None,
                  gradcam="pending checkpoint and original images" if best is None else "run src.xai.gradcam with selected checkpoint",
                  latency="historical medians only; rerun benchmark on the same machine for comparison")
    (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for index, (experiment, rows) in enumerate(metrics[metrics.split != "val"].groupby("experiment_id", sort=True)):
        style = dict(marker="o" if index % 2 == 0 else "x", markersize=8, fillstyle="none",
                     linestyle="-" if index % 2 == 0 else "--", label=experiment)
        axes[0].plot(rows.split, rows.f1, **style)
        axes[1].plot(rows.split, rows.recall, **style)
    for ax, name in zip(axes, ("F1", "Recall")):
        ax.set(ylabel=name, ylim=(0, 1.05)); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)
    print(metrics[["experiment_id", "split", "n", "precision", "recall", "f1", "pr_auc", "flagged_rate"]].to_string(index=False))
    return metrics, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=ROOT / "reports/experiments")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/xai_evaluation")
    parser.add_argument("--guatemala-manifest", type=Path)
    parser.add_argument("--guatemala-results", type=Path, help="Directory containing <experiment_id>/predictions.csv")
    args = parser.parse_args()
    run(args.experiments, args.output, guatemala_manifest=args.guatemala_manifest, guatemala_results=args.guatemala_results)
