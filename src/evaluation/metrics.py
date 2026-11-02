"""Shared D40 metrics and the predictions.csv contract used by every model."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

# contrato de docs/HANDOFF.md, las columnas extra van siempre despues
PREDICTION_COLUMNS = ("sample_id", "split", "y_true", "y_score", "threshold", "experiment_id", "dataset_version")


def best_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def split_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(n=len(y_true), positives=int(y_true.sum()), prevalence=float(y_true.mean()),
                precision=precision, recall=recall, f1=f1,
                pr_auc=float(average_precision_score(y_true, y_score)),
                confusion_matrix={"tn": tn, "fp": fp, "fn": fn, "tp": tp})


def prediction_frame(ids, split, y_true, y_score, threshold, experiment_id, dataset_version):
    """Build the 7 contract columns in order; callers append their extra columns afterwards."""
    values = (ids, split, y_true, y_score, threshold, experiment_id, dataset_version)
    return pd.DataFrame(dict(zip(PREDICTION_COLUMNS, values)))
