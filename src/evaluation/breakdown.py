"""Model-agnostic error analysis by damage group, country and other-damage class.

Works on any score where higher means "more likely D40" (ResNet probability or VAE error), so it can be
reused on every predictions.csv merged with images.csv. Empty or single-class subsets return None.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

GROUPS = ("no_damage", "other", "pothole")
GROUP_PAIRS = (("no_damage", "other"), ("no_damage", "pothole"), ("other", "pothole"))
CANONICAL_OTHER = ("D00", "D10", "D20")
OVERLAP_NOTE = "images with several damage classes are counted once in each of their classes"


def damage_group(frame):
    """pothole = has_D40; other = some annotated damage but no D40; no_damage = nothing annotated."""
    labels = np.select([frame.has_D40.to_numpy() == 1, frame.has_any_damage.to_numpy() == 1],
                       ["pothole", "other"], "no_damage")
    return pd.Series(labels, index=frame.index, name="group")


def other_damage_long(frame):
    """One row per (sample_id, damage_class) for images without D40: D00/D10/D20 plus parsed other_labels."""
    negatives = frame[frame.has_D40 == 0]
    parts = [negatives.loc[negatives[f"has_{name}"] == 1, ["sample_id"]].assign(damage_class=name)
             for name in CANONICAL_OTHER]
    # other_labels viene como "D44;D50" o NaN
    extra = negatives[["sample_id"]].assign(
        damage_class=negatives.other_labels.fillna("").astype(str).str.split(";")).explode("damage_class")
    extra["damage_class"] = extra.damage_class.str.strip()
    parts.append(extra[extra.damage_class != ""])
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def safe_score(metric, y_true, y_score):
    """sklearn metric, or None when the subset is empty or has a single class."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return None
    return float(metric(y_true, y_score))


def summarize(scores, threshold):
    """flagged_rate = share at or above the threshold (false-positive rate for groups without D40)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return dict(n=0, mean=None, median=None, p90=None, flagged_rate=None)
    return dict(n=int(len(scores)), mean=float(scores.mean()), median=float(np.median(scores)),
                p90=float(np.quantile(scores, 0.9)), flagged_rate=float((scores >= threshold).mean()))


def pairwise_auroc(groups, scores):
    """AUROC of separating each pair of groups, the higher group being the positive one."""
    result = {}
    for low, high in GROUP_PAIRS:
        mask = groups.isin([low, high]).to_numpy()
        result[f"{low}_vs_{high}"] = dict(
            n_low=int((groups == low).sum()), n_high=int((groups == high).sum()),
            auroc=safe_score(roc_auc_score, (groups[mask] == high).astype(int), scores[mask]))
    return result


def group_report(frame, scores, threshold):
    """Score distribution and threshold behaviour by group, country and other-damage class."""
    frame = frame.reset_index(drop=True)
    scores = np.asarray(scores, dtype=float)
    groups = damage_group(frame)
    by_country = {}
    for country, index in frame.groupby("country").groups.items():
        rows = np.asarray(index)
        sub_groups, sub_scores = groups.iloc[rows].reset_index(drop=True), scores[rows]
        by_country[country] = dict(
            n={name: int((sub_groups == name).sum()) for name in GROUPS},
            pr_auc_D40=safe_score(average_precision_score, frame.has_D40.iloc[rows], sub_scores),
            pairwise_auroc=pairwise_auroc(sub_groups, sub_scores))
    scored = pd.DataFrame(dict(sample_id=frame.sample_id, score=scores))
    classes = {name: summarize(rows.score, threshold) for name, rows in
               other_damage_long(frame).merge(scored, on="sample_id").groupby("damage_class")}
    return dict(groups={name: summarize(scores[(groups == name).to_numpy()], threshold) for name in GROUPS},
                pairwise_auroc=pairwise_auroc(groups, scores), by_country=by_country,
                other_damage_classes=classes, note=OVERLAP_NOTE)
