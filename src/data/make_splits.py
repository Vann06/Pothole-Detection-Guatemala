"""Create dataset_v1 once, or verify the frozen manifests with --check."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data/manifests"
CONFIG = ROOT / "configs/dataset_v1.json"
LOCK = MANIFESTS / "dataset_v1.lock.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def road_status(images):
    return images.apply(lambda r: "pothole" if r.has_D40 else
                        "other_damage" if r.has_any_damage else "no_annotated_damage", axis=1)


def validate(images, annotations, checks, splits, config):
    assert images.sample_id.is_unique and images.image_path.is_unique
    assert checks.sample_id.is_unique and splits.sample_id.is_unique
    assert set(images.sample_id) == set(checks.sample_id) == set(splits.sample_id)
    assert set(annotations.sample_id) <= set(images.sample_id)
    for label in ("D00", "D10", "D20", "D40"):
        counts = annotations[annotations["class"] == label].groupby("sample_id").size()
        assert (images.sample_id.map(counts).fillna(0).to_numpy() == images[f"num_{label}"].to_numpy()).all()
        assert (images[f"has_{label}"] == (images[f"num_{label}"] > 0).astype(int)).all()
    counts = annotations.groupby("sample_id").size()
    assert (images.sample_id.map(counts).fillna(0).to_numpy() == images.num_objects.to_numpy()).all()
    assert (images.has_any_damage == (images.num_objects > 0).astype(int)).all()
    assert (images.is_empty_annotation == (images.num_objects == 0).astype(int)).all()
    data = images.merge(splits, validate="one_to_one").merge(checks, validate="one_to_one")
    allowed = {"train", "val", "test_id", "test_ood", "test_perspective"}
    assert set(data.split) == allowed
    assert (data.split.eq("test_ood") == data.country.eq(config["ood_country"])).all()
    assert (data.split.eq("test_perspective") == data.country.eq(config["perspective_country"])).all()
    assert data.groupby("image_sha256").split.nunique().max() == 1, "Duplicate leakage"
    assert data.groupby("image_sha256").country.nunique().max() == 1, "Cross-domain duplicates"
    assert data.groupby("image_sha256").has_D40.nunique().max() == 1, "Conflicting duplicate targets"
    for split in allowed:
        assert set(data.loc[data.split == split, "has_D40"]) == {0, 1}
    return data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="Verify frozen manifests without modifying files")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    images = pd.read_csv(MANIFESTS / "images.csv", low_memory=False)
    annotations = pd.read_csv(MANIFESTS / "annotations.csv")
    checks = pd.read_csv(MANIFESTS / "file_checksums.csv")
    if args.check:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        for relative, expected in lock["sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Frozen file changed: {relative}")
        splits = pd.read_csv(MANIFESTS / "splits.csv")
        validate(images, annotations, checks, splits, config)
        print("dataset_v1 OK: hashes, labels, coverage, domains and duplicate separation")
        return
    if LOCK.exists() or (MANIFESTS / "splits.csv").exists():
        raise FileExistsError("Release already exists. Use --check; changes require a new dataset version.")
    assert abs(sum(config[k] for k in ("train_fraction", "val_fraction", "test_id_fraction")) - 1) < 1e-9
    data = images.merge(checks, on="sample_id", validate="one_to_one")
    data["road_status"] = road_status(data)
    assert data.groupby("image_sha256").country.nunique().max() == 1
    assert data.groupby("image_sha256").road_status.nunique().max() == 1
    data["split"] = ""
    data.loc[data.country == config["ood_country"], "split"] = "test_ood"
    data.loc[data.country == config["perspective_country"], "split"] = "test_perspective"
    groups = data[data.split == ""].sort_values("sample_id").drop_duplicates("image_sha256").copy()
    strata = groups.country + "__" + groups.road_status
    train, held = train_test_split(groups, test_size=1 - config["train_fraction"],
                                  stratify=strata, random_state=config["seed"])
    val, test = train_test_split(held, test_size=config["test_id_fraction"] /
                                (config["val_fraction"] + config["test_id_fraction"]),
                                stratify=held.country + "__" + held.road_status,
                                random_state=config["seed"])
    mapping = {row.image_sha256: name for name, frame in (("train", train), ("val", val), ("test_id", test))
               for row in frame.itertuples()}
    remaining = data.split == ""
    data.loc[remaining, "split"] = data.loc[remaining, "image_sha256"].map(mapping)
    splits = data[["sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
    merged = validate(images, annotations, checks, splits, config)
    splits.to_csv(MANIFESTS / "splits.csv", index=False, lineterminator="\n")
    merged["road_status"] = road_status(merged)
    summary = merged.groupby(["split", "country", "road_status"]).size().rename("images").reset_index()
    summary.to_csv(ROOT / "reports/tables/split_summary.csv", index=False, lineterminator="\n")
    files = ["configs/dataset_v1.json", "src/data/make_splits.py", "src/data/audit_release.py",
             "data/manifests/images.csv", "data/manifests/annotations.csv",
             "data/manifests/file_checksums.csv", "data/manifests/splits.csv",
             "reports/tables/release_audit.json", "reports/tables/split_summary.csv"]
    train_data = merged[merged.split == "train"]
    pos = int(train_data.has_D40.sum())
    lock = {"version": config["version"], "config": config,
            "split_counts": merged.split.value_counts().sort_index().to_dict(),
            "train_positive": pos, "train_negative": len(train_data) - pos,
            "train_pos_weight": (len(train_data) - pos) / pos,
            "sha256": {f: digest(ROOT / f) for f in files}}
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
