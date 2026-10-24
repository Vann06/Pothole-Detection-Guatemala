"""Audit the actual labeled RDD2022 files before freezing a release."""
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/RDD2022"
MANIFESTS = ROOT / "data/manifests"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(row):
    result = {"sample_id": row["sample_id"]}
    image = RAW / row["image_path"]
    xml = RAW / row["xml_path"]
    with Image.open(image) as picture:
        size = picture.size
        picture.verify()
    if size != (int(row["width"]), int(row["height"])):
        raise ValueError(f"Image dimensions disagree: {row['sample_id']}")
    tree = ET.parse(xml).getroot()
    labels = Counter(obj.findtext("name", "").strip() for obj in tree.findall("object"))
    if sum(labels.values()) != int(row["num_objects"]):
        raise ValueError(f"Object count disagrees: {row['sample_id']}")
    for label in ("D00", "D10", "D20", "D40"):
        if labels[label] != int(row[f"num_{label}"]):
            raise ValueError(f"Class count disagrees: {row['sample_id']} {label}")
    if int(row["has_any_damage"]) != int(bool(labels)):
        raise ValueError(f"Damage flag disagrees: {row['sample_id']}")
    result.update(image_sha256=sha256(image), xml_sha256=sha256(xml),
                  image_bytes=image.stat().st_size, xml_bytes=xml.stat().st_size)
    return result


def main():
    if (MANIFESTS / "dataset_v1.lock.json").exists():
        raise FileExistsError("dataset_v1 is frozen; use share_dataset.py --verify instead.")
    with (MANIFESTS / "images.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(tqdm(pool.map(inspect, rows), total=len(rows), desc="Audit JPG/XML", mininterval=2))
    checks.sort(key=lambda row: row["sample_id"])
    with (MANIFESTS / "file_checksums.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    counts = Counter(row["image_sha256"] for row in checks)
    report = {
        "images_verified": len(checks), "xml_verified": len(checks),
        "image_bytes": sum(row["image_bytes"] for row in checks),
        "xml_bytes": sum(row["xml_bytes"] for row in checks),
        "exact_duplicate_groups": sum(n > 1 for n in counts.values()),
        "exact_duplicate_extra_images": sum(n - 1 for n in counts.values()),
        "checks": ["Pillow verify", "image dimensions vs manifest", "XML parsing",
                   "XML object and canonical class counts vs manifest", "SHA256 JPG and XML"],
        "limitations": "No exhaustive pixel decoding, near-duplicate or same-road/sequence grouping; no sequence metadata available."
    }
    target = ROOT / "reports/tables/release_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
