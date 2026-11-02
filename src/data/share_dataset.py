"""Package the frozen labeled files by domain, or verify a recipient's copy."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/RDD2022"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", action="store_true", help="Create ZIPs in .dist/dataset_v1")
    parser.add_argument("--verify", action="store_true", help="Verify all local labeled JPG/XML SHA256 values")
    args = parser.parse_args()
    if args.package == args.verify:
        parser.error("Choose --package or --verify")
    subprocess.run([sys.executable, str(ROOT / "src/data/make_splits.py"), "--check"], check=True)
    with (ROOT / "data/manifests/images.csv").open(encoding="utf-8", newline="") as stream:
        images = list(csv.DictReader(stream))
    with (ROOT / "data/manifests/file_checksums.csv").open(encoding="utf-8", newline="") as stream:
        checks = {r["sample_id"]: r for r in csv.DictReader(stream)}
    if args.verify:
        for row in tqdm(images, desc="Verify local files", mininterval=2):
            for kind in ("image", "xml"):
                path = RAW / row[f"{kind}_path"]
                if sha256(path) != checks[row["sample_id"]][f"{kind}_sha256"]:
                    raise ValueError(f"Checksum mismatch: {row[f'{kind}_path']}")
        print(f"OK: {len(images)} images and their XML match dataset_v1.")
        return
    output = ROOT / ".dist/dataset_v1"
    output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for country in sorted({r["country"] for r in images}):
        target = output / f"RDD2022_{country}_dataset_v1.zip"
        # Exclusive creation prevents replacing a previously shared release.
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in tqdm([r for r in images if r["country"] == country], desc=country, mininterval=2):
                for kind in ("image", "xml"):
                    relative = row[f"{kind}_path"]
                    path = RAW / relative
                    if sha256(path) != checks[row["sample_id"]][f"{kind}_sha256"]:
                        raise ValueError(f"Source changed: {relative}; discard incomplete ZIP")
                    archive.write(path, arcname=relative)
        packages[target.name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    (output / "packages.json").write_text(json.dumps(packages, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {output}. Extract each ZIP inside data/raw/RDD2022/.")


if __name__ == "__main__":
    main()
