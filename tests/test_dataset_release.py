"""Regression checks for reproducibility, freeze protection and data leakage."""
import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_splits", PROJECT / "src/data/make_splits.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class DatasetReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("configs", "data/manifests", "src/data", "reports/tables"):
            (self.root / folder).mkdir(parents=True)
        for source in ("configs/dataset_v1.json", "src/data/make_splits.py", "src/data/audit_release.py"):
            shutil.copyfile(PROJECT / source, self.root / source)
        (self.root / "reports/tables/release_audit.json").write_text("{}", encoding="utf-8")
        self.manifests = self.root / "data/manifests"
        rows, objects, checks = [], [], []
        for country in ("India", "Japan", "Norway", "China_Drone"):
            for status in range(3):
                for index in range(40):
                    sid = f"{country}_{status}_{index:03}"
                    row = dict(sample_id=sid, country=country, image_path=sid + ".jpg",
                               num_objects=int(status > 0), has_any_damage=int(status > 0),
                               is_empty_annotation=int(status == 0))
                    for label in ("D00", "D10", "D20", "D40"):
                        row[f"num_{label}"] = row[f"has_{label}"] = int(
                            (status == 1 and label == "D00") or (status == 2 and label == "D40"))
                    rows.append(row)
                    if status:
                        objects.append(dict(sample_id=sid, **{"class": "D00" if status == 1 else "D40"}))
                    # Two identical images per stratum must never cross splits.
                    checks.append(dict(sample_id=sid, image_sha256=f"{country}_{status}_{max(1,index):03}"))
        pd.DataFrame(rows).to_csv(self.manifests / "images.csv", index=False)
        pd.DataFrame(objects).to_csv(self.manifests / "annotations.csv", index=False)
        pd.DataFrame(checks).to_csv(self.manifests / "file_checksums.csv", index=False)
        self.patcher = patch.multiple(release, ROOT=self.root, MANIFESTS=self.manifests,
                                      CONFIG=self.root / "configs/dataset_v1.json",
                                      LOCK=self.manifests / "dataset_v1.lock.json")
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def run_release(self, *args):
        with patch.object(sys, "argv", ["make_splits.py", *args]), contextlib.redirect_stdout(io.StringIO()):
            release.main()

    def test_reproducible_with_reordered_input_and_no_duplicate_leakage(self):
        self.run_release()
        first = (self.manifests / "splits.csv").read_bytes()
        splits = pd.read_csv(self.manifests / "splits.csv")
        checks = pd.read_csv(self.manifests / "file_checksums.csv")
        self.assertEqual(checks.merge(splits).groupby("image_sha256").split.nunique().max(), 1)
        self.assertEqual(len(splits), 480)
        self.run_release("--check")
        for name in ("splits.csv", "dataset_v1.lock.json"):
            (self.manifests / name).unlink()
        images = pd.read_csv(self.manifests / "images.csv")
        images.sample(frac=1, random_state=99).to_csv(self.manifests / "images.csv", index=False)
        self.run_release()
        self.assertEqual(first, (self.manifests / "splits.csv").read_bytes())

    def test_frozen_release_cannot_be_overwritten_and_detects_tampering(self):
        self.run_release()
        with self.assertRaises(FileExistsError):
            self.run_release()
        with (self.manifests / "splits.csv").open("a") as stream:
            stream.write("unexpected,train\n")
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            self.run_release("--check")

    def test_rejects_cross_domain_duplicate(self):
        checks = pd.read_csv(self.manifests / "file_checksums.csv")
        checks.loc[checks.sample_id == "Norway_0_000", "image_sha256"] = "India_0_001"
        checks.to_csv(self.manifests / "file_checksums.csv", index=False)
        with self.assertRaises(AssertionError):
            self.run_release()

    def test_shared_zip_preserves_manifest_paths_and_checksums(self):
        spec = importlib.util.spec_from_file_location("share_dataset", PROJECT / "src/data/share_dataset.py")
        share = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(share)
        raw = self.root / "data/raw/RDD2022"
        rows, checks = [], []
        for country in ("Japan", "Norway"):
            row = dict(sample_id=country, country=country,
                       image_path=f"{country}/{country}/train/images/example.jpg",
                       xml_path=f"{country}/{country}/train/annotations/xmls/example.xml")
            check = dict(sample_id=country)
            for kind in ("image", "xml"):
                path = raw / row[f"{kind}_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"example {country} {kind}".encode())
                check[f"{kind}_sha256"] = share.sha256(path)
            rows.append(row)
            checks.append(check)
        pd.DataFrame(rows).to_csv(self.manifests / "images.csv", index=False)
        pd.DataFrame(checks).to_csv(self.manifests / "file_checksums.csv", index=False)
        with patch.multiple(share, ROOT=self.root, RAW=raw), patch.object(share.subprocess, "run"), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, "argv", ["share_dataset.py", "--package"]):
                share.main()
            for row in rows:
                target = self.root / f".dist/dataset_v1/RDD2022_{row['country']}_dataset_v1.zip"
                with zipfile.ZipFile(target) as archive:
                    self.assertEqual(set(archive.namelist()), {row["image_path"], row["xml_path"]})
                    for kind in ("image", "xml"):
                        self.assertEqual(archive.read(row[f"{kind}_path"]),
                                         (raw / row[f"{kind}_path"]).read_bytes())
            with patch.object(sys, "argv", ["share_dataset.py", "--verify"]):
                share.main()
            (raw / rows[0]["image_path"]).write_bytes(b"changed")
            with patch.object(sys, "argv", ["share_dataset.py", "--verify"]), \
                    self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                share.main()


if __name__ == "__main__":
    unittest.main()
