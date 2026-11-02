"""Checks for the resized PNG cache (equivalence with the raw JPG path, safety, idempotence)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.cache_images import build_cache, cache_path, cached_frame  # noqa: E402
from src.data.dataset import PotholeDataset, build_transforms  # noqa: E402

SIZE = 16


class CacheImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw, self.cache = root / "RDD2022", root / f"RDD2022_{SIZE}"
        rng = np.random.default_rng(0)
        rows = []
        # tamaños distintos, incluido uno no cuadrado como norway
        for index, shape in enumerate([(48, 64, 3), (40, 40, 3), (30, 90, 3)]):
            path = Path("Country/Country/train/images") / f"img_{index}.jpg"
            (self.raw / path.parent).mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, shape, dtype=np.uint8)).save(self.raw / path)
            rows.append(dict(sample_id=f"s{index}", image_path=path.as_posix(), has_D40=index % 2))
        self.frame = pd.DataFrame(rows)

    def test_cached_tensor_equals_raw_tensor(self):
        build_cache(self.frame, self.raw, self.cache, SIZE, workers=2)
        transform = build_transforms(size=SIZE, normalize=False)
        raw = PotholeDataset(self.frame, transform, self.raw)
        cached = PotholeDataset(cached_frame(self.frame), transform, self.cache)
        for index in range(len(self.frame)):
            self.assertTrue(torch.equal(raw[index][0], cached[index][0]))
            self.assertEqual(raw[index][2], cached[index][2])

    def test_rejects_unsafe_paths(self):
        for bad in ("../x.jpg", "a/../../x.jpg", "/abs/x.jpg", "C:/x.jpg", "C:x.jpg", "\\\\server\\x.jpg"):
            with self.assertRaises(ValueError):
                cache_path(bad)
        with self.assertRaises(ValueError):
            build_cache(pd.DataFrame(dict(image_path=["../escape.jpg"])), self.raw, self.cache, SIZE)

    def test_many_workers_on_fresh_folders(self):
        # varios hilos creando las mismas carpetas a la vez, como en la corrida real
        rng = np.random.default_rng(1)
        rows = []
        for index in range(64):
            path = Path(f"Many/Many/train/images/d{index % 4}") / f"m_{index}.jpg"
            (self.raw / path.parent).mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)).save(self.raw / path)
            rows.append(dict(sample_id=f"m{index}", image_path=path.as_posix(), has_D40=0))
        self.assertEqual(build_cache(pd.DataFrame(rows), self.raw, self.cache, SIZE, workers=16), 64)

    def test_second_build_rewrites_nothing(self):
        self.assertEqual(build_cache(self.frame, self.raw, self.cache, SIZE), len(self.frame))
        files = sorted(self.cache.rglob("*.png"))
        stamps = [path.stat().st_mtime_ns for path in files]
        self.assertEqual(build_cache(self.frame, self.raw, self.cache, SIZE), 0)
        self.assertEqual([path.stat().st_mtime_ns for path in files], stamps)

    def test_orphan_temp_file_does_not_block_resume(self):
        target = self.cache / cache_path(self.frame.image_path[0])
        target.parent.mkdir(parents=True)
        target.with_name(target.name + ".tmp").write_bytes(b"partial")
        self.assertEqual(build_cache(self.frame, self.raw, self.cache, SIZE), len(self.frame))
        with Image.open(target) as image:
            self.assertEqual(image.size, (SIZE, SIZE))
        self.assertEqual(list(self.cache.rglob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
