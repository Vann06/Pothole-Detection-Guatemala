"""Folder-label contract and outcome counting for the external-photo check; synthetic inputs only."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.external_images import collect_images, outcome_counts


class ExternalImagesTests(unittest.TestCase):
    def test_labels_come_only_from_known_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["con_bache/a.jpg", "con_bache/notes.txt", "sin_bache/b.PNG"]:
                (root / name).parent.mkdir(exist_ok=True)
                (root / name).write_bytes(b"")
            frame = collect_images(root)
            self.assertEqual(frame.to_dict("records"), [dict(image="con_bache/a.jpg", label=1),
                                                        dict(image="sin_bache/b.PNG", label=0)])
            (root / "maybe").mkdir()
            with self.assertRaises(ValueError):
                collect_images(root)

    def test_empty_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "con_bache").mkdir()
            with self.assertRaises(ValueError):
                collect_images(directory)

    def test_outcome_counts_per_experiment(self):
        predictions = pd.DataFrame(dict(experiment_id=["m"] * 4 + ["n"], label=[1, 1, 0, 0, 1],
                                        predicted=[True, False, True, False, True]))
        counts = outcome_counts(predictions)
        self.assertEqual(counts.loc["m"].tolist(), [1, 1, 1, 1])
        self.assertEqual(counts.loc["n"].tolist(), [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
