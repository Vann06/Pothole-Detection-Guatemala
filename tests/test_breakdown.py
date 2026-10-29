"""Checks for the group / country / other-damage-class breakdown."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.evaluation.breakdown import damage_group, group_report, other_damage_long  # noqa: E402

COLUMNS = ("sample_id", "country", "has_D00", "has_D10", "has_D20", "has_D40", "has_any_damage", "other_labels")


def frame_of(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


class BreakdownTests(unittest.TestCase):
    def setUp(self):
        # A tiene los tres grupos, B no tiene normales (como US/MotorBike)
        self.frame = frame_of([
            ("n1", "A", 0, 0, 0, 0, 0, np.nan),
            ("n2", "A", 0, 0, 0, 0, 0, np.nan),
            ("o1", "A", 1, 0, 0, 0, 1, "Repair;D43"),
            ("o2", "A", 0, 1, 0, 0, 1, " D44 "),
            ("p1", "A", 1, 0, 0, 1, 1, "D50"),
            ("o3", "B", 1, 0, 1, 0, 1, np.nan),
            ("p2", "B", 0, 0, 0, 1, 1, np.nan),
        ])
        self.scores = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.7, 0.8])

    def test_groups(self):
        self.assertEqual(damage_group(self.frame).tolist(),
                         ["no_damage", "no_damage", "other", "other", "pothole", "other", "pothole"])

    def test_other_labels_parsing_skips_d40_images(self):
        long = other_damage_long(self.frame)
        pairs = set(map(tuple, long[["sample_id", "damage_class"]].to_numpy()))
        self.assertEqual(pairs, {("o1", "D00"), ("o1", "Repair"), ("o1", "D43"), ("o2", "D10"), ("o2", "D44"),
                                 ("o3", "D00"), ("o3", "D20")})

    def test_class_false_positive_rate_matches_manual_count(self):
        classes = group_report(self.frame, self.scores, threshold=0.5)["other_damage_classes"]
        # d00 aparece en o1 (0.6) y o3 (0.7): las dos pasan el threshold
        self.assertEqual(classes["D00"]["n"], 2)
        self.assertEqual(classes["D00"]["flagged_rate"], 1.0)
        self.assertEqual(classes["D10"]["flagged_rate"], 0.0)
        self.assertNotIn("D50", classes)

    def test_groups_and_pairwise(self):
        report = group_report(self.frame, self.scores, threshold=0.5)
        self.assertEqual(report["groups"]["no_damage"]["flagged_rate"], 0.0)
        self.assertEqual(report["groups"]["pothole"]["n"], 2)
        self.assertEqual(report["pairwise_auroc"]["no_damage_vs_pothole"]["auroc"], 1.0)
        self.assertIn("note", report)

    def test_country_without_normals_returns_none(self):
        country = group_report(self.frame, self.scores, threshold=0.5)["by_country"]["B"]
        self.assertEqual(country["n"]["no_damage"], 0)
        self.assertIsNone(country["pairwise_auroc"]["no_damage_vs_pothole"]["auroc"])
        self.assertEqual(country["pairwise_auroc"]["other_vs_pothole"]["auroc"], 1.0)

    def test_split_without_positives_does_not_fail(self):
        negatives = self.frame[self.frame.has_D40 == 0]
        report = group_report(negatives, self.scores[:4].tolist() + [0.7], threshold=0.5)
        self.assertIsNone(report["by_country"]["A"]["pr_auc_D40"])
        self.assertEqual(report["groups"]["pothole"]["n"], 0)
        self.assertIsNone(report["groups"]["pothole"]["mean"])


if __name__ == "__main__":
    unittest.main()
