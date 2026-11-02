"""Public evaluation contracts; synthetic inputs here are never project results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.evaluation.cross_domain import validate_predictions, metric_row, domain_deltas, select_cases, load_experiment
from src.evaluation.inference import load_guatemala, load_checkpoint, score_batch
from src.evaluation.latency import benchmark_latency, measure_latency
from src.evaluation.metrics import prediction_frame
from src.models.resnet import build_resnet18
from src.xai.gradcam import gradcam, bbox_alignment


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(sample_id=list("abcde"), split="test_id", has_D40=[0, 1, 0, 1, 0],
                                       has_any_damage=[0, 1, 1, 1, 0]))
        self.pred = prediction_frame(list("abcde"), "test_id", [0, 1, 0, 1, 0], [.1, .5, .8, .2, .3],
                                     .5, "model", "dataset_v1")

    def validate(self, pred):
        return validate_predictions(pred, self.frame, "model", "dataset_v1", .5)

    def test_metrics_threshold_boundary_and_strict_manifest(self):
        table = self.validate(self.pred)
        self.assertEqual(table.outcome.tolist(), ["TN", "TP", "FP", "FN", "TN"])
        metrics = metric_row(table)
        self.assertEqual([metrics[k] for k in ("tn", "fp", "fn", "tp")], [2, 1, 1, 1])
        self.assertEqual(metrics["f1"], .5)
        self.assertAlmostEqual(metrics["fpr"], 1/3)
        self.assertEqual(metrics["flagged_rate"], .4)
        self.assertEqual(table.group.tolist(), ["no_damage", "pothole", "other", "pothole", "no_damage"])
        bad = [self.pred.iloc[:-1], pd.concat([self.pred, self.pred.iloc[:1]]),
               self.pred.assign(split="test_ood"), self.pred.assign(y_true=1),
               self.pred.assign(threshold=.3), self.pred.assign(dataset_version="dataset_v2"),
               self.pred.assign(y_score=np.inf)]
        for pred in bad:
            with self.subTest(pred=pred.to_dict()), self.assertRaises(ValueError):
                self.validate(pred)
        changed = self.pred.copy(); changed.loc[0, "threshold"] = .6
        with self.assertRaises(ValueError): self.validate(changed)

    def test_case_selection_stable_and_deltas_use_test_not_val(self):
        table = self.validate(self.pred)
        a, b = select_cases(table, 1), select_cases(table.sample(frac=1, random_state=7), 1)
        self.assertEqual(a.sample_id.tolist(), b.sample_id.tolist())
        self.assertEqual(set(a.outcome), {"TP", "TN", "FP", "FN"})
        metrics = pd.DataFrame(dict(family="resnet", experiment_id="model",
            split=["val", "test_id", "test_ood"], f1=[.99, .6, .2], recall=[.99, .8, .9], prevalence=[.1, .2, .3]))
        delta = domain_deltas(metrics).iloc[0]
        self.assertAlmostEqual(delta.delta_f1, .4)
        self.assertAlmostEqual(delta.delta_recall, -.1)

    def test_test_optimal_threshold_cannot_replace_validation_threshold(self):
        # Validation optimum .8, external optimum .3: changing the latter must not affect acceptance.
        frame = pd.DataFrame(dict(sample_id=list("abcd"), split=["val", "val", "test_ood", "test_ood"],
                                  has_D40=[0, 1, 0, 1], has_any_damage=[0, 1, 0, 1]))
        pred = prediction_frame(frame.sample_id, frame.split, frame.has_D40, [.2, .8, .1, .3], .8, "model", "v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resnet/model"; path.mkdir(parents=True)
            (path/"config.yaml").write_text(yaml.safe_dump(dict(experiment_id="model", dataset_version="v1")))
            (path/"metrics.json").write_text(json.dumps({}))
            pred.to_csv(path/"predictions.csv", index=False)
            table, _, _ = load_experiment(path, frame, {"dataset_version": "v1"})
            self.assertEqual(table.loc[table.split == "test_ood", "outcome"].tolist(), ["TN", "FN"])
            pred.assign(threshold=.3).to_csv(path/"predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "max-val-F1"):
                load_experiment(path, frame, {"dataset_version": "v1"})

    def test_external_protocol_requires_review_and_distinct_ids(self):
        external = pd.DataFrame(dict(sample_id=["GT_01"], image_path=["images/a.jpg"], has_D40=[1],
            has_any_damage=[1], source=["own capture"], capture_session=["session1"], road_segment=["road1"],
            camera=["phone"], lighting=["day"], weather=["dry"], labeler=["A"], reviewer=["B"],
            review_status=["approved"], notes=[""]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.csv"
            external.to_csv(path, index=False)
            actual = load_guatemala(path, self.frame)
            self.assertEqual(actual.split.tolist(), ["test_guatemala"])
            for column, value in [("sample_id", "a"), ("review_status", "pending"), ("has_any_damage", 0),
                                  ("reviewer", ""), ("image_path", "../photo.jpg")]:
                external.assign(**{column: value}).to_csv(path, index=False)
                with self.subTest(column=column), self.assertRaises(ValueError):
                    load_guatemala(path, self.frame)


class ExplanationTests(unittest.TestCase):
    def test_gradcam_explains_d40_logit_not_auxiliary_or_probability(self):
        class KnownCAM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer4 = nn.Identity()
                self.weight = nn.Parameter(torch.tensor([[2., -1.], [-20., 10.]]), requires_grad=False)

            def forward(self, x):
                return self.layer4(x).mean((2, 3)) @ self.weight.T

        image = torch.tensor([[[4., 0., 1.], [2., 3., 0.]], [[0., 4., 1.], [1., 0., 2.]]])
        model = KnownCAM().eval()
        result = gradcam(model, image)
        # Independent analytic weights: d mean / d pixel = 1/6, target weights [2, -1].
        expected = np.array([[8., -4., 1.], [3., 6., -2.]])/6
        np.testing.assert_allclose(result["raw"], expected, atol=1e-7)
        np.testing.assert_allclose(result["heatmap"], np.maximum(expected, 0)/(8/6), atol=1e-7)
        self.assertAlmostEqual(result["logit"], 2, places=6)
        self.assertIsNone(model.weight.grad)
        zero = gradcam(model, torch.zeros_like(image))
        self.assertTrue(zero["constant"])
        self.assertTrue(np.isfinite(zero["heatmap"]).all())

    def test_actual_resnet_frozen_checkpoint_has_input_size_map(self):
        torch.set_num_threads(2)
        config = dict(experiment_id="synthetic", image_size=32, aux_classes=["D00", "D10", "D20"], best_epoch=2)
        model = build_resnet18(outputs=4).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"best.pt"
            torch.save(dict(model=model.state_dict(), epoch=2, config=config), path)
            restored = load_checkpoint(path, config, "resnet")
            for p in restored.parameters(): p.requires_grad_(False)
            image = torch.rand(3, 32, 32)
            result = gradcam(restored, image)
            self.assertEqual(result["heatmap"].shape, (32, 32))
            self.assertAlmostEqual(result["probability"], score_batch(restored, image[None], "resnet", config, {})[0].item())
            with self.assertRaises(ValueError): load_checkpoint(path, {**config, "best_epoch": 1}, "resnet")

    def test_box_union_on_nonsquare_original_and_zero_maps(self):
        heat = np.array([[0., 1., 2., 0.], [0., 3., 4., 0.]])
        boxes = [[2, 0, 6, 4], [4, 0, 6, 4]]  # original 8x4 -> map 4x2; overlap counted once
        result = bbox_alignment(heat, boxes, 8, 4)
        self.assertEqual(result["box_area_fraction"], .5)
        self.assertEqual(result["mass_in_boxes"], 1)
        self.assertEqual(result["mass_over_area"], 2)
        self.assertTrue(result["peak_in_box"])
        self.assertIsNone(bbox_alignment(heat*0, boxes, 8, 4)["mass_in_boxes"])
        self.assertIsNone(bbox_alignment(heat, [], 8, 4)["peak_in_box"])

    def test_vae_scores_use_patch_not_global_and_fixed_train_statistics(self):
        class ZeroRecon(nn.Module):
            def forward(self, x): return torch.zeros_like(x), None, None
        x = torch.tensor([[[[1., 1., 0., 0.], [1., 1., 0., 0.]]]]).repeat(1, 3, 1, 1)
        model = ZeroRecon().eval()
        scores = {}
        for primary in ("global", "patch", "patch_norm"):
            meta = dict(primary_score=primary, position_stats=dict(mean=[.5, 0], std_eff=[.25, 1]))
            scores[primary] = score_batch(model, x, "vae", {"patch": 2}, meta).item()
        self.assertEqual(scores, {"global": .5, "patch": 1., "patch_norm": 2.})

    def test_latency_short_dataset_still_records_requested_measurements(self):
        dataset = [(torch.ones(3, 4, 4),)]
        result = benchmark_latency(nn.Identity(), dataset, "cpu", count=5, warmup=8)
        self.assertEqual(len(result["samples_ms"]), 5)
        self.assertEqual(result["batch_size"], 1)
        self.assertEqual(result["input_shape"], [1, 3, 4, 4])
        self.assertTrue(all(ms > 0 for ms in result["samples_ms"]))
        self.assertGreaterEqual(result["p95_ms"], result["median_ms"])
        self.assertGreater(measure_latency(nn.Identity(), dataset, torch.device("cpu"), 2), 0)
        self.assertIsNone(measure_latency(nn.Identity(), dataset, torch.device("cpu"), 0))
        with self.assertRaises(ValueError): benchmark_latency(nn.Identity(), dataset, "cpu", count=0)


if __name__ == "__main__":
    unittest.main()
