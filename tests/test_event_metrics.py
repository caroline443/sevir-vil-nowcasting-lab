from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sevir_nowcasting.event_metrics import (
    EventVILStats,
    load_event_stats,
    paired_event_bootstrap,
)


class EventMetricsTest(unittest.TestCase):
    def _write_stats(
        self,
        path: Path,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        stats = EventVILStats(output_length=2, thresholds=(219,))
        stats.update(["event-a", "event-b"], prediction, target)
        stats.save(path)

    def test_accumulates_and_round_trips_event_statistics(self) -> None:
        target = torch.zeros(2, 2, 1, 2, 2)
        prediction = torch.zeros_like(target)
        target[0, :, :, 0, 0] = 1.0
        prediction[0, 0, :, 0, 0] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.npz"
            self._write_stats(path, prediction, target)
            loaded = load_event_stats(path)

        self.assertEqual(loaded["event_ids"].tolist(), ["event-a", "event-b"])
        self.assertEqual(loaded["hits"].shape, (2, 1, 2))
        self.assertEqual(loaded["hits"][0, 0].tolist(), [1.0, 0.0])
        self.assertEqual(loaded["misses"][0, 0].tolist(), [0.0, 1.0])

    def test_paired_bootstrap_detects_consistent_csi_gain(self) -> None:
        target = torch.zeros(2, 2, 1, 2, 2)
        target[:, :, :, 0, 0] = 1.0
        baseline_prediction = torch.zeros_like(target)
        proposed_prediction = target.clone()

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.npz"
            proposed_path = Path(directory) / "proposed.npz"
            self._write_stats(baseline_path, baseline_prediction, target)
            self._write_stats(proposed_path, proposed_prediction, target)
            result = paired_event_bootstrap(
                [baseline_path],
                [proposed_path],
                repetitions=100,
                seed=7,
                chunk_size=20,
            )

        interval = result["difference_intervals"]["csi_219"]
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(
            result["seed_aggregate"]["csi_219"]["difference_mean"], 1.0
        )
        self.assertIsNone(
            result["seed_aggregate"]["csi_219"]["difference_sample_std"]
        )
        self.assertGreater(interval["lower_95"], 0.0)
        self.assertEqual(interval["probability_difference_gt_zero"], 1.0)

    def test_paired_bootstrap_rejects_different_observations(self) -> None:
        baseline_target = torch.zeros(2, 2, 1, 2, 2)
        proposed_target = baseline_target.clone()
        proposed_target[0, 0, :, 0, 0] = 1.0
        prediction = torch.zeros_like(baseline_target)

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.npz"
            proposed_path = Path(directory) / "proposed.npz"
            self._write_stats(baseline_path, prediction, baseline_target)
            self._write_stats(proposed_path, prediction, proposed_target)
            with self.assertRaisesRegex(ValueError, "identical observations"):
                paired_event_bootstrap(
                    [baseline_path],
                    [proposed_path],
                    repetitions=10,
                )


if __name__ == "__main__":
    unittest.main()
