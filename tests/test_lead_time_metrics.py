from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sevir_nowcasting.metrics import LeadTimeVILMetrics


class LeadTimeMetricsTest(unittest.TestCase):
    def test_preserves_lead_axis_and_threshold_counts(self) -> None:
        target = torch.zeros(1, 2, 1, 2, 2)
        prediction = torch.zeros_like(target)
        target[:, 0, :, 0, 0] = 1.0
        prediction[:, 0, :, 0, 0] = 1.0
        target[:, 1, :, 0, 0] = 1.0

        metrics = LeadTimeVILMetrics(output_length=2, thresholds=(219,))
        metrics.update(prediction, target)
        result = metrics.compute()

        self.assertEqual(result["lead_minutes"], [5, 10])
        self.assertEqual(result["csi_by_threshold"]["219"], [1.0, 0.0])
        self.assertEqual(result["pod_by_threshold"]["219"], [1.0, 0.0])
        self.assertEqual(result["observed_pixels_by_threshold"]["219"], [1.0, 1.0])
        self.assertEqual(result["mse_by_lead"], [0.0, 0.25])


if __name__ == "__main__":
    unittest.main()
