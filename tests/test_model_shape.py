from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - local documentation environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelShapeTest(unittest.TestCase):
    def test_asymmetric_sequence_shape(self) -> None:
        from sevir_nowcasting.model import SimVP

        model = SimVP(hidden_spatial=8, hidden_temporal=16, temporal_depth=1)
        inputs = torch.randn(1, 13, 1, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (1, 12, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
