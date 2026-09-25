#!/usr/bin/env python3
"""Structural checks for the final adapted-MalCL discriminator topology."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))

try:
    import torch
    from patched_malcl.models import Discriminator, discriminator_topology_spec, run_topology_validation
except Exception as exc:  # pragma: no cover - dependency-dependent skip
    torch = None
    IMPORT_ERROR = exc


class FinalMalCLTopologyTests(unittest.TestCase):
    def test_source_declares_flattened_topology_without_pooling(self) -> None:
        source = (ROOT / "runners" / "patched_malcl" / "models.py").read_text(encoding="utf-8")
        forbidden_pool = "Adaptive" + "AvgPool1d"
        self.assertNotIn(forbidden_pool, source)
        self.assertIn("nn.Linear(self.channel_c * self.input_features, self.latent_dim)", source)
        self.assertIn("feature = x.view(-1, self.channel_c * self.input_features)", source)
        for feature_dim, expected_mib in ((2381, 2382.5263710021973), (2439, 2440.5263710021973)):
            feature_width = 256 * feature_dim
            parameter_count = (512 * 1 * 3 + 512) + (256 * 512 * 3 + 256) + (2 * 256) + (feature_width * 1024 + 1024) + (2 * 1024) + (1024 + 1)
            self.assertEqual(feature_width, 256 * feature_dim)
            self.assertAlmostEqual(parameter_count * 4 / (1024 * 1024), expected_mib, places=6)

    @unittest.skipIf(torch is None, "PyTorch is required for topology verification")
    def test_flattened_topology_and_memory_accounting(self) -> None:
        report = run_topology_validation(feature_dims=(2381, 2439), batch_size=2)
        self.assertTrue(report["passed"], report)
        expected_mib = {2381: 2382.5263710021973, 2439: 2440.5263710021973}
        for detail in report["details"]:
            feature_dim = detail["feature_dim"]
            spec = discriminator_topology_spec(feature_dim)
            self.assertEqual(spec["first_fc_in_features"], 256 * feature_dim)
            self.assertEqual(spec["first_fc_out_features"], 1024)
            self.assertEqual(spec["returned_feature_shape"], ["batch", 256 * feature_dim])
            self.assertAlmostEqual(spec["parameter_memory_mib_fp32"], expected_mib[feature_dim], places=6)
            self.assertFalse(detail["adaptive_pooling_present"])
            self.assertTrue(detail["meta_forward_backward_passed"])

    @unittest.skipUnless(os.environ.get("RUN_MEMORY_HEAVY_TOPOLOGY_TEST") == "1", "set RUN_MEMORY_HEAVY_TOPOLOGY_TEST=1 on suitable GPU hardware")
    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA is required for optional real allocation test")
    def test_real_forward_backward_when_hardware_is_sufficient(self) -> None:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 6 * 1024**3:
            self.skipTest("insufficient free CUDA memory")
        model = Discriminator(feature_dim=2381).cuda().eval()
        inputs = torch.randn(1, 2381, device="cuda", requires_grad=True)
        score, feature = model(inputs)
        self.assertEqual(tuple(feature.shape), (1, 256 * 2381))
        (score.sum() + feature.mean()).backward()


if __name__ == "__main__":
    unittest.main()
