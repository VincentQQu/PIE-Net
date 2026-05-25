"""Smoke tests for PIE-Net package (no GPU or camera required)."""

import unittest

import torch

from pie_net import (
    PIENet,
    PIENetLite,
    count_parameters,
    list_variants,
    load_model,
    load_model_lite,
    resolve_variant,
)


class TestVariants(unittest.TestCase):
    def test_resolve_aliases(self):
        self.assertEqual(resolve_variant("pie-net"), "pie-net")
        self.assertEqual(resolve_variant("lite"), "pie-net-lite")
        self.assertEqual(resolve_variant("full"), "pie-net")

    def test_registry_has_both_models(self):
        variants = list_variants()
        self.assertIn("pie-net", variants)
        self.assertIn("pie-net-lite", variants)


class TestModelLoading(unittest.TestCase):
    def test_load_full(self):
        model = load_model(pretrained=True, device="cpu", variant="pie-net")
        self.assertEqual(model.variant, "pie-net")
        self.assertEqual(count_parameters(model), 154_201)

    def test_load_lite(self):
        model = load_model_lite(pretrained=True, device="cpu")
        self.assertIsInstance(model, PIENetLite)
        self.assertEqual(model.variant, "pie-net-lite")
        self.assertEqual(count_parameters(model), 78_537)

    def test_forward_and_reset(self):
        events = torch.randn(1, 5, 64, 64)
        for loader in (load_model, load_model_lite):
            model = loader(pretrained=True, device="cpu")
            model.eval()
            with torch.no_grad():
                out = model(events)
            self.assertEqual(out["image"].shape, (1, 1, 64, 64))
            self.assertEqual(out["var"].shape, (1, 1, 64, 64))
            model.reset_states()
            self.assertIsNone(model.f0)


if __name__ == "__main__":
    unittest.main()
