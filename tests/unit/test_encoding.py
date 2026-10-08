import unittest

import torch

from scope2h3.encoding import H3SightlineCoordinatePE


class EncodingTests(unittest.TestCase):
    def test_zero_initialisation_is_an_exact_noop(self):
        module = H3SightlineCoordinatePE(inner_dim=12, gate_hidden=4)
        query, key, plucker = torch.randn(5, 3, 4), torch.randn(5, 3, 4), torch.randn(5, 6)
        got_query, got_key = module(query, key, plucker)
        torch.testing.assert_close(got_query, query, rtol=0, atol=0)
        torch.testing.assert_close(got_key, key, rtol=0, atol=0)

    def test_gate_bottleneck_expands_to_full_qk_width(self):
        module = H3SightlineCoordinatePE(inner_dim=7168, gate_hidden=64)
        self.assertEqual(tuple(module.scale_gate[0].weight.shape), (64, 1))
        self.assertEqual(tuple(module.scale_gate[2].weight.shape), (7168, 64))
