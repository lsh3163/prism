"""Check the paper's scalar polynomial map and its public training interface."""

import io
import unittest

import torch

from prism_robot import PRISM


class PRISMTest(unittest.TestCase):
    def test_quadratic_uses_separate_affines_and_one_unconstrained_scalar(self):
        layer = PRISM(1, 2)
        with torch.no_grad():
            layer.factors[0].weight.copy_(torch.tensor([[1.0], [2.0]]))
            layer.factors[0].bias.copy_(torch.tensor([1.0, -1.0]))
            layer.factors[1].weight.copy_(torch.tensor([[3.0], [-1.0]]))
            layer.factors[1].bias.copy_(torch.tensor([0.0, 2.0]))
            layer.interaction_scales.fill_(-2.0)
        # x=2: u=[3,3], v=[6,0], so u-2*(u*v)=[-33,3].
        torch.testing.assert_close(layer(torch.tensor([[2.0]])), torch.tensor([[-33.0, 3.0]]))
        self.assertEqual(layer.interaction_scales.numel(), 1)

    def test_zero_coefficients_recover_affine_exactly(self):
        layer = PRISM(4, 7, degree=3, scale_init=0)
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(layer(x), layer.factors[0](x), rtol=0, atol=0)

    def test_cubic_retains_lower_order_terms(self):
        layer = PRISM(1, 1, degree=3, scale_init=1)
        with torch.no_grad():
            for affine in layer.factors:
                affine.weight.fill_(1)
                affine.bias.zero_()
        x = torch.tensor([[2.0], [-0.5]])
        torch.testing.assert_close(layer(x), x + 2 * x.square() + x.pow(3))

    def test_degree_one_is_an_affine_layer(self):
        layer = PRISM(4, 7, degree=1)
        self.assertIsNone(layer.interaction_scales)
        x = torch.randn(2, 4)
        torch.testing.assert_close(layer(x), layer.factors[0](x), rtol=0, atol=0)

    def test_training_and_strict_state_round_trip(self):
        layer = PRISM(4, 7, degree=3)
        x = torch.randn(2, 3, 4, requires_grad=True)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        before = layer.interaction_scales.detach().clone()
        output = layer(x)
        self.assertEqual(output.shape, (2, 3, 7))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((layer.interaction_scales.grad.abs() > 0).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, layer.interaction_scales))
        buffer = io.BytesIO()
        torch.save(layer.state_dict(), buffer)
        buffer.seek(0)
        restored = PRISM(4, 7, degree=3)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        torch.testing.assert_close(restored(x), layer(x), rtol=0, atol=0)

    def test_invalid_dimensions_and_degree(self):
        for kwargs in [{"input_dim": 0}, {"output_dim": 0}, {"degree": 0}, {"degree": 1.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PRISM(**{"input_dim": 4, "output_dim": 7, **kwargs})
        with self.assertRaisesRegex(ValueError, "input dimension"):
            PRISM(4, 7)(torch.randn(2, 3))


if __name__ == "__main__":
    unittest.main()
