import io
import unittest

import torch

from prism_robot import PRISMConditioner, RMSNorm


class PRISMConditionerTest(unittest.TestCase):
    def test_default_gates_train_and_resume_with_optimizer_state(self) -> None:
        torch.manual_seed(17)
        model = PRISMConditioner(4, 3, hidden_dim=5, degree=3)
        scales = model.interaction_scales
        self.assertIsInstance(scales, torch.nn.Parameter)
        torch.testing.assert_close(scales, torch.full((2, 5), 0.01))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs, targets = torch.randn(8, 4), torch.randn(8, 3)

        def update(network, optim):
            optim.zero_grad()
            torch.nn.functional.mse_loss(network(inputs), targets).backward()
            gradient = network.interaction_scales.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient.abs().sum(dim=1) > 0).all())
            optim.step()

        initial = scales.detach().clone()
        update(model, optimizer)
        self.assertFalse(torch.equal(initial, scales))
        checkpoint = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = PRISMConditioner(4, 3, hidden_dim=5, degree=3)
        restored.load_state_dict(saved["model"], strict=True)
        resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
        update(model, optimizer)
        update(restored, resumed_optimizer)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_gates_are_unconstrained_feature_scales(self) -> None:
        model = PRISMConditioner(1, 2, hidden_dim=2, post_mlp_layers=1)
        with torch.no_grad():
            for factor in model.factors:
                factor.weight.fill_(1)
                factor.bias.zero_()
            model.interaction_scales.copy_(torch.tensor([[-2.0, 3.0]]))
        # At x=2, u=v=2 and u*(1+alpha*v) yields [-6, 14].
        torch.testing.assert_close(
            model.polynomial_features(torch.tensor([[2.0]])), torch.tensor([[-6.0, 14.0]])
        )

    def test_output_shape_and_end_to_end_gradients(self) -> None:
        model = PRISMConditioner(
            input_dim=7,
            output_dim=11,
            hidden_dim=13,
            degree=2,
            use_rmsnorm=True,
        )
        inputs = torch.randn(5, 3, 7, requires_grad=True)

        outputs = model(inputs)
        outputs.square().mean().backward()

        self.assertEqual(outputs.shape, (5, 3, 11))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_zero_gates_recover_first_order_features(self) -> None:
        model = PRISMConditioner(
            input_dim=4,
            output_dim=6,
            hidden_dim=6,
            degree=3,
            gate_init=0.0,
        )
        inputs = torch.randn(8, 4)

        expected = model.factors[0](inputs)
        actual = model.polynomial_features(inputs)

        torch.testing.assert_close(actual, expected)

    def test_degree_three_contains_first_to_third_order_terms(self) -> None:
        model = PRISMConditioner(
            input_dim=1,
            output_dim=1,
            hidden_dim=1,
            degree=3,
            gate_init=1.0,
        )
        with torch.no_grad():
            for factor in model.factors:
                factor.weight.fill_(1.0)
                factor.bias.zero_()

        inputs = torch.tensor([[2.0], [-0.5]])
        expected = inputs * (1.0 + inputs).square()

        torch.testing.assert_close(model.polynomial_features(inputs), expected)

    def test_factorized_mode_multiplies_affine_factors(self) -> None:
        model = PRISMConditioner(
            input_dim=1,
            output_dim=1,
            hidden_dim=1,
            degree=2,
            interaction_mode="factorized",
        )
        with torch.no_grad():
            for factor in model.factors:
                factor.weight.fill_(1.0)
                factor.bias.zero_()

        inputs = torch.tensor([[3.0]])
        torch.testing.assert_close(
            model.polynomial_features(inputs),
            torch.tensor([[9.0]]),
        )

    def test_rmsnorm_preserves_shape_and_dtype(self) -> None:
        norm = RMSNorm(6)
        inputs = torch.randn(4, 6, dtype=torch.float16)
        outputs = norm(inputs)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertEqual(outputs.dtype, inputs.dtype)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PRISMConditioner(4, 8, degree=0)
        with self.assertRaises(ValueError):
            PRISMConditioner(4, 8, interaction_mode="unknown")
        with self.assertRaisesRegex(ValueError, "hidden_dim must be positive"):
            PRISMConditioner(4, 8, hidden_dim=0)


if __name__ == "__main__":
    unittest.main()
