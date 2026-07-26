import unittest

import torch

from prism_robot import PRISMConditioner, RMSNorm


class PRISMConditionerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
