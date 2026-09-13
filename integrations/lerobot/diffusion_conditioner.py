"""Historical LIBERO Diffusion conditioner with its original checkpoint layout.

This adapter implements the representation used by the main Diffusion runs.
The standalone ``prism_robot.PRISMConditioner`` is a different actor variant.
"""

from torch import Tensor, nn


class PolynomialKernelConditioner(nn.Module):
    """Transform normalized state history through two learned affine factors.

    ``latent_quadratic`` refers to an elementwise product of affine factors;
    it does not enumerate all degree-two monomials. Both LayerNorm operations,
    the SiLU MLP, parameter names, and initialization match the experiment.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lift_mode: str,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if lift_mode not in {"raw", "latent_quadratic"}:
            raise ValueError(f"Unsupported historical polynomial kernel lift mode: {lift_mode}")
        self.lift_mode = lift_mode
        self.input_norm = nn.LayerNorm(input_dim)
        factor_dim = latent_dim if lift_mode == "latent_quadratic" else input_dim
        self.left_proj = nn.Linear(input_dim, factor_dim)
        self.right_proj = nn.Linear(input_dim, factor_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, global_cond: Tensor) -> Tensor:
        features = self.input_norm(global_cond)
        factors = self.left_proj(features) * self.right_proj(features)
        return self.net(factors)
