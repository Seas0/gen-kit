"""Gaussian probability paths and prediction parameterizations.

Time always runs from data (0) to noise (1); generation runs backwards.
Diffusion ``v`` = alpha * epsilon - sigma * x0 is distinct from the path
velocity dx/dt = alpha' * x0 + sigma' * epsilon.
"""

import math

import torch


PREDICTION_TYPES = ("epsilon", "x0", "v", "score", "velocity")
TRAJECTORIES = ("vp", "linear", "cosine", "vp_sde", "subvp", "ve", "edm")


def prediction_type(name: str) -> str:
    aliases = {"eps": "epsilon", "ε": "epsilon", "sample": "x0", "x": "x0", "v_prediction": "v", "flow": "velocity"}
    name = aliases.get(name, name)
    if name not in PREDICTION_TYPES:
        raise ValueError(f"Unknown prediction_type {name!r}; choose from {PREDICTION_TYPES}")
    return name


def expand(value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast one coefficient per sample to vectors or images."""
    return value.reshape(-1, *([1] * (x.ndim - 1)))


def training_target(kind, x0, noise, alpha, sigma, d_alpha=None, d_sigma=None):
    if kind == "epsilon":
        return noise
    if kind == "x0":
        return x0
    if kind == "v":
        return alpha * noise - sigma * x0
    if kind == "score":
        return -noise / sigma
    if kind == "velocity" and d_alpha is not None and d_sigma is not None:
        return d_alpha * x0 + d_sigma * noise
    raise ValueError(f"Cannot construct target {kind!r} without path derivatives")


def prediction_to_x0_eps(kind, output, x, alpha, sigma, d_alpha=None, d_sigma=None):
    """Invert a parameterization. Call at nonsingular times for eps/x0/score."""
    if kind == "epsilon" or kind == "score":
        eps = output if kind == "epsilon" else -sigma * output
        return (x - sigma * eps) / alpha, eps
    if kind == "x0":
        return output, (x - alpha * output) / sigma
    if kind == "v":
        norm = alpha.square() + sigma.square()
        return (alpha * x - sigma * output) / norm, (sigma * x + alpha * output) / norm
    if kind == "velocity" and d_alpha is not None and d_sigma is not None:
        det = alpha * d_sigma - sigma * d_alpha
        return (d_sigma * x - sigma * output) / det, (alpha * output - d_alpha * x) / det
    raise ValueError(f"Cannot convert prediction {kind!r} without path derivatives")


def path_coefficients(
    trajectory: str,
    t: torch.Tensor,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return alpha, sigma, d(alpha)/dt and d(sigma)/dt on a continuous path.

    The discrete VP path is handled by the model's trained beta schedule.
    VE starts at sigma_min; sampling ends with a final denoising projection.
    """
    if trajectory == "linear":
        return 1 - t, t, -torch.ones_like(t), torch.ones_like(t)
    if trajectory == "cosine":
        angle = t * (math.pi / 2)
        a, s = angle.cos(), angle.sin()
        return a, s, -(math.pi / 2) * s, (math.pi / 2) * a
    if trajectory in ("vp_sde", "subvp"):
        integral = beta_min * t + 0.5 * (beta_max - beta_min) * t.square()
        beta = beta_min + (beta_max - beta_min) * t
        a = (-0.5 * integral).exp()
        da = -0.5 * beta * a
        variance = -torch.expm1(-integral)
        if trajectory == "subvp":
            return a, variance, da, beta * (-integral).exp()
        s = variance.sqrt()
        return a, s, da, beta * a.square() / (2 * s.clamp_min(1e-12))
    if trajectory in ("ve", "edm"):
        rate = math.log(sigma_max / sigma_min)
        s = sigma_min * (rate * t).exp()
        return torch.ones_like(t), s, torch.zeros_like(t), rate * s
    raise ValueError(f"No continuous coefficients for trajectory {trajectory!r}")


def sigma_schedule(steps, sigma_min, sigma_max, rho=7.0, spacing="karras", *, device=None, dtype=None):
    """Descending positive noise levels, excluding a terminal zero."""
    if steps < 1:
        raise ValueError("steps must be positive")
    ramp = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    if spacing == "karras":
        result = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    elif spacing == "exponential":
        result = (math.log(sigma_max) + ramp * math.log(sigma_min / sigma_max)).exp()
    elif spacing == "linear":
        result = sigma_max + ramp * (sigma_min - sigma_max)
    else:
        raise ValueError("sigma spacing must be karras, exponential, or linear")
    result[0] = sigma_max
    if steps > 1:
        result[-1] = sigma_min
    return result
