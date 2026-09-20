"""Sampling algorithms, independent of the network's training target."""

from contextlib import contextmanager
import inspect

import torch

from .paths import prediction_to_x0_eps, sigma_schedule


# These schedulers share the exact trained VP beta schedule. No pretrained
# networks or remote assets are loaded by this adapter.
VP_SCHEDULERS = {
    "ddpm": ("DDPMScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_ancestral": ("EulerAncestralDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "pndm": ("PNDMScheduler", {"skip_prk_steps": False, "timestep_spacing": "leading", "set_alpha_to_one": True}),
    "plms": ("PNDMScheduler", {"skip_prk_steps": True, "timestep_spacing": "leading", "set_alpha_to_one": True}),
    "dpm_solver": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver", "final_sigmas_type": "sigma_min"}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    "dpmpp_3m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 3}),
    "dpmpp_sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}),
    "dpmpp_2s": ("DPMSolverSinglestepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    "deis": ("DEISMultistepScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
}
ODE_SCHEDULERS = ("euler", "midpoint", "heun", "rk4")
SDE_SCHEDULERS = ("euler_maruyama",)
SIGMA_SCHEDULERS = ("euler", "heun", "euler_ancestral", "dpmpp_2m")


@contextmanager
def evaluating(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def randn_like(x, generator=None):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def make_vp_scheduler(name, betas, kwargs=None):
    import diffusers

    if name not in VP_SCHEDULERS:
        raise ValueError(f"Unknown VP scheduler {name!r}; choose from {tuple(VP_SCHEDULERS)}")
    class_name, defaults = VP_SCHEDULERS[name]
    cls = getattr(diffusers, class_name)
    parameters = inspect.signature(cls.__init__).parameters
    kwargs = dict(kwargs or {})
    protected = {
        "trained_betas",
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
        "prediction_type",
        "use_flow_sigmas",
        "rescale_betas_zero_snr",
    }
    invalid = set(kwargs) - set(parameters) | set(kwargs) & protected
    if invalid:
        raise ValueError(f"Invalid or protected options for {name}: {sorted(invalid)}")
    options = dict(
        num_train_timesteps=len(betas), trained_betas=betas.detach().cpu().numpy(), prediction_type="epsilon"
    )
    if "clip_sample" in parameters:
        options["clip_sample"] = False
    if "timestep_spacing" in parameters:
        options["timestep_spacing"] = "linspace"
    options.update(defaults)
    options.update(kwargs)
    return cls(**options)


def ddim_step(scheduler, eps, x, t, next_t, eta, generator):
    """Use the actual next grid point, including nonuniform/rounded spacing.

    Diffusers DDIM assumes a constant integer stride in step(), even when
    set_timesteps() chooses a different linspace or trailing grid.
    """
    alpha = scheduler.alphas_cumprod[t.long()].to(x)
    next_alpha = (scheduler.final_alpha_cumprod if next_t is None else scheduler.alphas_cumprod[next_t.long()]).to(x)
    clean = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
    if scheduler.config.thresholding:
        clean = scheduler._threshold_sample(clean)
    elif scheduler.config.clip_sample:
        clean = clean.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    variance = ((1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha)).clamp_min(0)
    std = eta * variance.sqrt()
    result = next_alpha.sqrt() * clean + (1 - next_alpha - std.square()).clamp_min(0).sqrt() * eps
    return result + std * randn_like(x, generator) if eta else result


@torch.no_grad()
def sample_vp(model, shape, count, cids, name, steps, generator=None, trajectory=False, kwargs=None):
    if not 1 <= steps <= model.num_steps:
        raise ValueError(f"VP inference steps must be in [1, {model.num_steps}]")
    if name == "pndm" and steps < 4 or name == "plms" and steps < 2:
        raise ValueError("PNDM needs at least 4 steps; PLMS needs at least 2")
    kwargs = dict(kwargs or {})
    eta = kwargs.pop("eta", 0.0)
    if eta and name != "ddim":
        raise ValueError("eta is only supported by DDIM")
    if not 0 <= eta <= 1:
        raise ValueError("DDIM eta must be in [0, 1]")
    # A one-element linspace/leading grid otherwise starts at timestep zero.
    if steps == 1 and name in ("ddpm", "ddim", "euler", "euler_ancestral", "heun", "lms"):
        kwargs.setdefault("timestep_spacing", "trailing")
    scheduler = make_vp_scheduler(name, model.betas, kwargs)
    scheduler.set_timesteps(steps, device=model.device)
    x = torch.randn(count, *shape, device=model.device, dtype=model.dtype, generator=generator)
    x = x * torch.as_tensor(scheduler.init_noise_sigma, device=x.device, dtype=x.dtype)
    history = [x.clone()] if trajectory else None
    if cids is not None:
        cids = torch.as_tensor(cids, device=x.device).reshape(-1)
    step_options = {}
    if "generator" in inspect.signature(scheduler.step).parameters:
        step_options["generator"] = generator
    with evaluating(model):
        for index, t in enumerate(scheduler.timesteps):
            network_input = scheduler.scale_model_input(x, t)
            if hasattr(scheduler, "sigmas"):
                current = getattr(scheduler, "step_index", None)
                noise_scale = scheduler.sigmas[index if current is None else current].to(x)
                alpha = (1 + noise_scale.square()).rsqrt()
                sigma = noise_scale * alpha
            else:
                alpha = model.alphas_bar[t.long()].sqrt()
                sigma = (1 - model.alphas_bar[t.long()]).sqrt()
            times = t.to(x).reshape(1, 1).expand(count, 1) + 1
            output = model.eps_model(network_input, times, cids=cids)
            _, eps = prediction_to_x0_eps(model.prediction_type, output, network_input, alpha, sigma)
            if name == "ddim":
                next_t = scheduler.timesteps[index + 1] if index + 1 < len(scheduler.timesteps) else None
                x = ddim_step(scheduler, eps, x, t, next_t, eta, generator)
            else:
                x = scheduler.step(eps, t, x, **step_options).prev_sample
            if history is not None:
                history.append(x.clone())
    return torch.stack(history) if history is not None else x


def ode_step(field, x, t, next_t, method):
    """One explicit ODE step; supports either direction of time."""
    dt = next_t - t
    k1 = field(x, t)
    if method == "euler":
        return x + dt * k1
    if method == "midpoint":
        return x + dt * field(x + dt * k1 / 2, t + dt / 2)
    if method == "heun":
        return x + dt * (k1 + field(x + dt * k1, next_t)) / 2
    if method == "rk4":
        k2 = field(x + dt * k1 / 2, t + dt / 2)
        k3 = field(x + dt * k2 / 2, t + dt / 2)
        k4 = field(x + dt * k3, next_t)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    raise ValueError(f"Unknown ODE method {method!r}")


@torch.no_grad()
def sample_continuous(model, shape, count, cids, name, steps, generator=None, trajectory=False, kwargs=None):
    if name not in ODE_SCHEDULERS + SDE_SCHEDULERS or steps < 1:
        raise ValueError(f"Choose positive steps and a continuous scheduler from {ODE_SCHEDULERS + SDE_SCHEDULERS}")
    if name in SDE_SCHEDULERS and model.trajectory not in ("vp_sde", "subvp", "ve"):
        raise ValueError("Reverse SDE sampling requires trajectory=vp_sde, subvp, or ve")
    kwargs = dict(kwargs or {})
    shift = kwargs.pop("shift", 1.0)
    if kwargs or shift <= 0:
        raise ValueError("Continuous ODE schedulers accept only shift > 0")
    # Only direct velocity/v predictions on endpoint-safe paths can evaluate
    # at pure noise or clean data. Other parameterizations divide by alpha/sigma.
    exact = model.trajectory in ("linear", "cosine") and model.prediction_type in ("velocity", "v")
    high, low = (1.0, 0.0) if exact else (1 - model.time_eps, model.time_eps)
    times = torch.linspace(high, low, steps + 1, device=model.device, dtype=model.dtype)
    times = shift * times / (1 + (shift - 1) * times)
    _, sigma, _, _ = model.coefficients(times[0])
    x = torch.randn(count, *shape, device=model.device, dtype=model.dtype, generator=generator) * sigma
    history = [x.clone()] if trajectory else None
    with evaluating(model):
        for t, next_t in zip(times[:-1], times[1:]):
            if name == "euler_maruyama":
                a, s, da, ds = model.coefficients(t)
                drift = da / a
                diffusion_squared = (2 * s * ds - 2 * drift * s.square()).clamp_min(0)
                score = model.score(x, t, cids)
                dt = next_t - t
                x = x + (drift * x - diffusion_squared * score) * dt
                x = x + (-dt * diffusion_squared).sqrt() * randn_like(x, generator)
            else:
                x = ode_step(lambda state, time: model.velocity(state, time, cids), x, t, next_t, name)
            if history is not None:
                history.append(x.clone())
        if not exact:
            x = model.predict_x0(x, times[-1], cids)
            if history is not None:
                history.append(x.clone())
    return torch.stack(history) if history is not None else x


@torch.no_grad()
def sample_sigma(model, shape, count, cids, name, steps, generator=None, trajectory=False, kwargs=None):
    consistency = model.model_type == "consistency"
    allowed = ("consistency",) if consistency else SIGMA_SCHEDULERS
    if name not in allowed or steps < 1:
        raise ValueError(f"Choose a positive step count and a scheduler from {allowed}")
    kwargs = dict(kwargs or {})
    spacing = kwargs.pop("spacing", "karras")
    rho = kwargs.pop("rho", model.rho)
    if kwargs or rho <= 0:
        raise ValueError("Sigma schedulers accept only spacing and rho > 0")
    sigmas = sigma_schedule(
        steps, model.sigma_min, model.sigma_max, rho, spacing, device=model.device, dtype=model.dtype
    )
    x = torch.randn(count, *shape, device=model.device, dtype=model.dtype, generator=generator) * sigmas[0]
    history = [x.clone()] if trajectory else None
    old_denoised, old_h = None, None
    with evaluating(model):
        for i, sigma in enumerate(sigmas):
            denoised = model.denoise_sigma(x, sigma, cids)
            next_sigma = sigmas[i + 1] if i + 1 < len(sigmas) else sigma.new_zeros(())
            if consistency:
                x = denoised
                if next_sigma > 0:
                    x = x + (next_sigma.square() - model.sigma_min**2).clamp_min(0).sqrt() * randn_like(x, generator)
            elif next_sigma == 0:
                x = denoised
            elif name == "dpmpp_2m":
                h = sigma.log() - next_sigma.log()
                corrected = denoised
                if old_denoised is not None:
                    r = old_h / h
                    corrected = (1 + 1 / (2 * r)) * denoised - old_denoised / (2 * r)
                x = (next_sigma / sigma) * x - torch.expm1(-h) * corrected
                old_denoised, old_h = denoised, h
            else:
                derivative = (x - denoised) / sigma
                if name == "euler_ancestral":
                    up = (next_sigma.square() * (sigma.square() - next_sigma.square()) / sigma.square()).sqrt()
                    down = (next_sigma.square() - up.square()).clamp_min(0).sqrt()
                    x = x + (down - sigma) * derivative + up * randn_like(x, generator)
                else:
                    proposal = x + (next_sigma - sigma) * derivative
                    if name == "heun":
                        d2 = (proposal - model.denoise_sigma(proposal, next_sigma, cids)) / next_sigma
                        x = x + (next_sigma - sigma) * (derivative + d2) / 2
                    else:
                        x = proposal
            if history is not None:
                history.append(x.clone())
    return torch.stack(history) if history is not None else x
