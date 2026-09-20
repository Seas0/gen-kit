"""Native PyTorch diffusion, flow matching, EDM and consistency objectives."""

from collections.abc import Sequence
from copy import deepcopy

import torch

from .ddpm import DDPM, make_beta_schedule
from .models import CondDenseModel, UNet
from .paths import expand, path_coefficients, prediction_to_x0_eps, prediction_type as normalize_prediction_type
from .paths import sigma_schedule, training_target
from .sampling import ODE_SCHEDULERS, SDE_SCHEDULERS, SIGMA_SCHEDULERS, VP_SCHEDULERS, sample_continuous, sample_sigma


MODEL_TYPES = ("diffusion", "score", "flow_matching", "rectified_flow", "edm", "consistency")


class GenerativeModel(DDPM):
    """One backbone with explicit objective, path, target and sampler choices.

    ``backbone`` is dense for vectors or unet for images. ``forward`` returns
    the raw training parameterization. Use ``generate`` for samples,
    ``predict_x0`` for a denoising estimate, and ``velocity`` for an ODE field.
    For EDM/consistency, ``denoise_sigma`` applies the required preconditioning.
    """

    def __init__(
        self,
        backbone: str = "dense",
        model_type: str = "diffusion",
        trajectory: str | None = None,
        prediction_type: str | None = None,
        in_features: int = 2,
        mid_features: Sequence[int] = (128, 128, 128),
        in_channels: int = 1,
        mid_channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        padding: int = 1,
        norm: str | None = "group",
        activation: str = "leaky_relu",
        num_resblocks: int = 3,
        upsample_mode: str = "conv_transpose",
        embed_dim: int = 128,
        num_classes: int | None = None,
        num_steps: int = 1000,
        schedule: str = "cosine",
        beta_range: tuple[float, float] = (1e-4, 0.02),
        cosine_s: float = 0.008,
        sigmoid_range: tuple[float, float] = (-5.0, 5.0),
        sampling_scheduler: str | None = None,
        sampling_steps: int | None = None,
        scheduler_kwargs: dict | None = None,
        time_sampling: str = "uniform",
        coupling: str = "independent",
        time_eps: float = 1e-4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        log_sigma_mean: float = -1.2,
        log_sigma_std: float = 1.2,
        consistency_mode: str = "training",
        consistency_steps: int = 40,
        ema_decay: float = 0.95,
        teacher_checkpoint: str | None = None,
        criterion: str = "mse",
        lr: float = 1e-3,
        lr_schedule: str | None = "constant",
        lr_interval: str = "epoch",
        lr_warmup: int = 0,
    ):
        hparams = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}")
        trajectory = (
            trajectory
            or {
                "diffusion": "vp",
                "score": "ve",
                "flow_matching": "linear",
                "rectified_flow": "linear",
                "edm": "edm",
                "consistency": "edm",
            }[model_type]
        )
        target = normalize_prediction_type(
            prediction_type
            or {
                "diffusion": "epsilon",
                "score": "score",
                "flow_matching": "velocity",
                "rectified_flow": "velocity",
                "edm": "x0",
                "consistency": "x0",
            }[model_type]
        )
        if model_type in ("edm", "consistency"):
            if trajectory != "edm" or target != "x0":
                raise ValueError("EDM/consistency require trajectory=edm and prediction_type=x0 (preconditioned)")
        elif trajectory not in ("vp", "linear", "cosine", "vp_sde", "subvp", "ve"):
            raise ValueError("Choose vp, linear, cosine, vp_sde, subvp, or ve for this objective")
        if model_type == "rectified_flow" and (trajectory != "linear" or target != "velocity"):
            raise ValueError("Rectified flow uses trajectory=linear and prediction_type=velocity")
        if model_type == "flow_matching" and (target != "velocity" or trajectory == "vp"):
            raise ValueError("Flow matching requires velocity prediction and a continuous trajectory")
        if trajectory == "vp" and target == "velocity":
            raise ValueError("Discrete VP has no continuous derivative; choose trajectory=vp_sde or cosine")
        if model_type == "score" and target != "score":
            raise ValueError("The score objective requires prediction_type=score")
        if not (0 < time_eps < 0.5 and 0 < sigma_min < sigma_max and sigma_data > 0 and rho > 0):
            raise ValueError("Require 0 < time_eps < .5, 0 < sigma_min < sigma_max, sigma_data > 0 and rho > 0")
        if not (0 < beta_min <= beta_max and log_sigma_std > 0 and 0 <= ema_decay < 1):
            raise ValueError("Invalid beta range, log_sigma_std or ema_decay")
        if consistency_mode not in ("training", "distillation") or consistency_steps < 2:
            raise ValueError("consistency_mode must be training/distillation and consistency_steps >= 2")
        if time_sampling not in ("uniform", "logit_normal"):
            raise ValueError("time_sampling must be uniform or logit_normal")
        if coupling not in ("independent", "ot"):
            raise ValueError("coupling must be independent or ot (minibatch optimal transport)")
        if coupling == "ot" and model_type not in ("flow_matching", "rectified_flow"):
            raise ValueError("OT coupling is supported for flow objectives")
        if trajectory == "vp" and time_sampling != "uniform":
            raise ValueError("Discrete VP uses uniform integer training timesteps")
        if backbone == "dense":
            if num_classes is not None:
                raise ValueError("The dense backbone has no class embeddings; use unet for class conditioning")
            network = CondDenseModel((in_features, *mid_features, in_features), activation, embed_dim)
        elif backbone == "unet":
            network = UNet.from_params(
                in_channels,
                mid_channels,
                kernel_size,
                padding,
                norm,
                activation,
                num_resblocks,
                upsample_mode,
                embed_dim,
                num_classes,
            )
        else:
            raise ValueError("backbone must be dense or unet")
        sampler = sampling_scheduler or (
            "consistency" if model_type == "consistency" else "heun" if trajectory != "vp" else "ddpm"
        )
        allowed = (
            ("consistency",)
            if model_type == "consistency"
            else SIGMA_SCHEDULERS
            if model_type == "edm"
            else tuple(VP_SCHEDULERS)
            if trajectory == "vp"
            else ODE_SCHEDULERS
        )
        if sampler not in allowed:
            if not (sampler in SDE_SCHEDULERS and trajectory in ("vp_sde", "subvp", "ve")):
                raise ValueError(
                    f"Scheduler {sampler!r} is incompatible with {model_type}/{trajectory}; choose {allowed}"
                )
        if sampling_steps is not None and sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        betas = make_beta_schedule(num_steps, schedule, beta_range, cosine_s, sigmoid_range)
        super().__init__(
            network,
            betas,
            criterion,
            lr,
            lr_schedule,
            lr_interval,
            lr_warmup,
            "epsilon" if target == "velocity" else target,
            sampler,
            sampling_steps,
            scheduler_kwargs,
        )
        self.prediction_type = target
        self.model_type, self.trajectory = model_type, trajectory
        self.time_sampling, self.time_eps = time_sampling, time_eps
        self.coupling = coupling
        self.beta_min, self.beta_max = beta_min, beta_max
        self.sigma_min, self.sigma_max, self.sigma_data, self.rho = sigma_min, sigma_max, sigma_data, rho
        self.log_sigma_mean, self.log_sigma_std = log_sigma_mean, log_sigma_std
        self.consistency_mode, self.consistency_steps, self.ema_decay = consistency_mode, consistency_steps, ema_decay
        self.teacher_checkpoint = teacher_checkpoint
        self.target_model = deepcopy(network).requires_grad_(False).eval() if model_type == "consistency" else None
        # A teacher is training-only and is deliberately excluded from student
        # checkpoints. Sampling a distilled checkpoint never needs the teacher.
        object.__setattr__(self, "teacher", None)
        self.hparams = hparams

    def train(self, mode=True):
        super().train(mode)
        if self.target_model is not None:
            self.target_model.eval()
        return self

    def coefficients(self, t):
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return path_coefficients(
            self.trajectory,
            t,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
        )

    def _time(self, t, x):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1)
        if t.shape[0] not in (1, x.shape[0]):
            raise ValueError("Expected one timestep or one timestep per sample")
        return t.expand(x.shape[0], 1)

    def _labels(self, cids, x):
        return None if cids is None else torch.as_tensor(cids, device=x.device).reshape(-1)

    def interpolate(self, x0, t, noise=None):
        """Continuous data-to-noise interpolation; t is in [0, 1]."""
        if self.trajectory == "vp":
            raise ValueError("For discrete VP use diffuse(x0, zero_based_indices)")
        times = self._time(t, x0)
        if (times < 0).any() or (times > 1).any():
            raise ValueError("Interpolation times must be in [0, 1]")
        a, s, _, _ = self.coefficients(times)
        noise = torch.randn_like(x0) if noise is None else noise
        return expand(a, x0) * x0 + expand(s, x0) * noise

    def diffuse(self, x0, tids, return_eps=False):
        if self.trajectory != "vp":
            raise ValueError("Use interpolate(x0, t) for continuous trajectories")
        return super().diffuse(x0, tids, return_eps)

    def diffuse_step(self, x, tidx):
        if self.trajectory != "vp":
            raise ValueError("Use interpolate(x0, t) for continuous trajectories")
        return super().diffuse_step(x, tidx)

    def denoise_step(self, x, tids, cids=None, random_sample=False):
        if self.trajectory != "vp":
            raise ValueError("Use generate(return_trajectory=True) for continuous/consistency sampling")
        return super().denoise_step(x, tids, cids, random_sample)

    def score(self, x, t, cids=None):
        times = self._time(t, x)
        a, s, da, ds = [expand(c, x) for c in self.coefficients(times)]
        output = self.eps_model(x, times * self.num_steps, cids=self._labels(cids, x))
        _, eps = prediction_to_x0_eps(self.prediction_type, output, x, a, s, da, ds)
        return -eps / s

    def predict_x0(self, x, t, cids=None):
        if self.model_type in ("edm", "consistency"):
            raise ValueError("Use denoise_sigma(x, sigma) for EDM and consistency models")
        if self.trajectory == "vp":
            raise ValueError("Use denoise_step for discrete VP or denoise_sigma for teacher adaptation")
        times = self._time(t, x)
        coefficients = [expand(c, x) for c in self.coefficients(times)]
        output = self.eps_model(x, times * self.num_steps, cids=self._labels(cids, x))
        return prediction_to_x0_eps(self.prediction_type, output, x, *coefficients)[0]

    def velocity(self, x, t, cids=None):
        if self.model_type in ("edm", "consistency") or self.trajectory == "vp":
            raise ValueError("velocity requires a continuous non-EDM trajectory")
        times = self._time(t, x)
        output = self.eps_model(x, times * self.num_steps, cids=self._labels(cids, x))
        if self.prediction_type == "velocity":
            return output
        a, s, da, ds = [expand(c, x) for c in self.coefficients(times)]
        x0, eps = prediction_to_x0_eps(self.prediction_type, output, x, a, s, da, ds)
        return da * x0 + ds * eps

    def denoise_sigma(self, x, sigma, cids=None, *, target=False):
        if self.model_type not in ("edm", "consistency"):
            if self.trajectory == "vp":
                return super().denoise_sigma(x, sigma, cids)
            if self.trajectory != "ve":
                raise ValueError("Sigma denoising requires EDM, VE or VP")
            sigma = self._time(sigma, x)
            times = (sigma / self.sigma_min).log() / x.new_tensor(self.sigma_max / self.sigma_min).log()
            return self.predict_x0(x, times, cids)
        sigma = self._time(sigma, x)
        if (sigma <= 0).any():
            raise ValueError("sigma must be positive")
        s = expand(sigma, x)
        variance = s.square() + self.sigma_data**2
        boundary = s - self.sigma_min if self.model_type == "consistency" else s
        skip = self.sigma_data**2 / (boundary.square() + self.sigma_data**2)
        out_scale = boundary * self.sigma_data / variance.sqrt()
        network = self.target_model if target else self.eps_model
        output = network(x / variance.sqrt(), sigma.log() / 4, cids=self._labels(cids, x))
        return skip * x + out_scale * output

    def loss(self, x, cids=None, *, noise=None, times=None):
        if self.trajectory == "vp":
            if noise is not None or times is not None:
                raise ValueError("Explicit noise/times are supported for continuous objectives")
            return super().loss(x, cids)
        cids = self._labels(cids, x)
        noise = torch.randn_like(x) if noise is None else noise
        if self.coupling == "ot":
            from scipy.optimize import linear_sum_assignment

            cost = torch.cdist(x.detach().flatten(1), noise.detach().flatten(1)).square()
            _, columns = linear_sum_assignment(cost.cpu().numpy())
            noise = noise[torch.as_tensor(columns, device=x.device)]
        if self.model_type == "consistency":
            return self.consistency_loss(x, cids, noise)
        if self.model_type == "edm":
            sigma = (
                (torch.randn(x.shape[0], 1, device=x.device) * self.log_sigma_std + self.log_sigma_mean).exp()
                if times is None
                else self._time(times, x)
            )
            s = expand(sigma, x)
            denoised = self.denoise_sigma(x + s * noise, sigma, cids)
            weight = (s.square() + self.sigma_data**2) / (s * self.sigma_data).square()
            # Scaling the residual works for both MSE and MAE.
            return self.criterion(weight.sqrt() * denoised, weight.sqrt() * x)
        if times is None:
            times = (
                torch.rand(x.shape[0], 1, device=x.device)
                if self.time_sampling == "uniform"
                else torch.randn(x.shape[0], 1, device=x.device).sigmoid()
            )
            times = self.time_eps + (1 - 2 * self.time_eps) * times
        times = self._time(times, x)
        if (times <= 0).any() or (times >= 1).any():
            raise ValueError("Training times must lie strictly between 0 and 1")
        a, s, da, ds = [expand(c, x) for c in self.coefficients(times)]
        xt = a * x + s * noise
        output = self.eps_model(xt, times * self.num_steps, cids=cids)
        target = training_target(self.prediction_type, x, noise, a, s, da, ds)
        if self.prediction_type == "score":
            return self.criterion(s * output, s * target)
        return self.criterion(output, target)

    def set_teacher(self, teacher):
        if self.model_type != "consistency" or self.consistency_mode != "distillation":
            raise ValueError("A teacher is only used for consistency distillation")
        if not isinstance(teacher, DDPM) or getattr(teacher, "model_type", "diffusion") == "consistency":
            raise ValueError("The teacher must be a diffusion/EDM denoiser")
        if teacher.sampling_scheduler == "distilled":
            raise ValueError("Consistency distillation requires an undistilled diffusion/EDM teacher")
        if isinstance(teacher, GenerativeModel) and teacher.trajectory not in ("vp", "ve", "edm"):
            raise ValueError("The teacher must use a VP, VE or EDM trajectory")
        if teacher.class_cond != self.class_cond:
            raise ValueError("Teacher and student must use the same class conditioning")
        backbone = teacher.hparams.get("backbone", "dense" if "in_features" in teacher.hparams else "unet")
        if backbone != self.hparams["backbone"]:
            raise ValueError("Teacher and student input types differ")
        for key in ("in_features", "in_channels", "num_classes"):
            if key in teacher.hparams and teacher.hparams[key] != self.hparams[key]:
                raise ValueError(f"Teacher and student disagree on {key}")
        if getattr(teacher, "trajectory", "vp") == "vp":
            grid = ((1 - teacher.alphas_bar) / teacher.alphas_bar).sqrt()
            minimum, maximum = grid[0].item(), grid[-1].item()
        else:
            minimum, maximum = teacher.sigma_min, teacher.sigma_max
        if self.sigma_min < minimum * (1 - 1e-5) or self.sigma_max > maximum * (1 + 1e-5):
            raise ValueError(f"Student sigma range must fit the teacher range [{minimum:.6g}, {maximum:.6g}]")
        teacher = teacher.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        object.__setattr__(self, "teacher", teacher)

    def prepare_training(self):
        if self.model_type == "consistency" and self.consistency_mode == "distillation" and self.teacher is None:
            if self.teacher_checkpoint is None:
                raise ValueError("Consistency distillation requires teacher_checkpoint or set_teacher(teacher)")
            from .training import load_model

            self.set_teacher(load_model(self.teacher_checkpoint))

    def consistency_loss(self, x, cids, noise):
        grid = sigma_schedule(
            self.consistency_steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            device=x.device,
            dtype=x.dtype,
        )
        index = torch.randint(len(grid) - 1, (x.shape[0],), device=x.device)
        high, low = grid[index], grid[index + 1]
        noisy = x + expand(high, x) * noise
        prediction = self.denoise_sigma(noisy, high, cids)
        with torch.no_grad():
            if self.consistency_mode == "training":
                # Coupled adjacent perturbations using the SAME noise realization.
                next_x = x + expand(low, x) * noise
            else:
                self.prepare_training()
                teacher = self.teacher.to(device=x.device, dtype=x.dtype)
                first = (noisy - teacher.denoise_sigma(noisy, high, cids)) / expand(high, x)
                proposal = noisy + expand(low - high, x) * first
                second = (proposal - teacher.denoise_sigma(proposal, low, cids)) / expand(low, x)
                next_x = noisy + expand(low - high, x) * (first + second) / 2
            target = self.denoise_sigma(next_x, low, cids, target=True)
        return self.criterion(prediction, target)

    @torch.no_grad()
    def update_ema(self):
        """Update the stop-gradient consistency target once per optimizer step."""
        if self.target_model is None:
            return
        for target, online in zip(self.target_model.parameters(), self.eps_model.parameters()):
            target.lerp_(online, 1 - self.ema_decay)
        for target, online in zip(self.target_model.buffers(), self.eps_model.buffers()):
            target.copy_(online)

    @torch.no_grad()
    def generate(
        self,
        sample_shape,
        cids=None,
        num_samples=1,
        *,
        scheduler=None,
        num_steps=None,
        generator=None,
        return_trajectory=False,
        scheduler_kwargs=None,
    ):
        if num_samples < 1 or any(size < 1 for size in sample_shape):
            raise ValueError("Sample dimensions and num_samples must be positive")
        if self.trajectory == "vp":
            return super().generate(
                sample_shape,
                cids,
                num_samples,
                scheduler=scheduler,
                num_steps=num_steps,
                generator=generator,
                return_trajectory=return_trajectory,
                scheduler_kwargs=scheduler_kwargs,
            )
        default_steps = 1 if self.model_type == "consistency" else 40
        steps = num_steps if num_steps is not None else (self.sampling_steps or default_steps)
        sample = sample_sigma if self.model_type in ("edm", "consistency") else sample_continuous
        return sample(
            self,
            sample_shape,
            num_samples,
            cids,
            scheduler or self.sampling_scheduler,
            steps,
            generator,
            return_trajectory,
            {**self.scheduler_kwargs, **(scheduler_kwargs or {})},
        )
