"""Distribution and trajectory distillation with native PyTorch updates.

The DMD gradient follows Yin et al. (2311.18828, 2405.14867). DMD2 uses
a learned fake score, shared-feature adversarial head, two-timescale updates
and backward simulation. See docs/model_choices.md for demo-level choices.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .ddpm import DDPM
from .generative import GenerativeModel
from .paths import expand, prediction_to_x0_eps, training_target
from .sampling import evaluating, ode_step, randn_like


DISTILLATION_METHODS = ("dmd", "dmd2", "progressive", "trajectory", "reflow")


@contextmanager
def frozen(module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    mode = module.training
    module.requires_grad_(False).eval()
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)
        module.train(mode)


def distribution_matching_loss(generated, real_x0, fake_x0, eps=1e-6):
    """Stop-gradient score-difference surrogate; gradient points fake -> real.

    The per-example normalization is the mean absolute teacher residual.
    This function backpropagates through generated data only, never scores.
    """
    with torch.no_grad():
        scale = (generated - real_x0).abs().flatten(1).mean(1)
        gradient = (fake_x0 - real_x0) / expand(scale.clamp_min(eps), generated)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite distribution-matching gradient")
        target = generated.detach() - gradient
    return 0.5 * F.mse_loss(generated, target)


class DistillationModel(GenerativeModel):
    """Few-step student with DMD/DMD2, progressive, paired or reflow training.

    DMD/progressive/paired methods currently require a VP diffusion teacher
    with the same beta table. Reflow requires a linear velocity teacher.
    Architecture and optimizer options are the same as GenerativeModel.
    """

    def __init__(
        self,
        method: str = "dmd2",
        teacher_checkpoint: str | None = None,
        student_steps: int = 1,
        teacher_steps: int = 32,
        teacher_start_step: int | None = None,
        critic_updates: int | None = None,
        critic_lr: float = 1e-4,
        regression_weight: float | None = None,
        gan_weight: float | None = None,
        dm_min_time: float = 0.02,
        dm_max_time: float = 0.98,
        initialize_from_teacher: bool = True,
        **model_kwargs,
    ):
        if method not in DISTILLATION_METHODS:
            raise ValueError(f"method must be one of {DISTILLATION_METHODS}")
        if student_steps < 1 or teacher_steps < 1 or critic_lr <= 0:
            raise ValueError("Step counts and critic_lr must be positive")
        if not 0 <= dm_min_time < dm_max_time < 1:
            raise ValueError("Require 0 <= dm_min_time < dm_max_time < 1")
        if method in ("dmd", "trajectory") and student_steps != 1:
            raise ValueError("DMD and paired trajectory regression train one-step students")
        if method == "progressive" and teacher_steps != 2 * student_steps:
            raise ValueError("A progressive stage requires teacher_steps = 2 * student_steps")
        defaults = (
            {
                "model_type": "rectified_flow",
                "trajectory": "linear",
                "prediction_type": "velocity",
                "sampling_scheduler": "euler",
            }
            if method == "reflow"
            else {"model_type": "diffusion", "trajectory": "vp", "prediction_type": "v", "sampling_scheduler": "ddim"}
        )
        defaults.update(model_kwargs)
        if method != "reflow":
            if defaults["sampling_scheduler"] not in (None, "ddim", "distilled"):
                raise ValueError("Distilled VP students require sampling_scheduler=distilled")
            defaults["sampling_scheduler"] = "ddim"
        defaults["sampling_steps"] = student_steps
        super().__init__(teacher_checkpoint=teacher_checkpoint, **defaults)
        if method == "reflow":
            if self.trajectory != "linear" or self.prediction_type != "velocity":
                raise ValueError("Reflow requires a linear velocity student")
        elif self.trajectory != "vp" or self.model_type not in ("diffusion", "score"):
            raise ValueError("DMD/DMD2/progressive/trajectory students currently require a VP diffusion path")
        if student_steps > self.num_steps or (method == "progressive" and teacher_steps > self.num_steps):
            raise ValueError("Distillation grids cannot exceed num_steps")
        if teacher_steps > self.num_steps and method in ("dmd", "trajectory"):
            raise ValueError("teacher_steps cannot exceed the VP training grid")
        if teacher_start_step is not None:
            if method not in ("dmd", "trajectory"):
                raise ValueError("teacher_start_step is only supported for DMD/trajectory regression")
            if not isinstance(teacher_start_step, int) or not teacher_steps - 1 <= teacher_start_step < self.num_steps:
                raise ValueError("Require teacher_steps - 1 <= teacher_start_step < num_steps")
        ratio = critic_updates if critic_updates is not None else (5 if method == "dmd2" else 1)
        regression = regression_weight if regression_weight is not None else (1.0 if method == "dmd" else 0.0)
        adversarial = gan_weight if gan_weight is not None else (0.003 if method == "dmd2" else 0.0)
        if ratio < 1 or regression < 0 or adversarial < 0:
            raise ValueError("Invalid update ratio or loss weight")
        if method == "dmd" and regression <= 0:
            raise ValueError("DMD requires paired regression; use method=dmd2 to remove it")
        if method == "dmd2" and regression != 0:
            raise ValueError("DMD2 removes paired regression; regression_weight must be zero")
        if method != "dmd2" and adversarial:
            raise ValueError("The adversarial objective is supported by DMD2")
        self.method, self.student_steps, self.teacher_steps = method, student_steps, teacher_steps
        self.teacher_start_step = self.num_steps - 1 if teacher_start_step is None else teacher_start_step
        self.critic_updates, self.critic_lr = ratio, critic_lr
        self.regression_weight, self.gan_weight = regression, adversarial
        self.dm_min_time, self.dm_max_time = dm_min_time, dm_max_time
        self.initialize_from_teacher = initialize_from_teacher
        self.fake_score = deepcopy(self.eps_model) if method in ("dmd", "dmd2") else None
        width = (
            (self.hparams["mid_features"][-1] if self.hparams["mid_features"] else self.hparams["in_features"])
            if self.hparams["backbone"] == "dense"
            else self.hparams["mid_channels"][-1]
        )
        self.discriminator_head = nn.Linear(width, 1) if adversarial else None
        self.register_buffer("distill_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("teacher_initialized", torch.tensor(False))
        if method != "reflow":
            self.sampling_scheduler = "distilled"
            self.hparams["sampling_scheduler"] = "distilled"
        self.hparams.update(
            method=method,
            student_steps=student_steps,
            teacher_steps=teacher_steps,
            critic_updates=ratio,
            critic_lr=critic_lr,
            regression_weight=regression,
            gan_weight=adversarial,
            dm_min_time=dm_min_time,
            dm_max_time=dm_max_time,
            initialize_from_teacher=initialize_from_teacher,
        )
        # Preserve constructor metadata for checkpoints made before this option.
        if teacher_start_step is not None:
            self.hparams["teacher_start_step"] = teacher_start_step

    def set_teacher(self, teacher):
        if not isinstance(teacher, DDPM):
            raise ValueError("Teacher must be a model from this repository")
        if self.method == "reflow":
            if getattr(teacher, "trajectory", None) != "linear" or teacher.prediction_type != "velocity":
                raise ValueError("Reflow requires a linear velocity teacher")
        else:
            if getattr(teacher, "trajectory", "vp") != "vp":
                raise ValueError("This distillation method requires a VP diffusion teacher")
            if teacher.betas.shape != self.betas.shape or not torch.allclose(teacher.betas.to(self.betas), self.betas):
                raise ValueError("Teacher and student must use the same trained beta schedule")
        if teacher.class_cond != self.class_cond:
            raise ValueError("Teacher and student class conditioning differs")
        teacher_backbone = teacher.hparams.get("backbone", "dense" if "in_features" in teacher.hparams else "unet")
        if teacher_backbone != self.hparams["backbone"]:
            raise ValueError("Teacher and student input types differ")
        for key in ("in_features", "in_channels", "num_classes"):
            if key in teacher.hparams and teacher.hparams[key] != self.hparams[key]:
                raise ValueError(f"Teacher and student disagree on {key}")
        if isinstance(teacher, DistillationModel):
            if self.method in ("dmd", "dmd2"):
                raise ValueError("Distribution matching requires an undistilled diffusion teacher")
            if self.method in ("progressive", "trajectory"):
                if teacher.method != "progressive" or teacher.student_steps != self.teacher_steps:
                    raise ValueError(
                        "The previous progressive stage must have student_steps = this stage's teacher_steps"
                    )
        teacher = teacher.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        object.__setattr__(self, "teacher", teacher)
        if self.initialize_from_teacher and not self.teacher_initialized:
            if teacher.prediction_type != self.prediction_type:
                raise ValueError("Teacher initialization requires matching prediction_type")
            try:
                self.eps_model.load_state_dict(teacher.eps_model.state_dict())
                if self.fake_score is not None:
                    self.fake_score.load_state_dict(teacher.eps_model.state_dict())
            except RuntimeError as error:
                raise ValueError("Teacher initialization requires matching backbone sizes") from error
        self.teacher_initialized.fill_(True)

    def prepare_training(self):
        if self.teacher is None:
            if not self.teacher_checkpoint:
                raise ValueError("Distillation requires teacher_checkpoint or set_teacher(teacher)")
            from .training import load_model

            self.set_teacher(load_model(self.teacher_checkpoint))
        else:
            # The teacher is intentionally not registered as a child module,
            # so moving the student after set_teacher() must also move it here.
            self.teacher.to(device=self.device, dtype=self.dtype)

    def configure_optimizers(self, max_epochs=1, max_steps=1):
        # Only generator parameters belong to the main optimizer. Reuse the
        # ordinary LR schedule construction without including the critic.
        auxiliary = []
        for module in (self.fake_score, self.discriminator_head):
            if module is not None:
                auxiliary.extend(module.parameters())
        flags = [parameter.requires_grad for parameter in auxiliary]
        for parameter in auxiliary:
            parameter.requires_grad_(False)
        try:
            if self.fake_score is not None:
                max_steps = max(1, max_steps // self.critic_updates)
            return super().configure_optimizers(max_epochs, max_steps)
        finally:
            for parameter, flag in zip(auxiliary, flags):
                parameter.requires_grad_(flag)

    def configure_auxiliary_optimizers(self):
        if self.fake_score is None:
            return {}
        parameters = list(self.fake_score.parameters())
        if self.discriminator_head is not None:
            parameters.extend(self.discriminator_head.parameters())
        return {"critic": torch.optim.Adam(parameters, lr=self.critic_lr)}

    def _grid(self, steps):
        return torch.linspace(self.num_steps - 1, -1, steps + 1, device=self.device).round().long()

    def _coefficients(self, tids, x):
        tids = torch.as_tensor(tids, device=x.device).long().reshape(-1)
        alpha_bar = self.alphas_bar[tids.clamp_min(0)]
        alpha_bar = torch.where(tids < 0, torch.ones_like(alpha_bar), alpha_bar)
        return expand(alpha_bar.sqrt(), x), expand((1 - alpha_bar).sqrt(), x)

    def _prediction(self, network, x, tids, cids=None, kind=None):
        times = torch.as_tensor(tids, device=x.device, dtype=x.dtype).reshape(-1, 1) + 1
        a, s = self._coefficients(tids, x)
        output = network(x, times, cids=cids)
        return prediction_to_x0_eps(kind or self.prediction_type, output, x, a, s)

    def _ddim_step(self, network, x, t, next_t, cids=None, kind=None):
        x0, eps = self._prediction(network, x, t, cids, kind)
        a, s = self._coefficients(next_t, x)
        return a * x0 + s * eps

    def _student_sample(self, noise, cids=None, *, train_stage=False, generator=None, history=False):
        grid, x = self._grid(self.student_steps), noise
        states = [x.clone()] if history else None
        if self.method == "dmd2":
            selected = (
                int(torch.randint(self.student_steps, (), device=x.device)) if train_stage else self.student_steps - 1
            )
            for index in range(selected + 1):
                if train_stage and index < selected:
                    with torch.no_grad():
                        clean, _ = self._prediction(self.eps_model, x, grid[index], cids)
                else:
                    clean, _ = self._prediction(self.eps_model, x, grid[index], cids)
                if index < selected:
                    a, s = self._coefficients(grid[index + 1], x)
                    x = a * clean + s * randn_like(x, generator)
                else:
                    x = clean
                if states is not None:
                    states.append(x.clone())
        else:
            for t, next_t in zip(grid[:-1], grid[1:]):
                x = self._ddim_step(self.eps_model, x, t, next_t, cids)
                if states is not None:
                    states.append(x.clone())
        return torch.stack(states) if states is not None else x

    @torch.no_grad()
    def _teacher_sample(self, noise, cids=None):
        self.prepare_training()
        teacher, x = self.teacher, noise
        if self.method == "reflow":
            times = torch.linspace(1, 0, self.teacher_steps + 1, device=x.device, dtype=x.dtype)
            for t, next_t in zip(times[:-1], times[1:]):
                x = ode_step(lambda state, time: teacher.velocity(state, time, cids), x, t, next_t, "heun")
        else:
            # Epsilon teachers can amplify errors at nearly zero terminal SNR.
            # An explicit earlier start keeps the selected grid reproducible.
            grid = torch.linspace(self.teacher_start_step, -1, self.teacher_steps + 1, device=x.device).round().long()
            for t, next_t in zip(grid[:-1], grid[1:]):
                x = self._ddim_step(teacher.eps_model, x, t, next_t, cids, teacher.prediction_type)
        return x

    def _critic_logits(self, x, cids=None):
        times = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if self.hparams["backbone"] == "dense":
            features = x
            for layer in self.fake_score.dense_layers[:-1]:
                features = layer(features, times)
        else:
            features = self.fake_score.encoder(x, times, cids=cids)[-1]
            if self.fake_score.bottleneck is not None:
                features = self.fake_score.bottleneck(features, times, cids=cids)
            features = features.mean(dim=(-2, -1))
        return self.discriminator_head(features)

    def _dm_loss(self, generated, cids):
        self.prepare_training()
        with torch.no_grad(), evaluating(self.fake_score):
            low, high = int(self.dm_min_time * self.num_steps), int(self.dm_max_time * self.num_steps)
            tids = torch.randint(low, high + 1, (generated.shape[0],), device=generated.device)
            a, s = self._coefficients(tids, generated)
            noisy = a * generated.detach() + s * torch.randn_like(generated)
            real_x0, _ = self._prediction(self.teacher.eps_model, noisy, tids, cids, self.teacher.prediction_type)
            fake_x0, _ = self._prediction(self.fake_score, noisy, tids, cids)
        return distribution_matching_loss(generated, real_x0, fake_x0)

    def generator_loss(self, x, cids=None):
        self.prepare_training()
        noise = torch.randn_like(x)
        generated = self._student_sample(noise, cids, train_stage=self.method == "dmd2")
        losses = {}
        if self.method == "trajectory":
            losses["regression_loss"] = self.criterion(generated, self._teacher_sample(noise, cids))
        else:
            losses["dm_loss"] = self._dm_loss(generated, cids)
            if self.regression_weight:
                losses["regression_loss"] = self.regression_weight * self.criterion(
                    generated, self._teacher_sample(noise, cids)
                )
            if self.gan_weight:
                with frozen(self.fake_score), frozen(self.discriminator_head):
                    losses["generator_gan_loss"] = (
                        self.gan_weight * F.softplus(-self._critic_logits(generated, cids)).mean()
                    )
        return sum(losses.values()), losses

    def critic_loss(self, real, cids=None):
        with torch.no_grad(), evaluating(self.eps_model):
            fake = self._student_sample(torch.randn_like(real), cids, train_stage=self.method == "dmd2").detach()
        tids = torch.randint(self.num_steps, (real.shape[0],), device=real.device)
        a, s = self._coefficients(tids, fake)
        noise = torch.randn_like(fake)
        output = self.fake_score(a * fake + s * noise, (tids + 1).to(fake).reshape(-1, 1), cids=cids)
        target = training_target(self.prediction_type, fake, noise, a, s)
        loss = (
            self.criterion(s * output, s * target)
            if self.prediction_type == "score"
            else self.criterion(output, target)
        )
        losses = {"fake_score_loss": loss}
        if self.gan_weight:
            logits_real, logits_fake = self._critic_logits(real.detach(), cids), self._critic_logits(fake, cids)
            losses["critic_gan_loss"] = self.gan_weight * (
                F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
            )
        return sum(losses.values()), losses

    def progressive_target(self, x, noise, index, cids=None):
        self.prepare_training()
        grid = self._grid(self.teacher_steps)
        t, middle, end = grid[2 * index], grid[2 * index + 1], grid[2 * index + 2]
        a, s = self._coefficients(t, x)
        xt = a * x + s * noise
        with torch.no_grad():
            mid = self._ddim_step(self.teacher.eps_model, xt, t, middle, cids, self.teacher.prediction_type)
            final = self._ddim_step(self.teacher.eps_model, mid, middle, end, cids, self.teacher.prediction_type)
            a_end, s_end = self._coefficients(end, x)
            ratio = s_end / s
            x0_target = (final - ratio * xt) / (a_end - ratio * a)
            eps_target = (xt - a * x0_target) / s
            target = training_target(self.prediction_type, x0_target, eps_target, a, s)
        return xt, t, target, final, end

    def loss(self, x, cids=None, **kwargs):
        if kwargs:
            raise ValueError("Distillation draws its own teacher pairs and training times")
        self.prepare_training()
        cids = self._labels(cids, x)
        if self.method == "reflow":
            noise = torch.randn_like(x)
            paired = self._teacher_sample(noise, cids)
            return super().loss(paired, cids, noise=noise)
        if self.method == "progressive":
            index = torch.randint(self.student_steps, (x.shape[0],), device=x.device)
            xt, t, target, _, _ = self.progressive_target(x, torch.randn_like(x), index, cids)
            output = self.eps_model(xt, (t + 1).to(x).reshape(-1, 1), cids=cids)
            if self.prediction_type == "score":
                _, s = self._coefficients(t, x)
                return self.criterion(s * output, s * target)
            return self.criterion(output, target)
        return self.generator_loss(x, cids)[0]

    def optimize_batch(self, batch, batch_idx, optimizer, auxiliary, gradient_clip_val=0.0):
        batch = self._get_batch(batch)
        x, cids = (batch, None) if isinstance(batch, torch.Tensor) else batch
        self.prepare_training()
        metrics = {}

        def step(loss, opt):
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [p for group in opt.param_groups for p in group["params"]]
            if gradient_clip_val > 0:
                nn.utils.clip_grad_norm_(parameters, gradient_clip_val, error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if self.fake_score is None:
            loss = self.loss(x, cids)
            step(loss, optimizer)
            self.distill_updates.add_(1)
            return loss.detach(), {"generator_loss": loss.item()}, True
        critic_loss, parts = self.critic_loss(x, cids)
        step(critic_loss, auxiliary["critic"])
        metrics.update({key: value.item() for key, value in parts.items()})
        metrics["critic_loss"] = critic_loss.item()
        self.distill_updates.add_(1)
        update_generator = int(self.distill_updates) % self.critic_updates == 0
        total = critic_loss.detach()
        if update_generator:
            loss, parts = self.generator_loss(x, cids)
            step(loss, optimizer)
            metrics.update({key: value.item() for key, value in parts.items()})
            metrics["generator_loss"] = loss.item()
            total = total + loss.detach()
        return total, metrics, update_generator

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
        if self.method == "reflow":
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
        if scheduler not in (None, "distilled") or scheduler_kwargs:
            raise ValueError(
                "This student uses its trained distilled sampler; scheduler options are not interchangeable"
            )
        if num_steps is not None and num_steps != self.student_steps:
            raise ValueError(f"This student was trained for {self.student_steps} sampling steps")
        if num_samples < 1 or any(size < 1 for size in sample_shape):
            raise ValueError("Sample dimensions and num_samples must be positive")
        noise = torch.randn(num_samples, *sample_shape, device=self.device, dtype=self.dtype, generator=generator)
        cids = self._labels(cids, noise)
        with evaluating(self):
            return self._student_sample(noise, cids, generator=generator, history=return_trajectory)
