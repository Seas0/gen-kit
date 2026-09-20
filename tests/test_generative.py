"""Analytic contracts, sampler compatibility and native training integration."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from diffusion import DDPM2d, DDPMTab, GenerativeModel, VP_SCHEDULERS
from diffusion.data import SwissRollDataModule
from diffusion.paths import expand, path_coefficients, prediction_to_x0_eps, training_target
from diffusion.sampling import make_vp_scheduler, ode_step
from diffusion.training import Trainer, instantiate, load_model, read_config, seed_everything


torch.set_num_threads(1)


@pytest.mark.parametrize("path", ["linear", "cosine", "vp_sde", "subvp", "ve"])
@pytest.mark.parametrize("kind", ["epsilon", "x0", "v", "score", "velocity"])
@pytest.mark.parametrize("shape", [(4, 2), (4, 1, 8, 8)])
def test_prediction_round_trip(path, kind, shape):
    x0, noise = torch.randn(shape, dtype=torch.float64), torch.randn(shape, dtype=torch.float64)
    times = torch.tensor([0.05, 0.25, 0.6, 0.95], dtype=torch.float64)
    coefficients = [expand(value, x0) for value in path_coefficients(path, times)]
    a, s, da, ds = coefficients
    xt = a * x0 + s * noise
    target = training_target(kind, x0, noise, *coefficients)
    recovered, eps = prediction_to_x0_eps(kind, target, xt, *coefficients)
    torch.testing.assert_close(recovered, x0, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(eps, noise, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("path", ["linear", "cosine", "vp_sde", "subvp", "ve"])
def test_path_derivatives(path):
    times = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)
    _, _, da, ds = path_coefficients(path, times)
    ap, sp, _, _ = path_coefficients(path, times + 1e-6)
    am, sm, _, _ = path_coefficients(path, times - 1e-6)
    torch.testing.assert_close(da, (ap - am) / 2e-6)
    torch.testing.assert_close(ds, (sp - sm) / 2e-6)


class Oracle(nn.Module):
    def __init__(self, model, target):
        super().__init__()
        self.register_buffer("sigmas", ((1 - model.alphas_bar) / model.alphas_bar).sqrt())
        self.target = target

    def forward(self, x, t, cids=None):
        index = (t.flatten() - 1).clamp(0, len(self.sigmas) - 1)
        lo, hi = index.floor().long(), index.ceil().long()
        raw_sigma = self.sigmas[lo] + (index - lo) * (self.sigmas[hi] - self.sigmas[lo])
        a = expand((1 + raw_sigma.square()).rsqrt(), x)
        s = expand(raw_sigma, x) * a
        x0 = torch.full_like(x, 0.3)
        noise = (x - a * x0) / s
        return training_target(self.target, x0, noise, a, s)


@pytest.mark.parametrize("scheduler", VP_SCHEDULERS)
def test_vp_sampler_target_equivalence(scheduler):
    outputs = []
    for kind in ("epsilon", "x0", "v", "score"):
        model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=40, schedule="linear", prediction_type=kind)
        model.set_model(Oracle(model, kind))
        outputs.append(
            model.generate(
                (2,), num_samples=3, scheduler=scheduler, num_steps=8, generator=torch.Generator().manual_seed(22)
            )
        )
        assert outputs[-1].shape == (3, 2)
        assert outputs[-1].isfinite().all()
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=2e-4, rtol=2e-4)
    # Multistep epsilon solvers approximate this varying oracle field; DEIS
    # and the original DPM-Solver also retain a positive terminal sigma.
    if scheduler not in ("dpm_solver", "pndm", "plms", "deis"):
        torch.testing.assert_close(outputs[0], torch.full_like(outputs[0], 0.3), atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("method", ["euler", "midpoint", "heun", "rk4"])
def test_ode_step_direction(method):
    x = torch.randn(3, 2)
    result = ode_step(lambda state, t: torch.ones_like(state) * 2, x, torch.tensor(1.0), torch.tensor(0.0), method)
    torch.testing.assert_close(result, x - 2)


@pytest.mark.parametrize("spacing", ["linspace", "leading", "trailing"])
def test_ddim_uses_actual_selected_grid(spacing):
    class ConstantNoise(nn.Module):
        def forward(self, x, t, cids=None):
            return torch.full_like(x, 0.2)

    model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=40, schedule="linear")
    model.set_model(ConstantNoise())
    options = {"timestep_spacing": spacing}
    grid = make_vp_scheduler("ddim", model.betas, options)
    grid.set_timesteps(7)
    states = model.generate(
        (2,),
        num_samples=3,
        scheduler="ddim",
        num_steps=7,
        scheduler_kwargs=options,
        return_trajectory=True,
    )
    start_alpha = model.alphas_bar[grid.timesteps[0]]
    clean = (states[0] - (1 - start_alpha).sqrt() * 0.2) / start_alpha.sqrt()
    for state, timestep in zip(states[1:-1], grid.timesteps[1:]):
        alpha = model.alphas_bar[timestep]
        torch.testing.assert_close(state, alpha.sqrt() * clean + (1 - alpha).sqrt() * 0.2)
    torch.testing.assert_close(states[-1], clean)


@pytest.mark.parametrize("scheduler", [name for name in VP_SCHEDULERS if name not in ("pndm", "plms")])
def test_one_step_vp_starts_at_terminal_noise(scheduler):
    class RecordTime(nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []

        def forward(self, x, t, cids=None):
            self.times.append(t.detach().clone())
            return torch.zeros_like(x)

    model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=40, schedule="linear")
    network = RecordTime()
    model.set_model(network)
    result = model.generate((2,), scheduler=scheduler, num_steps=1)
    assert result.isfinite().all()
    torch.testing.assert_close(network.times[0], torch.full_like(network.times[0], model.num_steps))


@pytest.mark.parametrize(
    "model_type,path,target",
    [
        ("diffusion", "vp", "epsilon"),
        ("diffusion", "vp", "x0"),
        ("diffusion", "vp", "v"),
        ("score", "vp", "score"),
        ("score", "ve", "score"),
        ("score", "subvp", "score"),
        ("flow_matching", "linear", "velocity"),
        ("flow_matching", "cosine", "velocity"),
        ("flow_matching", "vp_sde", "velocity"),
        ("rectified_flow", "linear", "velocity"),
        ("edm", "edm", "x0"),
        ("consistency", "edm", "x0"),
    ],
)
@pytest.mark.parametrize("image", [False, True])
def test_training_and_sampling_all_families(model_type, path, target, image):
    model = GenerativeModel(
        backbone="unet" if image else "dense",
        model_type=model_type,
        trajectory=path,
        prediction_type=target,
        mid_features=(8,),
        mid_channels=(4, 8),
        embed_dim=8,
        num_steps=32,
        num_resblocks=1,
        num_classes=10 if image else None,
        sigma_max=5.0,
    )
    x = torch.randn(2, 1, 8, 8) if image else torch.randn(2, 2)
    cids = torch.tensor([1, 4]) if image else None
    loss = model.loss(x, cids)
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.eps_model.parameters())
    out = model.generate(x.shape[1:], cids=cids, num_samples=2, num_steps=4)
    assert out.shape == x.shape and out.isfinite().all()


def test_consistency_boundary_ema_and_distillation():
    model = GenerativeModel(model_type="consistency", mid_features=(8,), embed_dim=8, num_steps=16, sigma_max=5.0)
    x = torch.randn(4, 2)
    torch.testing.assert_close(model.denoise_sigma(x, model.sigma_min), x)
    previous = [p.clone() for p in model.target_model.parameters()]
    optimizer = model.configure_optimizers()[0][0]
    loss = model.loss(x)
    loss.backward()
    assert all(p.grad is None for p in model.target_model.parameters())
    optimizer.step()
    model.update_ema()
    assert any(not torch.equal(before, after) for before, after in zip(previous, model.target_model.parameters()))
    model.train()
    assert not model.target_model.training

    student = GenerativeModel(
        model_type="consistency",
        consistency_mode="distillation",
        mid_features=(8,),
        embed_dim=8,
        num_steps=16,
        sigma_max=5.0,
    )
    with pytest.raises(ValueError, match="teacher"):
        student.loss(x)
    teacher = GenerativeModel(model_type="edm", mid_features=(8,), embed_dim=8, num_steps=16, sigma_max=5.0)
    student.set_teacher(teacher)
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    student.loss(x).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(teacher_before[key], value) for key, value in teacher.state_dict().items())
    assert not any(key.startswith("teacher.") for key in student.state_dict())


@pytest.mark.parametrize("scheduler", ["euler", "heun", "euler_ancestral", "dpmpp_2m"])
def test_edm_sampler_oracle(scheduler):
    model = GenerativeModel(model_type="edm", mid_features=(8,), embed_dim=8, num_steps=16)
    model.denoise_sigma = lambda x, sigma, cids=None: torch.full_like(x, 0.3)
    x = model.generate((2,), num_samples=3, scheduler=scheduler, num_steps=5)
    torch.testing.assert_close(x, torch.full_like(x, 0.3))


def test_invalid_combinations_fail():
    for kwargs in (
        {"model_type": "rectified_flow", "prediction_type": "v"},
        {"model_type": "flow_matching", "sampling_scheduler": "pndm"},
        {"model_type": "consistency", "trajectory": "vp"},
        {"model_type": "diffusion", "prediction_type": "velocity"},
        {"model_type": "score", "prediction_type": "epsilon"},
    ):
        with pytest.raises(ValueError):
            GenerativeModel(**kwargs)
    model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=16)
    with pytest.raises(ValueError, match="protected"):
        model.generate((2,), scheduler_kwargs={"prediction_type": "sample"})
    with pytest.raises(ValueError, match="eta"):
        model.generate((2,), scheduler="euler", scheduler_kwargs={"eta": 0.5})


def test_all_config_overlays_construct():
    root = Path(__file__).resolve().parents[1]
    for base in ("swissroll", "mnist_uncond", "mnist_cond"):
        for overlay in (root / "config/alternatives").glob("*.yaml"):
            config = read_config([root / f"config/{base}.yaml", overlay], ["model.init_args.lr=1e-3"])
            assert isinstance(config["model"]["init_args"]["lr"], float)
            instantiate(config["model"])


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    def model():
        return GenerativeModel(model_type="consistency", mid_features=(8,), embed_dim=8, num_steps=16, sigma_max=5.0)

    def data():
        return SwissRollDataModule(16, 8, batch_size=4, random_state=42)

    def trainer(name, epochs):
        return Trainer(max_epochs=epochs, accelerator="cpu", save_dir=tmp_path, name=name, tensorboard=False)

    seed_everything(123)
    full = model()
    full_dir = trainer("full", 2).fit(full, data())
    seed_everything(123)
    split = model()
    first_dir = trainer("split", 1).fit(split, data())
    path = first_dir / "checkpoints/last.ckpt"
    loaded = load_model(path)
    trainer("split", 2).fit(loaded, data(), ckpt_path=path)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    full_metrics = [json.loads(line) for line in (full_dir / "metrics.jsonl").read_text().splitlines()]
    split_metrics = [json.loads(line) for line in (first_dir / "metrics.jsonl").read_text().splitlines()]
    assert full_metrics == split_metrics


def test_lightning_not_required():
    # Installation of optional software by another user must not fail a test.
    # Blocking imports verifies independence even if Lightning is present.
    import subprocess
    import sys

    script = """
import sys
class BlockLightning:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('lightning', 'pytorch_lightning'):
            raise ImportError('Lightning is blocked')
sys.meta_path.insert(0, BlockLightning())
from diffusion import GenerativeModel
from diffusion.training import Trainer
model = GenerativeModel(mid_features=(8,), embed_dim=8, num_steps=16)
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


def test_legacy_ddpm_classes_train_and_load(tmp_path):
    for model, shape in (
        (DDPMTab(mid_features=(8,), embed_dim=8, num_steps=16), (2,)),
        (DDPM2d(mid_channels=(4, 8), embed_dim=8, num_steps=16, num_resblocks=1), (1, 8, 8)),
    ):
        loss = model.loss(torch.randn(2, *shape))
        loss.backward()
        path = tmp_path / f"{type(model).__name__}.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": model.hparams}, path)
        restored = type(model).load_from_checkpoint(path)
        assert restored.generate(shape, num_samples=2, scheduler="ddim", num_steps=4).shape == (2, *shape)


@pytest.mark.parametrize("path", ["vp_sde", "subvp", "ve"])
def test_reverse_sde_reproducibility(path):
    model = GenerativeModel(
        model_type="score",
        trajectory=path,
        mid_features=(8,),
        embed_dim=8,
        num_steps=16,
        sigma_max=3.0,
        sampling_scheduler="euler_maruyama",
    )
    outputs = [
        model.generate((2,), num_samples=3, num_steps=8, generator=torch.Generator().manual_seed(9)) for _ in range(2)
    ]
    assert outputs[0].isfinite().all()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["euler", "midpoint", "heun", "rk4"])
def test_rectified_flow_constant_field_sample(method):
    model = GenerativeModel(model_type="rectified_flow", mid_features=(8,), embed_dim=8, num_steps=16)

    class Field(nn.Module):
        def forward(self, x, t, cids=None):
            return torch.full_like(x, 2.0)

    model.set_model(Field())
    initial = torch.randn(3, 2, generator=torch.Generator().manual_seed(4))
    output = model.generate(
        (2,),
        num_samples=3,
        scheduler=method,
        num_steps=4,
        generator=torch.Generator().manual_seed(4),
        return_trajectory=True,
    )
    assert output.shape == (5, 3, 2)
    torch.testing.assert_close(output[0], initial)
    torch.testing.assert_close(output[-1], initial - 2)


def test_minibatch_ot_uses_optimal_pairing():
    model = GenerativeModel(model_type="flow_matching", coupling="ot", mid_features=(8,), embed_dim=8, num_steps=16)
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

    # The matching noise is in the opposite order. Optimal assignment aligns
    # identical endpoints, so the target velocity is exactly zero.
    class Zero(nn.Module):
        def forward(self, x, t, cids=None):
            return torch.zeros_like(x)

    model.set_model(Zero())
    assert model.loss(x, noise=x.flip(0), times=torch.tensor([0.2, 0.7])).item() == 0


def test_vp_denoise_sigma_teacher_adapter():
    model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=40, schedule="linear", prediction_type="x0")
    model.set_model(Oracle(model, "x0"))
    sigmas = ((1 - model.alphas_bar[[4, 18, 30]]) / model.alphas_bar[[4, 18, 30]]).sqrt()
    x = torch.randn(3, 2)
    torch.testing.assert_close(model.denoise_sigma(x, sigmas), torch.full_like(x, 0.3))
    with pytest.raises(ValueError, match="sigma range"):
        model.denoise_sigma(x, 10000.0)


@pytest.mark.parametrize("scheduler", ["euler", "heun", "lms", "dpmpp_2m", "dpmpp_3m", "unipc", "deis"])
def test_karras_grid_sampling(scheduler):
    model = DDPMTab(mid_features=(8,), embed_dim=8, num_steps=32, schedule="linear", prediction_type="v")
    output = model.generate(
        (2,), num_samples=2, scheduler=scheduler, num_steps=8, scheduler_kwargs={"use_karras_sigmas": True}
    )
    assert output.shape == (2, 2) and output.isfinite().all()
