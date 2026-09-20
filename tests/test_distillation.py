"""Distribution-gradient, teacher, optimizer and few-step distillation contracts."""

import json

import pytest
import torch

from diffusion import DistillationModel, GenerativeModel
from diffusion.data import SwissRollDataModule
from diffusion.distillation import distribution_matching_loss
from diffusion.paths import prediction_to_x0_eps
from diffusion.training import Trainer, load_model, seed_everything


def small_options(image=False):
    return dict(
        backbone="unet" if image else "dense",
        mid_features=(8,),
        mid_channels=(4, 8),
        embed_dim=8,
        num_steps=32,
        num_resblocks=1,
        schedule="linear",
        num_classes=10 if image else None,
    )


def test_dmd_gradient_direction_and_stop_gradient():
    generated = torch.tensor([[2.0, 4.0]], requires_grad=True)
    real = torch.tensor([[1.0, 2.0]], requires_grad=True)
    fake = torch.tensor([[3.0, 5.0]], requires_grad=True)
    loss = distribution_matching_loss(generated, real, fake)
    loss.backward()
    expected = (fake.detach() - real.detach()) / 1.5 / generated.numel()
    torch.testing.assert_close(generated.grad, expected)
    assert real.grad is None and fake.grad is None
    zero = distribution_matching_loss(generated, real, real)
    assert zero.item() == 0


@pytest.mark.parametrize("method", ["dmd", "dmd2", "progressive", "trajectory", "reflow"])
@pytest.mark.parametrize("image", [False, True])
def test_distillation_training_sampling_and_checkpoint(method, image, tmp_path):
    options = small_options(image)
    teacher = (
        GenerativeModel(model_type="rectified_flow", **options)
        if method == "reflow"
        else GenerativeModel(prediction_type="v", **options)
    )
    student = DistillationModel(
        method=method,
        teacher_steps=4,
        student_steps=2 if method in ("dmd2", "progressive") else 1,
        critic_updates=1,
        **options,
    )
    student.set_teacher(teacher)
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    optimizer = student.configure_optimizers()[0][0]
    auxiliary = student.configure_auxiliary_optimizers()
    x = torch.randn(2, 1, 8, 8) if image else torch.randn(4, 2)
    labels = torch.tensor([1, 3]) if image else None
    batch = (x, labels) if image else x
    loss, metrics, updated = student.optimize_batch(batch, 0, optimizer, auxiliary, 1.0)
    assert loss.isfinite() and updated
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_before.items())
    assert all(p.grad is None for p in teacher.parameters())
    if method == "dmd":
        assert metrics["regression_loss"] >= 0 and "fake_score_loss" in metrics
    if method == "dmd2":
        assert "generator_gan_loss" in metrics and "critic_gan_loss" in metrics and "regression_loss" not in metrics
    path = tmp_path / "student.ckpt"
    torch.save(
        {"model_class": "DistillationModel", "hyper_parameters": student.hparams, "state_dict": student.state_dict()},
        path,
    )
    restored = load_model(path)
    assert restored.teacher is None
    result = restored.generate(x.shape[1:], num_samples=len(x), cids=labels, return_trajectory=True)
    assert result.shape == (student.student_steps + 1, *x.shape) and result.isfinite().all()


def test_dmd2_two_timescales_and_generator_gradient_isolation():
    options = small_options()
    model = DistillationModel(method="dmd2", teacher_steps=4, student_steps=2, critic_updates=3, **options)
    model.set_teacher(GenerativeModel(prediction_type="v", **options))
    optimizer = model.configure_optimizers()[0][0]
    auxiliary = model.configure_auxiliary_optimizers()
    generator_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    critic_ids = {id(p) for group in auxiliary["critic"].param_groups for p in group["params"]}
    assert not generator_ids & critic_ids
    before = [p.clone() for p in model.eps_model.parameters()]
    for index in range(3):
        _, _, updated = model.optimize_batch(torch.randn(4, 2), index, optimizer, auxiliary)
        assert updated == (index == 2)
        if not updated:
            assert all(torch.equal(old, new) for old, new in zip(before, model.eps_model.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(before, model.eps_model.parameters()))
    model.zero_grad(set_to_none=True)
    model.generator_loss(torch.randn(4, 2))[0].backward()
    assert any(p.grad is not None for p in model.eps_model.parameters())
    assert all(p.grad is None for p in model.fake_score.parameters())
    assert all(p.grad is None for p in model.discriminator_head.parameters())


def test_progressive_target_reproduces_two_teacher_steps():
    options = small_options()
    teacher = GenerativeModel(prediction_type="v", **options)
    model = DistillationModel(method="progressive", student_steps=4, teacher_steps=8, **options)
    model.set_teacher(teacher)
    x = torch.randn(4, 2)
    xt, t, target, teacher_final, end = model.progressive_target(x, torch.randn_like(x), torch.arange(4))
    a, s = model._coefficients(t, x)
    x0, eps = prediction_to_x0_eps("v", target, xt, a, s)
    a_end, s_end = model._coefficients(end, x)
    torch.testing.assert_close(a_end * x0 + s_end * eps, teacher_final, atol=1e-6, rtol=1e-5)
    assert not target.requires_grad


def test_dmd2_resume_preserves_critic_optimizer_and_update_phase(tmp_path):
    options = small_options()
    teacher = GenerativeModel(prediction_type="v", **options)
    teacher_path = tmp_path / "teacher.ckpt"
    torch.save(
        {"model_class": "GenerativeModel", "hyper_parameters": teacher.hparams, "state_dict": teacher.state_dict()},
        teacher_path,
    )

    def model():
        return DistillationModel(
            method="dmd2",
            student_steps=2,
            teacher_steps=4,
            critic_updates=3,
            teacher_checkpoint=str(teacher_path),
            **options,
        )

    def trainer(name, epochs):
        return Trainer(max_epochs=epochs, accelerator="cpu", save_dir=tmp_path, name=name, tensorboard=False)

    def data():
        return SwissRollDataModule(16, 8, batch_size=4, random_state=42)

    seed_everything(99)
    full = model()
    full_dir = trainer("full", 2).fit(full, data())
    seed_everything(99)
    split = model()
    split_dir = trainer("split", 1).fit(split, data())
    checkpoint = split_dir / "checkpoints/last.ckpt"
    restored = load_model(checkpoint)
    trainer("split", 2).fit(restored, data(), ckpt_path=checkpoint)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["auxiliary_optimizers"]["critic"]["state"]
    assert int(restored.distill_updates) == 8
    a = [json.loads(line) for line in (full_dir / "metrics.jsonl").read_text().splitlines()]
    b = [json.loads(line) for line in (split_dir / "metrics.jsonl").read_text().splitlines()]
    assert a == b


def test_distilled_sampler_rejects_untrained_step_counts():
    model = DistillationModel(method="dmd2", student_steps=4, teacher_steps=8, **small_options())
    with pytest.raises(ValueError, match="trained for"):
        model.generate((2,), num_steps=1)
    with pytest.raises(ValueError, match="trained distilled"):
        model.generate((2,), scheduler="unipc")
    with pytest.raises(ValueError, match=r"2 \* student_steps"):
        DistillationModel(method="progressive", student_steps=3, teacher_steps=8)


def test_teacher_follows_student_device_and_dtype():
    options = small_options()
    model = DistillationModel(method="dmd2", student_steps=2, **options)
    teacher = GenerativeModel(prediction_type="v", **options)
    model.set_teacher(teacher)
    model.double()
    loss = model.loss(torch.randn(4, 2, dtype=torch.float64))
    assert teacher.dtype == torch.float64 and loss.isfinite()


@pytest.mark.parametrize("method", ["dmd", "dmd2", "trajectory"])
def test_incompatible_distilled_teacher_is_rejected(method):
    options = small_options()
    model = DistillationModel(method=method, **options)
    teacher = DistillationModel(method="dmd2", **options)
    with pytest.raises(ValueError, match="undistilled|previous progressive"):
        model.set_teacher(teacher)


def test_consistency_teacher_requires_matching_input_type():
    student = GenerativeModel(model_type="consistency", consistency_mode="distillation", **small_options())
    teacher = GenerativeModel(model_type="edm", **{**small_options(True), "num_classes": None})
    with pytest.raises(ValueError, match="input types"):
        student.set_teacher(teacher)
