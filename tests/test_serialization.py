"""Safetensors reconstruction, state fidelity and training/CLI integration."""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from diffusion import DDPM2d, DDPMTab, DistillationModel, GenerativeModel, load_model, save_model
from diffusion.data import SwissRollDataModule
from diffusion import serialization
from diffusion.training import Trainer, main


def make_model(kind):
    options = dict(mid_features=(8,), embed_dim=8, num_steps=16, schedule="linear")
    if kind == "tab":
        return DDPMTab(**options)
    if kind == "image":
        return DDPM2d(mid_channels=(4, 8), num_classes=10, embed_dim=8, num_steps=16, num_resblocks=1)
    if kind in ("dmd", "dmd2", "progressive", "trajectory", "reflow"):
        return DistillationModel(
            method=kind,
            student_steps=2 if kind in ("dmd2", "progressive") else 1,
            teacher_steps=4,
            teacher_checkpoint="missing-teacher.ckpt",
            **options,
        )
    return GenerativeModel(model_type=kind, sigma_max=3.0, **options)


@pytest.mark.parametrize(
    "kind",
    [
        "tab",
        "image",
        "diffusion",
        "score",
        "rectified_flow",
        "edm",
        "consistency",
        "dmd",
        "dmd2",
        "progressive",
        "trajectory",
        "reflow",
    ],
)
def test_model_round_trip(kind, tmp_path):
    original = make_model(kind)
    path = tmp_path / "nested" / "model.safetensors"
    original.train()
    assert original.save_safetensors(path) == path
    assert original.training
    restored = load_model(path, map_location=torch.device("cpu"))
    assert type(restored) is type(original)
    assert json.loads(json.dumps(restored.hparams)) == json.loads(json.dumps(original.hparams))
    assert original.state_dict().keys() == restored.state_dict().keys()
    for key, value in original.state_dict().items():
        actual = restored.state_dict()[key]
        assert actual.dtype == value.dtype
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert [p.requires_grad for p in original.parameters()] == [p.requires_grad for p in restored.parameters()]
    assert getattr(restored, "teacher", None) is None
    if kind == "consistency":
        assert not restored.target_model.training
    shape, labels = ((1, 8, 8), torch.tensor([1, 4])) if kind == "image" else ((2,), None)
    kwargs = {} if isinstance(original, DistillationModel) else {"num_steps": 2}
    expected = original.generate(shape, labels, 2, generator=torch.Generator().manual_seed(77), **kwargs)
    actual = restored.generate(shape, labels, 2, generator=torch.Generator().manual_seed(77), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.isfinite().all()


def test_safetensors_preserves_dtype_and_never_uses_pickle(tmp_path, monkeypatch):
    model = make_model("dmd2").double()
    model.distill_updates.fill_(7)
    model.teacher_initialized.fill_(True)
    path = tmp_path / "model.safetensors"

    def reject_pickle(*args, **kwargs):
        pytest.fail("Safetensors must not use torch.save or torch.load")

    monkeypatch.setattr(torch, "save", reject_pickle)
    monkeypatch.setattr(torch, "load", reject_pickle)
    save_model(model, path)
    with safe_open(path, framework="pt") as file:
        assert file.metadata()["model_class"] == "DistillationModel"
        assert file.metadata()["format"] == "pt"
    restored = DistillationModel.load_from_checkpoint(path)
    assert restored.dtype == torch.float64
    assert restored.distill_updates.dtype == torch.int64 and int(restored.distill_updates) == 7
    assert restored.teacher_initialized.dtype == torch.bool and restored.teacher_initialized
    assert restored.generate((2,), num_samples=2).dtype == torch.float64
    with pytest.raises(ValueError, match="contains DistillationModel"):
        GenerativeModel.load_from_checkpoint(path)


def test_noncontiguous_and_shared_state_tensors(tmp_path):
    model = make_model("diffusion")
    # Use a strided weight without changing its shape or values.
    weight = next(p for p in model.parameters() if p.ndim == 2 and min(p.shape) > 1)
    weight.data = weight.data.T.contiguous().T
    assert not weight.is_contiguous()
    # Frequencies can be shared between embeddings of identical width.
    embeddings = [module for module in model.modules() if hasattr(module, "omega")]
    embeddings[1].omega = embeddings[0].omega
    path = save_model(model, tmp_path / "shared.safetensors")
    restored = load_model(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert not weight.is_contiguous()  # Export does not mutate the live layout.


@pytest.mark.parametrize("kind", ["consistency", "dmd2"])
def test_safetensors_teacher_and_student_are_self_contained(kind, tmp_path):
    teacher = make_model("edm" if kind == "consistency" else "diffusion")
    teacher_path = save_model(teacher, tmp_path / "teacher.safetensors")
    options = dict(
        mid_features=(8,), embed_dim=8, num_steps=16, schedule="linear", teacher_checkpoint=str(teacher_path)
    )
    if kind == "consistency":
        student = GenerativeModel(model_type=kind, consistency_mode="distillation", sigma_max=3.0, **options)
    else:
        student = DistillationModel(method=kind, prediction_type="epsilon", teacher_steps=4, **options)
    student.loss(torch.randn(4, 2)).backward()
    assert student.teacher is not None and all(p.grad is None for p in student.teacher.parameters())
    path = save_model(student, tmp_path / "student.safetensors")
    assert not any(key.startswith("teacher.") for key in load_file(path))
    teacher_path.unlink()
    restored = load_model(path)
    assert restored.teacher is None
    assert restored.generate((2,), num_samples=2).isfinite().all()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"gen_kit_format": "2"},
        {"gen_kit_format": "1"},
        {"gen_kit_format": "1", "model_class": "GenerativeModel", "hyper_parameters": "not json"},
        {"gen_kit_format": "1", "model_class": "GenerativeModel", "hyper_parameters": "[]"},
        {"gen_kit_format": "1", "model_class": "arbitrary.module.Class", "hyper_parameters": "{}"},
    ],
)
def test_invalid_metadata_is_rejected(metadata, tmp_path):
    path = tmp_path / "invalid.safetensors"
    save_file({"x": torch.ones(1)}, path, metadata=metadata)
    with pytest.raises(ValueError, match="metadata"):
        load_model(path)


def test_invalid_tensor_state_is_rejected(tmp_path):
    path = save_model(make_model("diffusion"), tmp_path / "model.safetensors")
    with safe_open(path, framework="pt") as file:
        metadata = file.metadata()
    save_file({"unexpected": torch.zeros(1)}, path, metadata=metadata)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_model(path)


def test_failed_export_preserves_existing_file(tmp_path, monkeypatch):
    model = make_model("diffusion")
    path = save_model(model, tmp_path / "model.safetensors")
    previous = path.read_bytes()

    def fail(tensors, filename, metadata):
        Path(filename).write_bytes(b"partial write")
        raise OSError("simulated write failure")

    monkeypatch.setattr(serialization, "save_file", fail)
    with pytest.raises(OSError, match="simulated"):
        save_model(model, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_trainer_exports_alongside_resumable_checkpoints(tmp_path):
    model = make_model("consistency")
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        save_dir=tmp_path,
        tensorboard=False,
        checkpoint_every_n_epochs=1,
        export_safetensors=True,
    )
    output = trainer.fit(model, SwissRollDataModule(8, 4, batch_size=4, random_state=42))
    for name in ("best", "last", "epoch=0"):
        checkpoint = output / "checkpoints" / f"{name}.ckpt"
        exported = checkpoint.with_suffix(".safetensors")
        assert torch.load(checkpoint, weights_only=True)["optimizer"]["state"]
        restored = load_model(exported)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="optimizer/RNG"):
        trainer.fit(model, None, ckpt_path=exported)


def test_cli_export_and_sample(tmp_path):
    model = make_model("rectified_flow")
    checkpoint, exported, samples = (tmp_path / name for name in ("model.ckpt", "model.safetensors", "samples.pt"))
    torch.save(
        {"model_class": "GenerativeModel", "hyper_parameters": model.hparams, "state_dict": model.state_dict()},
        checkpoint,
    )
    main(["export", "--checkpoint", str(checkpoint), "--output", str(exported)])
    main(["sample", "--checkpoint", str(exported), "--output", str(samples), "--num-samples", "3", "--num-steps", "2"])
    result = torch.load(samples, weights_only=True)
    assert result.shape == (3, 2) and result.isfinite().all()
