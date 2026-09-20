"""Self-contained safetensors model exports and compatible checkpoint loading."""

import inspect
import json
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _model_classes():
    # Import lazily so model class methods can delegate here without a cycle.
    from .ddpm import DDPM2d, DDPMTab
    from .distillation import DistillationModel
    from .generative import GenerativeModel

    return {cls.__name__: cls for cls in (DDPMTab, DDPM2d, GenerativeModel, DistillationModel)}


def save_model(model, path):
    """Atomically export weights, buffers and constructor metadata to safetensors.

    Includes consistency EMA and distillation critics, but no teacher weights,
    optimizer or RNG state. Use the trainer's .ckpt files for exact resume.
    """
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Model exports must use a .safetensors filename")
    if type(model) not in _model_classes().values():
        raise ValueError("Model export requires DDPMTab, DDPM2d, GenerativeModel or DistillationModel")
    try:
        parameters = json.dumps(model.hparams, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Safetensors export requires JSON-serializable constructor arguments") from error
    metadata = {
        "format": "pt",
        "gen_kit_format": "1",
        "model_class": type(model).__name__,
        "hyper_parameters": parameters,
    }
    # Independent contiguous copies also support aliased/noncontiguous buffers
    # without changing the live model's device, strides, mode or gradients.
    tensors = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in model.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_model(path, map_location):
    if Path(path).suffix.lower() != ".safetensors":
        return torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(map_location, (str, torch.device)):
        raise ValueError("Safetensors map_location must be a device string or torch.device")
    with safe_open(path, framework="pt", device=str(map_location)) as file:
        metadata = file.metadata() or {}
        if metadata.get("gen_kit_format") != "1":
            raise ValueError("Missing or unsupported gen-kit safetensors metadata; export with save_model()")
        try:
            parameters = json.loads(metadata["hyper_parameters"])
            name = metadata["model_class"]
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("Invalid safetensors model metadata") from error
        if not isinstance(parameters, dict) or name not in _model_classes():
            raise ValueError("Invalid safetensors model class or constructor metadata")
        return {
            "model_class": name,
            "hyper_parameters": parameters,
            "state_dict": {key: file.get_tensor(key) for key in file.keys()},
        }


def load_model(path, map_location="cpu", *, model_class=None, **overrides):
    """Reconstruct a model from .safetensors or a native/legacy .ckpt file.

    Model class names are resolved through a fixed registry, never imported
    from file metadata. Constructor overrides match load_from_checkpoint().
    """
    checkpoint = _read_model(path, map_location)
    parameters = dict(checkpoint["hyper_parameters"])
    name = checkpoint.get("model_class", "").split(".")[-1]
    if model_class is not None:
        if name and model_class.__name__ != name:
            raise ValueError(f"Checkpoint contains {name}, not {model_class.__name__}")
        cls = model_class
    else:
        if not name:
            name = (
                "DistillationModel"
                if "method" in parameters
                else "GenerativeModel"
                if "model_type" in parameters
                else "DDPMTab"
                if "in_features" in parameters
                else "DDPM2d"
                if "in_channels" in parameters
                else ""
            )
        classes = _model_classes()
        if name not in classes:
            raise ValueError("Unrecognized model_class; load legacy checkpoints with DDPMTab/DDPM2d directly")
        cls = classes[name]
    # Legacy checkpoints may contain the derived beta table as well as its
    # constructor settings. The registered state still restores the exact table.
    if "betas" not in inspect.signature(cls.__init__).parameters:
        parameters.pop("betas", None)
    parameters.update(overrides)
    model = cls(**parameters)
    # Keep each saved tensor's dtype/device, including integer/bool counters,
    # while preserving requires_grad flags from the model constructor.
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    return model
