"""Native PyTorch training, YAML configuration, logging and resumable checkpoints."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from .data import MNISTDataModule, SwissRollDataModule
from .ddpm import DDPM2d, DDPMTab
from .generative import GenerativeModel
from .distillation import DistillationModel
from .serialization import load_model, save_model


CLASSES = {
    cls.__name__: cls
    for cls in (MNISTDataModule, SwissRollDataModule, DDPM2d, DDPMTab, GenerativeModel, DistillationModel)
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def instantiate(spec):
    name = spec["class_path"].split(".")[-1]
    if name not in CLASSES:
        raise ValueError(f"Unknown class {spec['class_path']!r}; choose from {tuple(CLASSES)}")
    return CLASSES[name](**spec.get("init_args", {}))


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return type(batch)(to_device(value, device) for value in batch)


def batch_size(batch):
    if isinstance(batch, dict):
        return batch["features"].shape[0]
    return batch.shape[0] if isinstance(batch, torch.Tensor) else batch[0].shape[0]


def choose_device(accelerator):
    if accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device("cuda" if accelerator == "gpu" else accelerator)


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, values, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), pos, gaussian, cached))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


class Trainer:
    """Single-device training with an EMA hook after each optimizer step."""

    def __init__(
        self,
        max_epochs=50,
        accelerator="auto",
        save_dir="run",
        name="experiment",
        version=None,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        checkpoint_every_n_epochs=10,
        gradient_clip_val=0.0,
        limit_train_batches=None,
        limit_val_batches=None,
        tensorboard=True,
        patience=0,
        num_threads=None,
        export_safetensors=False,
    ):
        if min(max_epochs, check_val_every_n_epoch, checkpoint_every_n_epochs, log_every_n_steps) < 1:
            raise ValueError("Epoch counts, logging and checkpoint intervals must be positive")
        for limit in (limit_train_batches, limit_val_batches):
            if limit is not None and (not isinstance(limit, int) or limit < 1):
                raise ValueError("Batch limits must be positive integers or null")
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.max_epochs, self.device = max_epochs, choose_device(accelerator)
        self.save_dir, self.name, self.version = Path(save_dir), name, version
        self.log_every_n_steps, self.check_val_every_n_epoch = log_every_n_steps, check_val_every_n_epoch
        self.checkpoint_every_n_epochs, self.gradient_clip_val = checkpoint_every_n_epochs, gradient_clip_val
        self.limit_train_batches, self.limit_val_batches = limit_train_batches, limit_val_batches
        self.tensorboard, self.patience = tensorboard, patience
        self.export_safetensors = export_safetensors
        self.global_step, self.best_loss, self.bad_epochs = 0, math.inf, 0

    def _run_dir(self, checkpoint):
        if checkpoint:
            return Path(checkpoint).resolve().parent.parent
        root = self.save_dir / self.name
        if self.version is not None:
            return root / str(self.version)
        index = 0
        while (root / f"version_{index}").exists():
            index += 1
        return root / f"version_{index}"

    @torch.no_grad()
    def validate(self, model, loader):
        model.eval()
        total, count = 0.0, 0
        for index, batch in enumerate(loader):
            if self.limit_val_batches is not None and index >= self.limit_val_batches:
                break
            batch = to_device(batch, self.device)
            loss = model.validation_step(batch, index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            size = batch_size(batch)
            total, count = total + loss.item() * size, count + size
        return total / count if count else None

    def _save(self, path, model, optimizer, schedule, epoch, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "model_class": type(model).__name__,
            "hyper_parameters": model.hparams,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None if schedule is None else schedule.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "bad_epochs": self.bad_epochs,
            "rng_state": rng_state(),
            "config": config,
            "auxiliary_optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.auxiliary_optimizers.items()
            },
        }
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        if self.export_safetensors:
            save_model(model, path.with_suffix(".safetensors"))

    def fit(self, model, data, ckpt_path=None, config=None):
        if ckpt_path and Path(ckpt_path).suffix.lower() == ".safetensors":
            raise ValueError("Safetensors exports have no optimizer/RNG state; use a .ckpt file for training resume")
        data.prepare_data()
        data.setup("fit")
        train_loader, val_loader = data.train_dataloader(), data.val_dataloader()
        train_batches = min(len(train_loader), self.limit_train_batches or len(train_loader))
        if train_batches == 0:
            raise ValueError("Empty training loader; increase data size or reduce batch_size")
        model.to(self.device)
        if hasattr(model, "prepare_training"):
            model.prepare_training()
        configured = model.configure_optimizers(self.max_epochs, self.max_epochs * train_batches)
        self.auxiliary_optimizers = (
            model.configure_auxiliary_optimizers() if hasattr(model, "configure_auxiliary_optimizers") else {}
        )
        optimizer, schedule, interval = configured, None, None
        if isinstance(configured, tuple):
            optimizer = configured[0][0]
            schedule, interval = configured[1][0]["scheduler"], configured[1][0]["interval"]
        start_epoch = 0
        if ckpt_path:
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if saved["hyper_parameters"] != model.hparams:
                raise ValueError("Resume requires the same model configuration as the checkpoint")
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            auxiliary_states = saved.get("auxiliary_optimizers", {})
            if set(auxiliary_states) != set(self.auxiliary_optimizers):
                raise ValueError("Checkpoint auxiliary optimizers do not match the model")
            for name, state in auxiliary_states.items():
                self.auxiliary_optimizers[name].load_state_dict(state)
            if schedule is not None and saved["lr_scheduler"] is not None:
                schedule.load_state_dict(saved["lr_scheduler"])
            start_epoch = saved["epoch"] + 1
            self.global_step, self.best_loss = saved["global_step"], saved["best_loss"]
            self.bad_epochs = saved.get("bad_epochs", 0)
            restore_rng(saved["rng_state"])
        self.run_dir = self._run_dir(ckpt_path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (self.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        writer = None
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(self.run_dir))
        print(f"Training {type(model).__name__} on {self.device}; output: {self.run_dir}", flush=True)
        try:
            for epoch in range(start_epoch, self.max_epochs):
                model.train()
                total, count = 0.0, 0
                component_totals, component_counts = {}, {}
                generator_updates = 0
                for index, batch in enumerate(train_loader):
                    if index >= train_batches:
                        break
                    batch = to_device(batch, self.device)
                    if hasattr(model, "optimize_batch"):
                        loss, components, updated = model.optimize_batch(
                            batch,
                            index,
                            optimizer,
                            self.auxiliary_optimizers,
                            self.gradient_clip_val,
                        )
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss = model.training_step(batch, index)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss at step {self.global_step}")
                        loss.backward()
                        if self.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                self.gradient_clip_val,
                                error_if_nonfinite=True,
                            )
                        optimizer.step()
                        components, updated = {}, True
                    generator_updates += int(updated)
                    for key, value in components.items():
                        component_totals[key] = component_totals.get(key, 0.0) + value
                        component_counts[key] = component_counts.get(key, 0) + 1
                    if updated and hasattr(model, "update_ema"):
                        model.update_ema()
                    if updated and schedule is not None and interval == "step":
                        schedule.step()
                    self.global_step += 1
                    size = batch_size(batch)
                    total, count = total + loss.item() * size, count + size
                    if writer is not None and self.global_step % self.log_every_n_steps == 0:
                        writer.add_scalar("train_loss", loss.item(), self.global_step)
                        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], self.global_step)
                        for key, value in components.items():
                            writer.add_scalar(key, value, self.global_step)
                val_loss = self.validate(model, val_loader) if (epoch + 1) % self.check_val_every_n_epoch == 0 else None
                if generator_updates and schedule is not None and interval == "epoch":
                    schedule.step()
                metrics = {"epoch": epoch, "step": self.global_step, "train_loss": total / count, "val_loss": val_loss}
                metrics.update({key: value / component_counts[key] for key, value in component_totals.items()})
                with (self.run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics) + "\n")
                if writer is not None and val_loss is not None:
                    writer.add_scalar("val_loss", val_loss, self.global_step)
                print(json.dumps(metrics), flush=True)
                monitored = val_loss if len(val_loader) else total / count
                improved = monitored is not None and monitored < self.best_loss
                if improved:
                    self.best_loss, self.bad_epochs = monitored, 0
                elif monitored is not None:
                    self.bad_epochs += 1
                checkpoint_dir = self.run_dir / "checkpoints"
                self._save(checkpoint_dir / "last.ckpt", model, optimizer, schedule, epoch, config)
                if improved:
                    self._save(checkpoint_dir / "best.ckpt", model, optimizer, schedule, epoch, config)
                if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                    self._save(checkpoint_dir / f"epoch={epoch}.ckpt", model, optimizer, schedule, epoch, config)
                if self.patience > 0 and self.bad_epochs >= self.patience:
                    break
        finally:
            if writer is not None:
                writer.close()
        return self.run_dir


def read_config(paths, overrides=()):
    def merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    config = {}
    for path in paths:
        merge(config, OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    for override in overrides:
        if "=" not in override:
            raise ValueError("Overrides must use --set section.key=value")
        path, value = override.split("=", 1)
        keys, target = path.split("."), config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = OmegaConf.to_container(OmegaConf.from_dotlist([f"value={value}"]))["value"]
    return config


def main(argv=None, default_config=None):
    parser = argparse.ArgumentParser(description="Train diffusion, flow and consistency models with plain PyTorch")
    parser.add_argument("command", nargs="?", choices=("fit", "validate", "sample", "export"), default="fit")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-shape", type=int, nargs="+")
    parser.add_argument("--scheduler")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "export":
        if not args.checkpoint:
            parser.error("export requires --checkpoint")
        output = save_model(load_model(args.checkpoint), args.output or "run/model.safetensors")
        print(f"Exported model to {output}")
        return
    if args.command == "sample":
        if not args.checkpoint:
            parser.error("sample requires --checkpoint")
        seed_everything(args.seed)
        model = load_model(args.checkpoint).eval()
        shape = args.sample_shape or (
            (model.hparams.get("in_channels", 1), 28, 28)
            if model.hparams.get("backbone") == "unet" or isinstance(model, DDPM2d)
            else (model.hparams.get("in_features", 2),)
        )
        cids = None if args.class_id is None else torch.full((args.num_samples,), args.class_id, dtype=torch.long)
        samples = model.generate(shape, cids, args.num_samples, scheduler=args.scheduler, num_steps=args.num_steps)
        path = Path(args.output or "run/samples.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples.cpu(), path)
        print(f"Saved {list(samples.shape)} samples to {path}")
        return
    paths = ([default_config] if default_config else []) + args.config
    if not paths:
        parser.error("fit/validate requires --config")
    config = read_config(paths, args.set)
    seed_everything(config.get("seed_everything", 42))
    model, data = instantiate(config["model"]), instantiate(config["data"])
    trainer = Trainer(**config.get("trainer", {}))
    checkpoint = args.checkpoint or config.get("ckpt_path")
    if args.command == "fit":
        trainer.fit(model, data, checkpoint, config)
    else:
        if checkpoint:
            model = load_model(checkpoint)
        model.to(trainer.device)
        data.prepare_data()
        data.setup("validate")
        print({"val_loss": trainer.validate(model, data.val_dataloader())})
