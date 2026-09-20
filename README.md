# Denoising diffusion demo

This repository is a small PyTorch playground for diffusion, flow matching, rectified flow, EDM, consistency models and few-step distillation. Training uses a native PyTorch loop; Lightning is not required. The same dense network (Swiss roll) or U-Net (MNIST) can be trained with different objectives and sampled with different compatible solvers.

## Model and sampler alternatives

| Choice | Available alternatives |
| --- | --- |
| Model family | Diffusion, score matching, flow matching, rectified flow, EDM, consistency training, consistency distillation |
| Training target | Noise `epsilon`, clean data `x0`, diffusion `v`, score, flow `velocity`; EDM and consistency use preconditioned denoising |
| Probability path | Discrete VP, linear, trigonometric/cosine, continuous VP, sub-VP, VE, EDM sigma space |
| VP sampling | DDPM, DDIM, Euler, ancestral Euler, Heun, LMS, PNDM, PLMS, DPM-Solver, DPM++ 2M/3M/2S/SDE, DEIS, UniPC |
| Continuous sampling | Euler, midpoint, Heun, RK4; reverse-SDE Euler–Maruyama on VP-SDE/sub-VP/VE |
| EDM sampling | Euler, Heun, ancestral Euler, DPM++ 2M; Karras, exponential or linear sigma grids |
| Consistency sampling | One-step generation and stochastic multistep refinement |
| Teacher distillation | DMD, DMD2, progressive DDIM distillation, paired trajectory regression, reflow, consistency distillation |

See [the implementation guide](docs/model_choices.md) for equations, compatibility rules, source references and the file map. Diffusion `v` and flow `velocity` are distinct targets. Changing the target or path requires training; compatible inference solvers can be changed on an ordinary diffusion checkpoint. Distilled VP students use their trained step count and sampler.

## Setup and quick start

Python 3.12 or newer is required. From the repository root:

```bash
uv sync --extra dev
source .venv/bin/activate

# Original VP diffusion experiment.
python scripts/main.py fit --config config/swissroll.yaml

# Apply a model alternative after the dataset config.
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/rectified_flow.yaml

python scripts/main.py fit --config config/mnist_cond.yaml \
  --config config/alternatives/v_prediction.yaml

# Small CPU smoke run; does not establish sample quality.
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/flow_matching.yaml \
  --set trainer.accelerator=cpu --set trainer.max_epochs=1 \
  --set trainer.limit_train_batches=2 --set trainer.limit_val_batches=1
```

Dataset configs set the data, backbone and optimizer. Files in `config/alternatives/` override the objective, trajectory, prediction target and default sampler. Multiple `--config` files merge in order; `--set section.key=value` overrides a value. The former per-example scripts now provide dataset shortcuts with this same CLI.

The trainer writes `metrics.jsonl`, TensorBoard events, the resolved config, and `best.ckpt`/`last.ckpt` under `run/<name>/version_<n>/`. It supports CPU, CUDA and MPS, validation, gradient clipping, learning-rate scheduling, early stopping and checkpoint resume. It runs on one device; it does not implement Lightning callbacks or distributed training.

```bash
# Resume with the same model settings; max_epochs is the total desired count.
python scripts/main.py fit --config config/swissroll.yaml \
  --checkpoint run/swissroll/version_0/checkpoints/last.ckpt \
  --set trainer.max_epochs=1200

# Change a VP model's sampler without retraining.
python scripts/main.py sample \
  --checkpoint run/swissroll/version_0/checkpoints/last.ckpt \
  --scheduler unipc --num-steps 30 --num-samples 1000 \
  --output run/swissroll_samples.pt
```

For MNIST, sampling infers `(1, 28, 28)` from the backbone. Add `--class-id 7` for a conditional checkpoint. Set `--sample-shape` explicitly for other image dimensions.

## Consistency training and distillation

Standalone consistency training uses coupled noisy pairs and a frozen EMA target. Distillation instead uses a frozen diffusion teacher's Heun step to construct the pair:

```bash
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/consistency.yaml

# First train an EDM teacher.
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/edm.yaml

# Then distill it into a consistency student.
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/consistency_distillation.yaml \
  --set model.init_args.teacher_checkpoint=run/edm/version_0/checkpoints/best.ckpt
```

A distilled checkpoint is self-contained for sampling. Resume of distillation training still requires its teacher checkpoint. The student and teacher must agree on input shape, conditioning, data scaling and supported noise range.

```python
from diffusion.training import load_model

model = load_model("run/consistency/version_0/checkpoints/last.ckpt").eval()
samples = model.generate((2,), num_samples=1000, num_steps=1)
trajectory = model.generate((2,), num_samples=16, num_steps=4, return_trajectory=True)
```

## One-step and few-step distillation

`DistillationModel` provides the following training methods:

| Preset in `config/alternatives/` | Objective | Default student steps |
| --- | --- | --- |
| `dmd.yaml` | Distribution matching, learned fake score and paired teacher regression | 1 |
| `dmd2.yaml` | Distribution matching, 5 critic updates per generator update, adversarial loss and backward simulation | 4 |
| `progressive_distillation.yaml` | Replace two deterministic teacher DDIM steps with one student step | 8, from a 16-step teacher |
| `trajectory_distillation.yaml` | Regress directly from noise to a teacher trajectory endpoint | 1 |
| `reflow.yaml` | Straighten a linear flow using teacher-generated noise/sample pairs | 1 Euler step |

Train a VP teacher using the same dataset and backbone as the student, then choose a distillation preset:

```bash
python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/v_prediction.yaml

python scripts/main.py fit --config config/swissroll.yaml \
  --config config/alternatives/dmd2.yaml \
  --set model.init_args.teacher_checkpoint=run/v_prediction/version_0/checkpoints/best.ckpt

python scripts/main.py sample \
  --checkpoint run/dmd2/version_0/checkpoints/best.ckpt \
  --num-steps 4 --num-samples 1000 --output run/dmd2_samples.pt
```

Replace `dmd2.yaml` with `dmd.yaml`, `progressive_distillation.yaml` or `trajectory_distillation.yaml` to change the objective. To train one-step DMD2, set `model.init_args.student_steps=1` during training. A progressive stage requires `teacher_steps=2*student_steps`; repeat with the previous student checkpoint as teacher to halve the step count again. Reflow instead takes a trained `rectified_flow` or linear `flow_matching` checkpoint.

DMD/DMD2/progressive/trajectory methods currently require a VP teacher with the same beta schedule. Initialization copies the teacher weights, so the backbone and prediction type must also match; the `v_prediction` and distillation presets match each other. Set `initialize_from_teacher=false` to train a differently sized student from scratch. Teachers are unnecessary for sampling saved students. Training checkpoints include auxiliary optimizers and update counters for exact resume.

These implementations target Swiss roll and MNIST experiments. DMD computes teacher pairs online and uses MSE/MAE instead of LPIPS. DMD2 uses a small discriminator head on shared fake-score features and clean inputs. There are no text/latent models, pretrained downloads or claims of reproducing published image-quality results. Distillation losses are diagnostics, not sample-quality metrics; inspect generated samples as well.

Run the mathematical and integration checks with `python -m pytest -q tests`.

A short theoretical intro to standard DDPMs can be found [here](notebooks/intro_1_ddpm.ipynb). DDIMs for accelerated sampling are discussed in the [companion notebook](notebooks/intro_2_ddim.ipynb). Two example applications establish a small experimentation playground. They are prepared in such a way that they can be easily modified and extended.


## Notebooks

- [Introduction to DDPMs](notebooks/intro_1_ddpm.ipynb)
- [Introduction to DDIMs](notebooks/intro_2_ddim.ipynb)
- [Swiss roll example](notebooks/swissroll.ipynb)
- [Unconditional model on MNIST](notebooks/mnist_uncond.ipynb)
- [Conditional model on MNIST](notebooks/mnist_cond.ipynb)


## Swiss roll

As a first example, a generative DDPM is trained on a 2D Swiss roll distribution. The main training script can be called to that end with a config file that allows one to adjust the problem setup and model definition:
```bash
python scripts/main.py fit --config config/swissroll.yaml
```
After the training has finished, the final model can be tested and analyzed in [this notebook](notebooks/swissroll.ipynb).

For monitoring the experiment, run `tensorboard --logdir run/swissroll/` and open [localhost:6006](http://localhost:6006). Set `trainer.tensorboard=false` to keep only JSONL metrics and checkpoints.

<p>
  <img src="assets/swissroll_forward.jpg" alt="Forward process diffusing data into noise" title="Forward diffusion process" width="700">
</p>

<p>
  <img src="assets/swissroll_reverse.jpg" alt="Reverse process generating data from noise" title="Trained reverse process" width="700">
</p>


## MNIST

The second application is based on the MNIST dataset. Here, one can construct a DDPM that is either unconditioned (generates randomly) or conditioned on the class (generates controllably). Such models generating images of handwritten digits can be learned by running the main script in the following ways:
```bash
python scripts/main.py fit --config config/mnist_uncond.yaml
```
```bash
python scripts/main.py fit --config config/mnist_cond.yaml
```
Two dedicated notebooks [here](notebooks/mnist_uncond.ipynb) and [here](notebooks/mnist_cond.ipynb) are provided in order to test the unconditional and the conditional model after training, respectively.

<p>
  <img src="assets/mnist_forward.svg" alt="Forward process diffusing data into noise" title="Forward diffusion process" width="700">
</p>

<p>
  <img src="assets/mnist_reverse.svg" alt="Reverse process generating data from noise" title="Trained reverse process" width="700">
</p>
