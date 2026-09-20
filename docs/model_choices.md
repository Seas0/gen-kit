# Models, targets, trajectories and samplers

The code separates four choices: **what is learned**, **which probability path is used**, **what the network predicts**, and **how inference integrates the learned model**. It covers continuous Gaussian vector/image models with representative major model and solver families. Discrete/token diffusion, latent autoencoders, text conditioning, classifier-free guidance and learned-variance heads are outside this demo.

## File map

| File | Responsibility |
| --- | --- |
| `src/diffusion/generative.py` | Model selection, continuous objectives, EDM preconditioning, consistency training/distillation |
| `src/diffusion/distillation.py` | DMD/DMD2, progressive and paired distillation, reflow, auxiliary optimizers and student samplers |
| `src/diffusion/paths.py` | Path coefficients, analytic derivatives and invertible target conversions |
| `src/diffusion/sampling.py` | VP scheduler adapters, ODE/SDE integration, EDM and consistency samplers |
| `src/diffusion/training.py` | Native PyTorch training, config merging, logging, saving and resume |
| `src/diffusion/serialization.py` | Atomic safetensors exports, embedded model configuration and checkpoint reconstruction |
| `src/diffusion/ddpm/base.py` | Discrete VP objective, forward noising, DDPM reverse steps and legacy API |
| `src/diffusion/models/` | Time-conditioned MLP and U-Net; U-Net optionally receives class labels |
| `src/diffusion/data/` | Swiss roll and MNIST/FashionMNIST/KMNIST data loaders |
| `config/alternatives/` | Model presets applied after a dataset config |
| `tests/test_generative.py` | Analytic identities, sampler equivalence, gradient and checkpoint tests |
| `tests/test_distillation.py` | Score-difference gradients, frozen teachers, progressive targets, critic isolation and resume |
| `tests/test_serialization.py` | Model/state round trips, dtype preservation, teacher loading, atomic writes and CLI exports |

## Training targets

For Gaussian noise `epsilon`, define `x_t = a(t) x0 + s(t) epsilon`. Throughout this repo, `t=0` is the data end and `t=1` is the noise end. Sampling moves **backward** in time. This reverses the time convention used in some flow matching papers; the velocity sign is adjusted accordingly.

| `prediction_type` | Regression target | Meaning |
| --- | --- | --- |
| `epsilon` / `eps` | `epsilon` | Added Gaussian noise |
| `x0` / `x` / `sample` | `x0` | Clean data |
| `v` / `v_prediction` | `a epsilon - s x0` | Diffusion velocity parameterization |
| `score` | `-epsilon / s` | Conditional score; squared error is weighted by `s²` |
| `velocity` / `flow` | `a' x0 + s' epsilon` | Derivative of the probability path |

These are different training objectives because unweighted error in one parameterization implies time-dependent weighting in another. Conversion at sampling time does not turn a model trained on one loss into a model trained on another loss.

For general paths, converting diffusion `v` uses the divisor `a²+s²`. It equals one only on variance-preserving paths. Flow velocity is a different quantity: on the linear path its target is `epsilon-x0`; on the trigonometric path it is `(pi/2) v`.

All ordinary targets can be converted into estimates of `x0` and `epsilon`. The continuous ODE field is then `a' x0_hat + s' epsilon_hat`. Epsilon/x0/score conversions need nonsingular coefficients; inference avoids singular endpoints and performs a final denoising projection when required. The defaults use `time_eps=1e-4`.

`forward(x, t)` remains a low-level call to the backbone in its own time coordinates. For continuous models use `velocity`, `score`, `predict_x0` or `interpolate` with normalized time; for EDM/consistency use `denoise_sigma` with a positive noise standard deviation. The old `diffuse`/`denoise_step` APIs use discrete zero-based VP indices and reject continuous models.

## Trajectories

| `trajectory` | `a(t)` | `s(t)` | Notes |
| --- | --- | --- | --- |
| `vp` | `sqrt(alpha_bar[index])` | `sqrt(1-alpha_bar[index])` | Discrete beta schedule; original DDPM behavior |
| `linear` | `1-t` | `t` | Straight conditional paths; independent-pair rectified flow / linear flow matching |
| `cosine` | `cos(pi t/2)` | `sin(pi t/2)` | Trigonometric interpolation |
| `vp_sde` | `exp(-B(t)/2)` | `sqrt(1-exp(-B(t)))` | `B(t)=beta_min*t + (beta_max-beta_min)*t²/2` |
| `subvp` | `exp(-B(t)/2)` | `1-exp(-B(t))` | Sub-VP standard deviation, not VP variance |
| `ve` | `1` | `sigma_min*(sigma_max/sigma_min)^t` | Variance-exploding path |
| `edm` | `1` | `sigma` | Training and sampling operate directly in sigma space |

The discrete VP **noise schedule** (`linear`, `quadratic`, `cosine`, `sigmoid`) and the continuous **trajectory** named `cosine` are distinct options. `num_steps` is the VP training schedule length and the continuous backbone's time-embedding scale; `sampling_steps` controls inference resolution independently.

Flow objectives support uniform or logit-normal time draws. `coupling=independent` samples random data/noise pairs. `coupling=ot` solves a minimum-cost assignment within each minibatch. This is minibatch optimal transport, not a guarantee of global dataset-level optimal transport. The rectified-flow preset implements initial straight-path regression. Explicit paired `noise=` in `loss` supports supplied couplings; `DistillationModel(method="reflow")` generates teacher couplings online.

## Model families and compatibility

| `model_type` | Default trajectory / target | Compatible sampling |
| --- | --- | --- |
| `diffusion` | VP / epsilon; also x0 or v | VP scheduler set for `vp`; continuous solvers for continuous paths |
| `score` | VE / score | VP schedulers for `vp`; ODE or reverse-SDE solvers for continuous SDE paths |
| `flow_matching` | Linear / velocity | Euler, midpoint, Heun, RK4; other continuous paths are selectable |
| `rectified_flow` | Linear / velocity | Euler, midpoint, Heun, RK4 |
| `edm` | Sigma space / preconditioned x0 | Euler, Heun, ancestral Euler, DPM++ 2M |
| `consistency` | Sigma space / boundary-conditioned denoiser | One-step or multistep `consistency` sampler |

Invalid family/target/path/sampler combinations raise errors. PNDM and UniPC are VP solvers here; selecting them for a flow model is rejected. A standard diffusion checkpoint cannot be converted into a one-step consistency model merely by choosing a different scheduler.

## Sampling choices

VP samplers use the model's **actual trained betas**, and the adapter converts the model's target into epsilon before invoking Diffusers. It applies scheduler input scaling and the repo's one-based network timestep convention. Every sampling call creates fresh solver state. No pretrained models are downloaded.

DDIM uses the selected grid's actual next timestep in its update, including rounded or nonuniform spacing. A default one-step VP call starts at the terminal training timestep. Explicit `timestep_spacing` overrides still select the requested grid.

The VP names are `ddpm`, `ddim`, `euler`, `euler_ancestral`, `heun`, `lms`, `pndm`, `plms`, `dpm_solver`, `dpmpp_2m`, `dpmpp_3m`, `dpmpp_2s`, `dpmpp_sde`, `deis`, and `unipc`. `dpmpp_2s` is the deterministic single-step DPM-Solver++ implementation; it is not the ancestral algorithm sometimes called DPM++ 2S a. `dpmpp_sde` selects second-order multistep SDE-DPM-Solver++.

Use `scheduler_kwargs` for supported constructor options such as `use_karras_sigmas`, `use_exponential_sigmas`, `solver_order` or `timestep_spacing`. Unsupported options fail instead of being ignored. Options which replace the training betas, change the prediction type or switch to flow sigmas are protected. DDIM additionally accepts `eta` in `[0,1]` (zero is deterministic given the initial noise).

PNDM needs at least four inference steps, PLMS at least two. Other VP schedulers allow `1 <= sampling_steps <= num_steps`. The requested number of solver steps need not equal network evaluations: Heun, PNDM and single-step higher-order solvers may evaluate the network more often. Some methods end at a positive minimum sigma; approximation errors and results differ even when the training target conversion is exact.

Continuous ODE methods are `euler`, `midpoint`, `heun` and `rk4`. A positive `scheduler_kwargs.shift` warps the inference time grid using `shift*t / (1+(shift-1)*t)`. On `vp_sde`, `subvp` and `ve`, `euler_maruyama` integrates the reverse SDE with fresh Gaussian noise each step. It uses drift `a'/a` and diffusion variance `2 s s' - 2 (a'/a) s²`, derived from the same path coefficients used in training.

EDM uses positive sigma grids followed by a denoising step to zero. Select `scheduler_kwargs.spacing=karras|exponential|linear` and optionally `rho`. Consistency refinement uses the same grid choices but injects noise with standard deviation `sqrt(sigma_next²-sigma_min²)` between denoising calls.

`generate(..., generator=torch.Generator().manual_seed(42))` controls initial and stochastic sampling noise. The generator must be on the sample device. Sampling temporarily sets evaluation mode and restores the previous mode. `return_trajectory=True` includes the initial state and each solver update; sigma-based VP solvers store states in their scheduler's coordinates.

## EDM and consistency details

EDM samples training sigma from a log-normal distribution (`log_sigma_mean=-1.2`, `log_sigma_std=1.2`). Its denoiser applies input scaling, a skip connection and output scaling, then minimizes a weighted denoising loss. This is more than simply choosing an x0 target on the original VP model.

Consistency models modify that parameterization to enforce `f(x, sigma_min)=x` exactly. Standalone training compares adjacent sigma levels built from the **same** clean sample and noise. Distillation uses two evaluations of a frozen teacher for a Heun step between those levels. Both regress against a stop-gradient EMA copy of the student at the lower sigma; the EMA updates once after every optimizer step and is included in checkpoints. This demo uses a fixed Karras training grid and MSE/MAE, not an adaptive grid curriculum or perceptual loss.

Group normalization is the default for new image models and the model overlays; it avoids batch-statistic drift between an online consistency model and its frozen target. Existing baseline MNIST configs retain their explicit batch-normalization setting.

Teachers can be native EDM, VE or discrete VP models. VP teacher inputs are rescaled into VP coordinates, and log-sigma is mapped to the trained timestep range. Requests outside a teacher's supported noise range fail; adjust the student's sigma bounds when using VP teachers. Distillation loads the teacher only when training or computing its loss. Sampling the student does not need the teacher file or weights.

## Distribution and trajectory distillation

`DistillationModel` separates the training method from `model_type`, which still describes the underlying path and parameterization. Its `method` choices are `dmd`, `dmd2`, `progressive`, `trajectory` and `reflow`. These are training objectives, not additional entries in the ordinary scheduler registry.

### DMD and DMD2

Both methods freeze a teacher representing the real-data distribution and learn a separate fake-score network on detached student samples. At a sampled VP noise level, both score networks denoise the same noisy generated sample. With their clean estimates `real_x0` and `fake_x0`, the generator receives the surrogate gradient

```text
scale = mean(abs(generated - real_x0))        # per sample
gradient = (fake_x0 - real_x0) / max(scale, 1e-6)
loss_dm = 0.5 * MSE(generated, stopgrad(generated - gradient))
```

The score estimates and normalization are detached. Only the generator receives this gradient. The fake-score network has a separate optimizer and learns the configured denoising target. `dm_min_time` and `dm_max_time` bound the noise-level sampling range.

- **DMD** adds paired regression against a deterministic teacher DDIM trajectory using the same initial noise. This implementation produces pairs online and uses the configured MSE/MAE, rather than a precomputed dataset and LPIPS. `regression_weight` must be positive; DMD trains a one-step generator.
- **DMD2** removes paired regression. By default it updates the critic five times per generator update, with `critic_updates` controlling that ratio. A small discriminator head shares the fake-score network's hidden features; softplus adversarial losses compare real and generated clean data. `gan_weight=0` disables this term for ablations. Unlike the original large-image setup, this demo uses a linear head on pooled U-Net bottleneck features (or dense hidden features), with no diffusion augmentation or perceptual network.
- **Few-step DMD2** trains a random stage using backward simulation: preceding student steps run without gradients, each prediction is re-noised to the next scheduled level, and only the selected prediction receives the generator gradient. Inference uses the same denoise/re-noise process. `student_steps=1` selects single-step DMD2; the preset uses four steps.

Train DMD/DMD2 through `Trainer.fit` or call `configure_optimizers`, `configure_auxiliary_optimizers` and `optimize_batch` in a custom loop. Calling `loss().backward()` alone computes only the generator objective and does not train the required fake-score model. Logged fake-score, discriminator, distribution-matching and regression terms remain separate; their sum and validation surrogate are not perceptual-quality measures.

For DMD and paired trajectory regression, `teacher_start_step` optionally selects the zero-based first timestep of the teacher's DDIM grid. It defaults to the terminal training timestep. Starting earlier can avoid error amplification from epsilon predictions at nearly zero SNR; it also approximates the prior at that earlier noise level, so inspect the resulting teacher samples before training. The student grid and distribution-matching noise range remain independently configured. A v-prediction student with `initialize_from_teacher=false` can use an epsilon teacher without an unstable epsilon-to-x0 conversion at the student's terminal timestep.

### Progressive, paired trajectory and reflow methods

Progressive distillation samples an adjacent pair of teacher DDIM steps, `t -> middle -> end`, and derives a target so one student update lands at the same endpoint. Given the teacher endpoint `x_end`:

```text
r = s(end) / s(t)
x0_target = (x_end - r * x_t) / (a(end) - r * a(t))
```

The clean target is then converted to the student's parameterization. Each stage requires `teacher_steps = 2 * student_steps`. For example, distill a 16-step teacher into eight steps, then use that trained student as the next teacher with `teacher_steps=8, student_steps=4`. When the teacher is a previous progressive student, its trained grid is checked. Stages run as separate CLI jobs, allowing inspection of each checkpoint before another halving.

The `trajectory` method is a simple paired endpoint-regression baseline: sample a deterministic multi-step DDIM teacher and regress a one-step student's clean output from the same noise. It does not use score matching or a critic, and is not claimed as an implementation of a separate named perceptual-distillation algorithm.

Reflow samples a frozen linear-velocity teacher with Heun, retaining each initial noise and final sample as a pair. It regresses the straight-path velocity between those paired endpoints. Repeating with a reflow checkpoint produces another reflow round. This generates pairs online; no offline dataset needs to be prepared. Euler, midpoint, Heun and RK4 remain available, with one-step Euler as the preset.

### Teacher and inference contracts

DMD/DMD2/progressive/trajectory currently support **VP** teachers and require identical training beta tables. Distribution matching needs an undistilled teacher trained across the noise schedule; progressive/trajectory objectives also accept a previous progressive teacher at its trained step count. Reflow requires a **linear velocity** teacher. Input dimensions, class conditioning and data scaling must agree. By default teacher weights initialize the generator and fake-score network, requiring matching backbone sizes, normalization and prediction type. To use different hidden sizes or targets, set `initialize_from_teacher=false`; that starts those networks from scratch.

Distilled VP students use `sampling_scheduler=distilled` and their trained `student_steps` grid. DMD2 sampling is stochastic for multiple steps; progressive sampling uses deterministic DDIM updates. Changing the sampler or requested step count is rejected because it would change the process the student learned. Reflow and consistency students retain their respective sampler contracts.

All student checkpoints sample independently of the teacher file. Resuming training still needs the teacher. DMD/DMD2 checkpoints also contain the fake-score model, discriminator, both optimizers and critic-update phase. Teachers are excluded from student state dictionaries and never receive gradients.

## Native training and checkpoint format

The trainer owns device placement, Adam, validation, optional TensorBoard logging, JSONL metrics, gradient clipping, early stopping and checkpoint cadence. Models are `torch.nn.Module`; data modules are ordinary Python classes. There is no Lightning import or dependency.

Native `.ckpt` checkpoints contain the model class, constructor arguments, parameters and buffers, optimizer/scheduler states, epoch/step counters, EMA target, auxiliary optimizer states, random-number state and resolved configuration. Resume checks the model configuration before loading. Native checkpoints load through `diffusion.load_model` (also available from `diffusion.training`); legacy class-specific checkpoints can be loaded with `DDPMTab.load_from_checkpoint` or `DDPM2d.load_from_checkpoint` when their constructor metadata matches.

The loop is single-device and full precision. It does not emulate Lightning callbacks, automatic SWA, distributed launch or the old standalone scripts' individual argparse flags. Use `--set` overrides on the shared CLI instead.

### Safetensors model files

`save_model(model, "model.safetensors")` and `model.save_safetensors(...)` write the complete registered model state with the [safetensors tensor API](https://huggingface.co/docs/safetensors/api/torch). The file includes backbone weights, noise-schedule buffers, consistency EMA, distillation critics and update counters. The unregistered frozen teacher remains external. Optimizer state, LR scheduler state, trainer counters and RNG state are available in the resumable `.ckpt` format.

The safetensors header embeds `format=pt`, `gen_kit_format=1`, the model class name and JSON `hyper_parameters`. Constructor arguments must be JSON-serializable. `load_model` reconstructs one of the registered model classes from that metadata, restores tensors strictly, and preserves saved dtypes and the requested device. Class names in metadata are resolved through a fixed registry. Files with missing/unsupported metadata or mismatched tensor keys fail explicitly; a failed safetensors read does not fall back to pickle.

Exports use independent, contiguous CPU copies of each tensor, supporting strided and shared source storage without changing the live model. A temporary file in the destination directory is replaced atomically after a successful write. Tensor-only readers such as `safetensors.torch.load_file` can also read the exported weights directly.

The `export` CLI command converts an existing `.ckpt` into this model format. Set `trainer.export_safetensors=true` to write matching `best.safetensors`, `last.safetensors` and periodic epoch exports alongside the full training checkpoints. Sampling, validation and frozen-teacher loading accept either format. `fit --checkpoint` requires a full `.ckpt`; exported models can instead initialize a fresh training run through `Trainer.fit(load_model(path), data)`.

## Primary references

- [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), and [v parameterization](https://arxiv.org/abs/2202.00512).
- [Score-based SDEs](https://arxiv.org/abs/2011.13456), [Flow Matching](https://arxiv.org/abs/2210.02747), and [Rectified Flow](https://arxiv.org/abs/2209.03003).
- [EDM](https://arxiv.org/abs/2206.00364) and the authors' [loss/preconditioning implementation](https://github.com/NVlabs/edm/tree/main/training).
- [Consistency Models](https://arxiv.org/abs/2303.01469) and the authors' [training and sampling implementation](https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py).
- [DPM-Solver++](https://arxiv.org/abs/2211.01095) and [Diffusers scheduler documentation](https://huggingface.co/docs/diffusers/api/schedulers/overview).
- [DMD](https://arxiv.org/abs/2311.18828), [DMD2](https://arxiv.org/abs/2405.14867), and the authors' [score-gradient and adversarial implementation](https://github.com/tianweiy/DMD2/blob/main/main/sd_guidance.py).
- [Progressive Distillation](https://arxiv.org/abs/2202.00512) and [Rectified Flow / reflow](https://arxiv.org/abs/2209.03003).
