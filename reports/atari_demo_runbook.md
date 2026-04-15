# DreamerV2 Atari Demo Runbook

## What was added

- `dreamerv2/check_env.py`
- `dreamerv2/export_demo.py`
- `dreamerv2/visualize_demo.py`
- `configs.yaml` presets:
  - `atari_demo_pong`
  - `atari_demo_pong_smoke`
  - `atari_demo_pong_extend`

## Important environment note

This repository still expects the legacy Atari Gym stack used by the original
project. The current local Windows `.venv` is not the target runtime for Atari
training. Use a fresh Linux environment on AutoDL first.

The current local environment already shows three incompatible pieces:

- `tensorflow_probability` expects a newer TensorFlow than `2.6.0`
- `keras` is newer than the installed TensorFlow runtime
- `gym==0.26.x` does not expose `gym.envs.atari`

## 1. Environment check on AutoDL

Run this first in the Linux training environment:

```bash
python dreamerv2/check_env.py --configs atari atari_demo_pong --task atari_pong
```

Expected result:

- TensorFlow sees at least one GPU
- `ffmpeg` is available
- `gym.envs.atari` imports successfully
- `atari_pong` can `reset()` and `step()`

## 2. Smoke run

This is the shortest run that validates logging, checkpoint save/load, replay,
and evaluation outputs.

```bash
python dreamerv2/train.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_smoke \
  --configs atari atari_demo_pong atari_demo_pong_smoke \
  --task atari_pong
```

Expected artifacts in the logdir:

- `config.yaml`
- `variables.pkl`
- `metrics.jsonl`
- `train_episodes/`
- `eval_episodes/`
- TensorBoard event files

## 3. Fast demo run

Use the same checkpointing behavior as the original training loop, but with the
smaller `Pong` demo preset.

```bash
python dreamerv2/train.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_demo \
  --configs atari atari_demo_pong \
  --task atari_pong
```

Default target:

- `5e5` training steps

## 4. Continue training from the same logdir

The repository already resumes from:

- `variables.pkl`
- `train_episodes/`
- `eval_episodes/`

Resume by re-running the same command with the same `--logdir`.

For a longer run:

```bash
python dreamerv2/train.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_demo \
  --configs atari atari_demo_pong_extend \
  --task atari_pong
```

For an even longer run, keep the same command and override the step budget:

```bash
python dreamerv2/train.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_demo \
  --configs atari atari_demo_pong_extend \
  --task atari_pong \
  --steps 2e6
```

## 5. Export rollout and world-model videos

After training finishes or after any checkpoint you want to inspect:

```bash
python dreamerv2/export_demo.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_demo \
  --episodes 1 \
  --max_steps 3000
```

Default export location:

```text
<logdir>/exports/demo
```

Expected outputs:

- `episode_000_rollout.gif`
- `episode_000_trace.csv`
- `episode_000_trace.npz`
- `episode_000/openl_image.gif`
- `episode_000/openl_image.npy`
- `training_metrics.csv`
- `summary.json`

## 6. Generate offline plots for local viewing

Run this on the training machine after export:

```bash
python dreamerv2/visualize_demo.py \
  --logdir /root/autodl-tmp/dreamerv2/pong_demo
```

Default plot directory:

```text
<logdir>/exports/demo/plots
```

Expected outputs:

- `training_curves.png`
- `episode_000_trace_diagnostics.png`
- `episode_000_trace_latent_heatmap.png`
- `visualization_summary.json`

## 7. TensorBoard

For live monitoring during training:

```bash
tensorboard --logdir /root/autodl-tmp/dreamerv2
```

Useful tags to inspect:

- `scalars/train_return`
- `scalars/eval_return`
- `scalars/model_kl`
- `scalars/prior_ent`
- `scalars/post_ent`
- `scalars/actor_ent`
- `scalars/critic_target`
- `scalars/actor_grad_norm`
- `scalars/critic_grad_norm`
- `train_openl_image`
- `eval_openl_image`

## 8. Local viewing workflow

If the local machine stays CPU-only and does not have the Atari runtime stack,
copy back only the exported artifacts:

- rollout GIFs
- open-loop prediction GIFs
- `training_metrics.csv`
- generated PNG plots
- `summary.json`

That is enough to review the demo without replaying Atari locally.
