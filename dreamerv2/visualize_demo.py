import argparse
import json
import pathlib

import numpy as np


def import_matplotlib():
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise SystemExit(
        'matplotlib is required for visualize_demo.py. '
        'Install it with `pip install matplotlib`.') from exc
  return plt


def read_metrics_jsonl(filename):
  filename = pathlib.Path(filename)
  if not filename.exists():
    return []
  records = []
  with filename.open() as stream:
    for line in stream:
      line = line.strip()
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return records


def plot_metrics(records, outdir):
  plt = import_matplotlib()
  desired = [
      'train_return', 'eval_return', 'model_loss', 'model_kl',
      'prior_ent', 'post_ent', 'actor_ent', 'critic_target',
      'actor_grad_norm', 'critic_grad_norm', 'fps',
  ]
  available = [
      key for key in desired
      if any(key in record for record in records)]
  if not available:
    return []

  cols = 2
  rows = int(np.ceil(len(available) / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
  for axis, key in zip(axes.flat, available):
    xs = [record['step'] for record in records if key in record]
    ys = [record[key] for record in records if key in record]
    axis.plot(xs, ys, linewidth=1.5)
    axis.set_title(key)
    axis.set_xlabel('step')
    axis.grid(alpha=0.3)
  for axis in axes.flat[len(available):]:
    axis.axis('off')
  fig.tight_layout()
  filename = outdir / 'training_curves.png'
  fig.savefig(filename, dpi=150)
  plt.close(fig)
  return available


def plot_trace(trace_file, outdir):
  plt = import_matplotlib()
  data = np.load(trace_file)
  reward = data['reward']
  if len(reward) == 0:
    return None
  steps = np.arange(len(reward))

  fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
  axes[0].plot(steps, data['cumulative_return'], label='cumulative_return')
  axes[0].plot(steps, data['value'], label='value')
  axes[0].plot(steps, data['reward_pred'], label='reward_pred')
  axes[0].legend(loc='best')
  axes[0].grid(alpha=0.3)
  axes[0].set_title('Episode diagnostics')

  axes[1].plot(steps, data['actor_entropy'], label='actor_entropy')
  axes[1].plot(steps, data['discount_pred'], label='discount_pred')
  axes[1].legend(loc='best')
  axes[1].grid(alpha=0.3)

  action_probs = data['action_probs']
  image = axes[2].imshow(action_probs.T, aspect='auto', origin='lower')
  axes[2].set_ylabel('action')
  axes[2].set_xlabel('step')
  axes[2].set_title('Action probabilities')
  fig.colorbar(image, ax=axes[2], shrink=0.8)

  fig.tight_layout()
  filename = outdir / f'{trace_file.stem}_diagnostics.png'
  fig.savefig(filename, dpi=150)
  plt.close(fig)

  latent = data['latent_feat']
  latent_dims = min(latent.shape[1], 128)
  fig, axis = plt.subplots(figsize=(14, 6))
  heatmap = axis.imshow(
      latent[:, :latent_dims].T, aspect='auto', origin='lower', cmap='magma')
  axis.set_title('Latent feature heatmap')
  axis.set_xlabel('step')
  axis.set_ylabel('latent dim')
  fig.colorbar(heatmap, ax=axis, shrink=0.8)
  fig.tight_layout()
  heatmap_file = outdir / f'{trace_file.stem}_latent_heatmap.png'
  fig.savefig(heatmap_file, dpi=150)
  plt.close(fig)

  return {
      'diagnostics': filename.name,
      'latent_heatmap': heatmap_file.name,
      'steps': int(len(reward)),
      'return': float(data['cumulative_return'][-1]),
  }


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=pathlib.Path, required=True)
  parser.add_argument('--exportdir', type=pathlib.Path, default=None)
  parser.add_argument('--outdir', type=pathlib.Path, default=None)
  args = parser.parse_args(argv)

  logdir = args.logdir.expanduser()
  exportdir = (args.exportdir or (logdir / 'exports' / 'demo')).expanduser()
  outdir = (args.outdir or (exportdir / 'plots')).expanduser()
  outdir.mkdir(parents=True, exist_ok=True)

  records = read_metrics_jsonl(logdir / 'metrics.jsonl')
  plotted_metrics = plot_metrics(records, outdir)

  trace_summaries = []
  for trace_file in sorted(exportdir.glob('episode_*_trace.npz')):
    summary = plot_trace(trace_file, outdir)
    if summary:
      trace_summaries.append(summary)

  summary = {
      'logdir': str(logdir),
      'exportdir': str(exportdir),
      'plotted_metrics': plotted_metrics,
      'trace_summaries': trace_summaries,
  }
  (outdir / 'visualization_summary.json').write_text(
      json.dumps(summary, indent=2))
  print(f'Saved visualizations to {outdir}')


if __name__ == '__main__':
  main()
