import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def load_yaml_file(filename):
  loader = yaml.YAML(typ='safe', pure=True)
  return loader.load(pathlib.Path(filename).read_text())


def load_config(argv):
  common = import_common()
  configs = load_yaml_file(pathlib.Path(__file__).parent / 'configs.yaml')
  parsed, remaining = common.Flags(configs=['defaults']).parse(
      argv, known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  return common.Flags(config).parse(remaining)


def import_common():
  try:
    import common
  except Exception as exc:
    raise SystemExit(
        'Failed to import DreamerV2 runtime modules. '
        'Check TensorFlow / TensorFlow Probability / Keras compatibility '
        f'first. Original error: {exc}') from exc
  return common


def import_agent():
  try:
    import agent
  except Exception as exc:
    raise SystemExit(
        'Failed to import DreamerV2 agent modules. '
        'Check TensorFlow / Keras compatibility first. '
        f'Original error: {exc}') from exc
  return agent


def configure_tensorflow(config):
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(False)
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))
  return tf


def make_env(config):
  common = import_common()
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = common.DMC(
        task, config.action_repeat, config.render_size, config.dmc_camera)
    env = common.NormalizeAction(env)
  elif suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.render_size,
        config.atari_grayscale)
    env = common.OneHotAction(env)
  elif suite == 'crafter':
    outdir = None
    reward = task == 'reward'
    env = common.Crafter(outdir, reward)
    env = common.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  return env


def transition_batch(episode):
  batch = {}
  for key in episode[0]:
    batch[key] = np.array([np.array([step[key] for step in episode])])
  return batch


def build_variables(agnt, env, config):
  episode = []
  obs = env.reset()
  zero_action = np.zeros(env.act_space['action'].shape, np.float32)
  episode.append({**obs, 'action': zero_action})
  while len(episode) < max(config.dataset.length, 10):
    action = env.act_space['action'].sample()
    obs = env.step({'action': action})
    episode.append({**obs, 'action': np.array(action)})
    if obs['is_last']:
      break
  agnt.train(transition_batch(episode))


def tensor_to_numpy(value):
  if hasattr(value, 'numpy'):
    return value.numpy()
  return np.array(value)


def save_gif(filename, frames, fps):
  common = import_common()
  frames = np.asarray(frames)
  if np.issubdtype(frames.dtype, np.floating):
    frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
  data = common.encode_gif(frames, fps)
  pathlib.Path(filename).write_bytes(data)


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


def write_metrics_csv(records, filename):
  filename = pathlib.Path(filename)
  if not records:
    return
  keys = sorted({key for record in records for key in record})
  with filename.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for record in records:
      writer.writerow(record)


def summarize_metrics(records, keys):
  if not records:
    return {}
  summary = {'last_step': records[-1].get('step', 0)}
  for key in keys:
    values = [record[key] for record in records if key in record]
    if values:
      summary[key] = values[-1]
  eval_returns = [record['eval_return'] for record in records
                  if 'eval_return' in record]
  if eval_returns:
    summary['best_eval_return'] = max(eval_returns)
  return summary


def run_episode(agnt, env, config, max_steps):
  import tensorflow as tf
  common = import_common()
  obs = env.reset()
  episode = []
  zero_action = np.zeros(env.act_space['action'].shape, np.float32)
  episode.append({**obs, 'action': zero_action})
  frames = [np.array(obs['image'])]
  rewards = []
  action_ids = []
  action_probs = []
  values = []
  reward_preds = []
  discount_preds = []
  entropies = []
  latent_feats = []
  state = None
  total_return = 0.0

  for _ in range(max_steps):
    batch_obs = {
        key: tf.convert_to_tensor(value[None])
        for key, value in obs.items()}
    if state is None:
      latent = agnt.wm.rssm.initial(1)
      action = tf.zeros((1,) + agnt.act_space.shape)
      state = (latent, action)
    latent, prev_action = state
    embed = agnt.wm.encoder(agnt.wm.preprocess(batch_obs))
    sample = not config.eval_state_mean
    latent, _ = agnt.wm.rssm.obs_step(
        latent, prev_action, embed, batch_obs['is_first'], sample)
    feat = agnt.wm.rssm.get_feat(latent)
    actor = agnt._task_behavior.actor(feat)
    action = actor.mode()
    action = common.action_noise(action, config.eval_noise, agnt.act_space)
    value = agnt._task_behavior.critic(feat).mode()
    reward_pred = agnt.wm.heads['reward'](feat).mode()
    if 'discount' in agnt.wm.heads:
      discount_pred = agnt.wm.heads['discount'](feat).mean()
    else:
      discount_pred = np.array([config.discount], np.float32)
    probs = actor.probs_parameter() if hasattr(actor, 'probs_parameter') else action
    state = (latent, action)

    action_np = tensor_to_numpy(action)[0]
    probs_np = tensor_to_numpy(probs)[0]
    value_np = float(tensor_to_numpy(value).reshape(-1)[0])
    reward_pred_np = float(tensor_to_numpy(reward_pred).reshape(-1)[0])
    discount_pred_np = float(np.array(tensor_to_numpy(discount_pred)).reshape(-1)[0])
    entropy_np = float(tensor_to_numpy(actor.entropy()).reshape(-1)[0])
    feat_np = tensor_to_numpy(feat)[0]

    obs = env.step({'action': action_np})
    total_return += float(obs['reward'])
    frames.append(np.array(obs['image']))
    rewards.append(float(obs['reward']))
    action_ids.append(int(np.argmax(action_np)))
    action_probs.append(probs_np.astype(np.float32))
    values.append(value_np)
    reward_preds.append(reward_pred_np)
    discount_preds.append(discount_pred_np)
    entropies.append(entropy_np)
    latent_feats.append(feat_np.astype(np.float32))
    episode.append({**obs, 'action': action_np.astype(np.float32)})
    if obs['is_last']:
      break

  trace = {
      'reward': np.array(rewards, np.float32),
      'cumulative_return': np.cumsum(np.array(rewards, np.float32)),
      'action_id': np.array(action_ids, np.int32),
      'action_probs': np.array(action_probs, np.float32),
      'value': np.array(values, np.float32),
      'reward_pred': np.array(reward_preds, np.float32),
      'discount_pred': np.array(discount_preds, np.float32),
      'actor_entropy': np.array(entropies, np.float32),
      'latent_feat': np.array(latent_feats, np.float32),
  }
  return episode, trace, frames, total_return


def write_trace_csv(trace, filename):
  filename = pathlib.Path(filename)
  rows = len(trace['reward'])
  prob_dim = trace['action_probs'].shape[-1] if rows else 0
  fieldnames = [
      'step',
      'reward',
      'cumulative_return',
      'value',
      'reward_pred',
      'discount_pred',
      'actor_entropy',
      'action_id',
  ] + [f'action_prob_{index}' for index in range(prob_dim)]
  with filename.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(rows):
      row = {
          'step': index,
          'reward': float(trace['reward'][index]),
          'cumulative_return': float(trace['cumulative_return'][index]),
          'value': float(trace['value'][index]),
          'reward_pred': float(trace['reward_pred'][index]),
          'discount_pred': float(trace['discount_pred'][index]),
          'actor_entropy': float(trace['actor_entropy'][index]),
          'action_id': int(trace['action_id'][index]),
      }
      for prob_index in range(prob_dim):
        row[f'action_prob_{prob_index}'] = float(
            trace['action_probs'][index, prob_index])
      writer.writerow(row)


def export_report_videos(agnt, episode, outdir, fps):
  outdir.mkdir(parents=True, exist_ok=True)
  batched_episode = transition_batch(episode)
  report = agnt.report(batched_episode)
  saved = {}
  for name, video in report.items():
    array = tensor_to_numpy(video)
    np.save(outdir / f'{name}.npy', array)
    if array.shape[-1] == 1:
      array = array[..., 0]
      array = array[..., None]
    save_gif(outdir / f'{name}.gif', array, fps)
    saved[name] = {
        'gif': f'{name}.gif',
        'npy': f'{name}.npy',
        'shape': list(array.shape),
    }
  return saved


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, default=None)
  parser.add_argument('--checkpoint', type=pathlib.Path, default=None)
  parser.add_argument('--episodes', type=int, default=1)
  parser.add_argument('--max_steps', type=int, default=3000)
  parser.add_argument('--fps', type=int, default=20)
  parser.add_argument(
      '--configs', nargs='+', default=['atari', 'atari_demo_pong'],
      help='Config presets from configs.yaml.')
  parser.add_argument('--task', default='atari_pong')
  args = parser.parse_args(argv)

  config = load_config([
      '--configs', *args.configs,
      '--task', args.task,
      '--logdir', str(args.logdir),
  ])
  args.logdir = args.logdir.expanduser()
  outdir = (args.outdir or (args.logdir / 'exports' / 'demo')).expanduser()
  outdir.mkdir(parents=True, exist_ok=True)
  checkpoint = (args.checkpoint or (args.logdir / 'variables.pkl')).expanduser()
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)

  configure_tensorflow(config)
  common = import_common()
  agent = import_agent()
  env = make_env(config)
  step = common.Counter(0)
  agnt = agent.Agent(config, env.obs_space, env.act_space, step)
  build_variables(agnt, env, config)
  agnt.load(checkpoint)

  episode_summaries = []
  exports = []
  for episode_index in range(args.episodes):
    episode, trace, frames, score = run_episode(agnt, env, config, args.max_steps)
    prefix = f'episode_{episode_index:03d}'
    save_gif(outdir / f'{prefix}_rollout.gif', np.asarray(frames), args.fps)
    np.savez_compressed(outdir / f'{prefix}_trace.npz', **trace)
    write_trace_csv(trace, outdir / f'{prefix}_trace.csv')
    report_files = export_report_videos(agnt, episode, outdir / prefix, args.fps)
    episode_summaries.append({
        'episode': episode_index,
        'return': float(score),
        'steps': len(trace['reward']),
        'rollout_gif': f'{prefix}_rollout.gif',
        'trace_npz': f'{prefix}_trace.npz',
        'trace_csv': f'{prefix}_trace.csv',
    })
    exports.append({
        'episode': episode_index,
        'report_files': report_files,
    })

  metrics = read_metrics_jsonl(args.logdir / 'metrics.jsonl')
  write_metrics_csv(metrics, outdir / 'training_metrics.csv')
  selected_metrics = [
      'train_return', 'eval_return', 'model_loss', 'model_kl',
      'prior_ent', 'post_ent', 'actor_ent', 'critic_target',
      'actor_grad_norm', 'critic_grad_norm', 'fps',
  ]
  summary = {
      'logdir': str(args.logdir),
      'checkpoint': str(checkpoint),
      'config_task': config.task,
      'episodes': episode_summaries,
      'reports': exports,
      'metrics': summarize_metrics(metrics, selected_metrics),
  }
  (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
  env.close()
  print(f'Exported demo artifacts to {outdir}')


if __name__ == '__main__':
  main()
