import argparse
import pathlib
import shutil
import subprocess
import sys
import types

import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def load_yaml_file(filename):
  loader = yaml.YAML(typ='safe', pure=True)
  return loader.load(pathlib.Path(filename).read_text())


def load_config(argv):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--configs', nargs='+', default=['defaults'])
  parser.add_argument('--task', default=None)
  args, _ = parser.parse_known_args(argv)
  presets = load_yaml_file(pathlib.Path(__file__).parent / 'configs.yaml')
  merged = dict(presets['defaults'])
  for name in args.configs:
    merged = merge_nested(merged, presets[name])
  if args.task:
    merged['task'] = args.task
  return to_namespace(merged)


def merge_nested(base, update):
  result = dict(base)
  for key, value in update.items():
    if isinstance(value, dict) and isinstance(result.get(key), dict):
      result[key] = merge_nested(result[key], value)
    else:
      result[key] = value
  return result


def to_namespace(mapping):
  converted = {}
  for key, value in mapping.items():
    if isinstance(value, dict):
      converted[key] = to_namespace(value)
    else:
      converted[key] = value
  return types.SimpleNamespace(**converted)


def import_common():
  try:
    import common
  except Exception as exc:
    raise SystemExit(
        'Failed to import DreamerV2 runtime modules. '
        'Check TensorFlow / TensorFlow Probability / Keras compatibility '
        f'first. Original error: {exc}') from exc
  return common


def check_tensorflow(config):
  import tensorflow as tf
  result = {
      'tensorflow_version': tf.__version__,
      'gpu_devices': [x.name for x in tf.config.list_physical_devices('GPU')],
      'precision_error': None,
  }
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  if config.precision == 16:
    try:
      from tensorflow.keras.mixed_precision import experimental as prec
      prec.set_policy(prec.Policy('mixed_float16'))
    except Exception as exc:
      result['precision_error'] = str(exc)
  return result


def make_env(config):
  common = import_common()
  suite, task = config.task.split('_', 1)
  if suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.render_size,
        config.atari_grayscale)
    env = common.OneHotAction(env)
  elif suite == 'dmc':
    env = common.DMC(
        task, config.action_repeat, config.render_size, config.dmc_camera)
    env = common.NormalizeAction(env)
  elif suite == 'crafter':
    reward = task == 'reward'
    env = common.Crafter(None, reward)
    env = common.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  return env


def check_ffmpeg():
  path = shutil.which('ffmpeg')
  if not path:
    return {'available': False, 'path': None}
  command = [path, '-version']
  output = subprocess.run(
      command, check=False, capture_output=True, text=True)
  return {
      'available': output.returncode == 0,
      'path': path,
      'version': output.stdout.splitlines()[0] if output.stdout else '',
  }


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--configs', nargs='+', default=['atari', 'atari_demo_pong'],
      help='Config presets from configs.yaml.')
  parser.add_argument('--task', default='atari_pong')
  args = parser.parse_args(argv)

  config = load_config([
      '--configs', *args.configs,
      '--task', args.task,
  ])
  failures = []

  print('Checking DreamerV2 environment...')
  tf_result = check_tensorflow(config)
  print(f"TensorFlow: {tf_result['tensorflow_version']}")
  print(f"Visible GPUs: {tf_result['gpu_devices'] or 'none'}")
  if not tf_result['gpu_devices']:
    failures.append('No GPU visible to TensorFlow.')
  if tf_result['precision_error']:
    failures.append(
        'TensorFlow mixed precision setup failed: ' +
        tf_result['precision_error'])

  ffmpeg = check_ffmpeg()
  print(f"ffmpeg: {ffmpeg['path'] or 'missing'}")
  if not ffmpeg['available']:
    failures.append('ffmpeg not available in PATH.')

  try:
    import gym.envs.atari  # pylint: disable=unused-import
    print('gym.envs.atari import: ok')
  except Exception as exc:
    failures.append(f'Legacy gym Atari import failed: {exc}')
    print(f'gym.envs.atari import failed: {exc}')

  try:
    env = make_env(config)
    obs = env.reset()
    action = env.act_space['action'].sample()
    obs = env.step({'action': action})
    print(f"Environment reset/step: ok ({config.task})")
    print(f"Observation keys: {sorted(obs.keys())}")
    env.close()
  except BaseException as exc:
    failures.append(f'Environment check failed for {config.task}: {exc}')
    print(f'Environment check failed: {exc}')

  if failures:
    print('\nFAILED CHECKS:')
    for item in failures:
      print(f'- {item}')
    raise SystemExit(1)

  print('\nAll checks passed.')


if __name__ == '__main__':
  main()
