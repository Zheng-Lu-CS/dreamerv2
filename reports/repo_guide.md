# DreamerV2 仓库解读与运行指南

## 1. 这个仓库是做什么的

这是 `DreamerV2` 的 TensorFlow 2 实现，核心思想是：

- 先学习一个世界模型（world model），把高维观测压缩到隐状态里。
- 再在隐空间里“想象”未来轨迹。
- 基于 imagined trajectories 训练 actor 和 critic，而不是完全依赖真实环境交互。

仓库默认重点支持 3 类环境：

- `atari`
- `dmc`（DeepMind Control Suite）
- `crafter`

另外，它还提供了一个 Python API，可以接入一般的 Gym 环境，`examples/minigrid.py` 就是示例。

## 2. 文件组织形式

根目录大致可以按下面理解：

```text
dreamerv2/
|-- README.md
|-- requirements.txt
|-- setup.py
|-- Dockerfile
|-- examples/
|   `-- minigrid.py
|-- scores/
|   `-- *.json
|-- dreamerv2/
|   |-- train.py
|   |-- api.py
|   |-- agent.py
|   |-- expl.py
|   |-- configs.yaml
|   `-- common/
|       |-- envs.py
|       |-- nets.py
|       |-- replay.py
|       |-- driver.py
|       |-- logger.py
|       |-- tfutils.py
|       |-- dists.py
|       |-- ...
`-- reports/
    `-- repo_guide.md
```

各部分职责如下：

- `README.md`
  - 官方使用说明，给出 Atari / DMC 的训练命令。
- `requirements.txt`
  - 基础依赖列表，主要是 TensorFlow、TF Probability、Gym、DM Control。
- `setup.py`
  - 包安装入口，定义了 `dreamerv2` 命令行入口。
- `examples/minigrid.py`
  - 最直接的 demo，用 Python API 在 MiniGrid 上训练。
- `scores/*.json`
  - 论文/实验结果曲线相关数据，不参与训练主逻辑。
- `dreamerv2/train.py`
  - 命令行训练入口，适合 Atari / DMC / Crafter。
- `dreamerv2/api.py`
  - 对外的 Python API，适合自定义 Gym 环境。
- `dreamerv2/agent.py`
  - 核心 agent：世界模型 + actor-critic。
- `dreamerv2/expl.py`
  - 探索策略，比如 `Plan2Explore`。
- `dreamerv2/configs.yaml`
  - 所有训练配置和预设。
- `dreamerv2/common/`
  - 通用基础设施：网络、环境包装、经验回放、日志、分布、训练工具。

## 3. 核心算法文件

如果你想抓主线，建议按这个顺序看：

### 3.1 `dreamerv2/train.py`

这是命令行训练主入口，负责把整个训练流程串起来：

- 读取 `configs.yaml`
- 创建环境
- 创建 replay buffer
- 创建 agent
- 先随机填充 replay（prefill）
- 反复执行：
  - 采样真实环境数据
  - 更新 world model
  - 更新 actor / critic
  - 定期评估
  - 定期保存参数

这里还能直接看出 CLI 版本只支持：

- `dmc_*`
- `atari_*`
- `crafter_*`

也就是任务名会按 `suite_task` 形式解析。

### 3.2 `dreamerv2/agent.py`

这是最核心的算法实现文件，包含两个主模块：

- `WorldModel`
- `ActorCritic`

#### `WorldModel`

职责：

- 对观测做预处理
- 用 `Encoder` 编码图像/向量观测
- 用 RSSM 建模 latent dynamics
- 用 `Decoder` 重建观测
- 预测 reward
- 可选预测 discount

最关键的方法：

- `train()`
  - 训练世界模型
- `loss()`
  - 计算重建损失、reward 损失、discount 损失和 KL 损失
- `imagine()`
  - 在 latent space 中 rollout imagined trajectories
- `video_pred()`
  - 生成重建/开放预测视频，用于可视化

#### `ActorCritic`

职责：

- 从世界模型给出的 latent state 出发，想象未来轨迹
- 用 imagined reward 和 imagined value 训练策略和值函数

最关键的方法：

- `train()`
- `actor_loss()`
- `critic_loss()`
- `target()`

这里就是 Dreamer 类方法的核心：不是直接在真实像素轨迹上做 model-free RL，而是在 latent dynamics 上训练 policy。

### 3.3 `dreamerv2/common/nets.py`

这里是神经网络组件库，尤其重要的是：

- `EnsembleRSSM`
  - DreamerV2 的 latent dynamics 主体
  - 包含 deterministic state 和 stochastic state
  - 支持离散随机变量 `discrete`
- `Encoder`
  - 编码图像/向量输入
- `Decoder`
  - 从 latent feature 重建观测
- `MLP`
  - reward head、actor、critic 等都基于它
- `DistLayer`
  - 统一输出各种概率分布

如果你只想弄懂 DreamerV2 的“状态空间模型”是怎么写的，重点看 `EnsembleRSSM`。

### 3.4 `dreamerv2/common/replay.py`

经验回放缓冲区：

- 按 episode 存盘为 `.npz`
- 支持从 replay 中切片出固定长度序列
- 用 `tf.data.Dataset` 提供训练数据

DreamerV2 训练依赖的是“序列片段”，不是单步 transition，这个文件很关键。

### 3.5 `dreamerv2/common/envs.py`

环境适配层，提供：

- `DMC`
- `Atari`
- `Crafter`
- `GymWrapper`
- `OneHotAction`
- `NormalizeAction`
- `ResizeImage`
- `TimeLimit`

如果你要接自己的环境，通常会先看这里，再看 `api.py`。

### 3.6 `dreamerv2/expl.py`

探索策略实现：

- `Plan2Explore`
- `ModelLoss`

默认配置里 `expl_behavior: greedy`，所以很多基础实验不会走到复杂探索分支；但如果你想研究探索机制，这个文件就是入口。

## 4. 训练流程是怎样串起来的

主流程可以概括成：

1. 从配置文件读取超参数。
2. 创建环境和 replay buffer。
3. 先随机和环境交互，填满一部分 replay。
4. 训练世界模型：
   - 编码观测
   - 学 latent dynamics
   - 重建观测
   - 预测 reward / discount
5. 从 replay 里的后验状态出发，在 latent space 中 imagine 未来轨迹。
6. 用 imagined trajectories 更新 actor 和 critic。
7. 再回到真实环境继续收集数据。
8. 循环执行，并定期评估、记录日志、保存模型。

这是一个典型的 model-based RL 训练闭环。

## 5. 如何运行 demo

这个仓库最像 demo 的是：

- `examples/minigrid.py`

它不是通过 `train.py` 命令行入口跑，而是通过 Python API：

```python
import dreamerv2.api as dv2
dv2.train(env, config)
```

### 5.1 建议安装步骤

在仓库根目录执行：

```powershell
pip install -r requirements.txt
pip install gym-minigrid
```

如果你想把当前仓库作为本地包使用，也可以执行：

```powershell
pip install -e .
```

注意：

- `examples/minigrid.py` 里使用的是 `gym_minigrid` 包名，所以你需要安装对应包。
- `requirements.txt` 里没有包含 `gym-minigrid`，这是额外依赖。
- 仓库 README 推荐的 TensorFlow 版本是 `2.6.0`。

### 5.2 直接运行 demo

在仓库根目录执行：

```powershell
python examples/minigrid.py
```

这个 demo 做的事情是：

- 创建 `MiniGrid-DoorKey-6x6-v0`
- 包一层 `RGBImgPartialObsWrapper`
- 调用 `dreamerv2.api.train()` 开始训练

### 5.3 demo 运行时要注意

- `dreamerv2/train.py` 明确要求 GPU；代码里有 `assert tf.config.experimental.list_physical_devices('GPU')`
- `api.py` 没有这个 GPU 断言，所以 MiniGrid demo 相对更容易在普通环境里先跑通
- 即便如此，DreamerV2 依然比较吃显存和算力，CPU 上训练会慢很多

## 6. 如何进行训练

这里分两类。

### 6.1 用命令行训练仓库内置支持的任务

#### 训练 Atari

```powershell
python dreamerv2/train.py --logdir ./logdir/atari_pong/dreamerv2/1 --configs atari --task atari_pong
```

#### 训练 DM Control

```powershell
python dreamerv2/train.py --logdir ./logdir/dmc_walker_walk/dreamerv2/1 --configs dmc_vision --task dmc_walker_walk
```

#### 训练 Crafter

```powershell
python dreamerv2/train.py --logdir ./logdir/crafter_reward/dreamerv2/1 --configs crafter --task crafter_reward
```

说明：

- `--configs atari` / `dmc_vision` / `crafter` 会从 `dreamerv2/configs.yaml` 读取对应预设。
- `--task` 需要和 `train.py` 的环境分发逻辑匹配。
- 训练日志、回放数据、模型参数都会写到 `--logdir`。

### 6.2 用 Python API 训练你自己的 Gym 环境

如果是通用 Gym 环境，更推荐仿照 `examples/minigrid.py`：

```python
import gym
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'logdir': './logdir/myenv',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e4,
}).parse_flags()

env = gym.make('YourEnvName')
dv2.train(env, config)
```

这条路线会自动做几件事：

- 用 `GymWrapper` 包装普通 Gym env
- 自动区分离散动作和连续动作
- 自动创建 replay 和训练循环

如果你的环境没有直接输出图像，而是向量观测，也能走这套框架；对应编码器/解码器由配置决定。

## 7. 配置文件怎么理解

主配置文件是：

- `dreamerv2/configs.yaml`

它包含：

- `defaults`
  - 通用默认配置
- `atari`
  - Atari 专用超参数
- `dmc_vision`
  - DMC 图像任务超参数
- `dmc_proprio`
  - DMC 低维状态任务超参数
- `crafter`
  - Crafter 任务超参数
- `debug`
  - 调试配置，缩小 batch、减少步数、更适合快速排错

如果你只是想先跑通，建议用：

```powershell
python dreamerv2/train.py --logdir ./logdir/debug_run --configs atari debug --task atari_pong
```

或者：

```powershell
python dreamerv2/train.py --logdir ./logdir/debug_dmc --configs dmc_vision debug --task dmc_walker_walk
```

这样会更容易验证训练链路是否通了。

## 8. 训练输出会保存到哪里

以 `--logdir ./logdir/atari_pong/dreamerv2/1` 为例，训练中通常会生成：

- `config.yaml`
  - 本次运行的配置快照
- `variables.pkl`
  - 模型参数
- `train_episodes/`
  - 训练 episode 数据
- `eval_episodes/`
  - 评估 episode 数据
- TensorBoard 日志
- JSONL 日志

查看训练曲线：

```powershell
tensorboard --logdir ./logdir
```

## 9. 我对这个仓库的实用建议

如果你现在的目标是“先跑起来，再深入读代码”，推荐顺序是：

1. 先读 `README.md`
2. 再读 `examples/minigrid.py`
3. 然后读 `dreamerv2/api.py`
4. 再读 `dreamerv2/train.py`
5. 最后重点读 `dreamerv2/agent.py` 和 `dreamerv2/common/nets.py`

这样理解成本最低。

如果你的目标是“看清算法”，推荐顺序是：

1. `dreamerv2/agent.py`
2. `dreamerv2/common/nets.py`
3. `dreamerv2/common/replay.py`
4. `dreamerv2/train.py`
5. `dreamerv2/expl.py`

## 10. 需要特别注意的坑

- `train.py` 默认强制要求 GPU，没有 GPU 会直接触发断言失败。
- `requirements.txt` 很简短，不包含 `gym-minigrid`，跑 demo 需要你额外安装。
- `train.py` 的命令行入口不是“任意 Gym 环境通用入口”，它只分发 `dmc`、`atari`、`crafter` 三类任务。
- 自定义环境更适合走 `dreamerv2/api.py`。
- 仓库依赖 TensorFlow 2.6 这一代接口，环境过新时可能需要做兼容处理。

## 11. 一句话总结

这是一个“世界模型 + latent imagination + actor-critic”的 DreamerV2 实现。

- 想跑官方预设任务：用 `dreamerv2/train.py`
- 想跑自定义 Gym demo：看 `examples/minigrid.py` 和 `dreamerv2/api.py`
- 想看核心算法：重点读 `dreamerv2/agent.py` 和 `dreamerv2/common/nets.py`
