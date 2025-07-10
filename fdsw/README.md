# f-dsw + UCB 实验模块

本项目是 D3-RB 实验平台的子模块，专注于将**滑动窗口 (Sliding Window)** 机制与 **UCB** 算法结合，以研究其在非平稳环境下的适应能力。

## 1. 核心算法

我们已经实现了一个新的 Bandit 算法：

- **`UCB_SW`** (`project_root/bandits/ucb_fdsw.py`):
  - 继承自 `UCB1`，因此它隐式地包含并使用 `alpha` 参数（默认为1.0）。
  - 采用 `collections.deque` 实现了一个高效的滑动窗口。
  - 通过 `window_size` 参数控制窗口大小，调节算法的“记忆长度”。

## 2. 实验配置

本模块的实验由 `.yaml` 配置文件驱动。一个典型的配置示例如下 (`config/movielens_abrupt_ucb_sw.yaml`):

```yaml
# 实验元数据
experiment_name: "MovieLens-Abrupt-UCB-SW-Test"
dataset: "movielens"
drift_type: "ml_abrupt"
num_runs: 5
results_dir: "results/"

# 直接定义要对比的算法列表
algorithms:
  - name: "UCB1"
    params:
      alpha: 1.0
  - name: "UCB_SW"
    params:
      window_size: 2000
      alpha: 1.0  # UCB_SW 的 alpha 参数也可在此显式配置
```

### 参数设置参考

根据我们对参考论文 [Cavenaghi et al., 2021] 的分析，对于 `f-dsw` 类算法，其参数的典型搜索范围为：
- **`window_size` (窗口大小)**: `[25, 50, 75, 100, ...]` (通常需要根据时间序列的长度和漂移速度调整)
- **`gamma` (折扣因子)**: `[0.9, 0.95, 0.99]` (暂未实现)

这为我们未来的超参数调优提供了一个很好的起点。

## 3. 实验执行与评估

### 3.1. 实验执行

我们提供两个版本的实验执行脚本：

- **并行执行 (推荐)**:
  `python project_root/fdsw/main.py --config ...`
- **单线程执行 (用于调试)**:
  `python project_root/fdsw/main_single_thread.py --config ...`

### 3.2. 结果评估与可视化

使用此目录下的专属 `evaluate.py` 脚本来生成结果图表。

**执行命令:**
```bash
python project_root/fdsw/evaluate.py -e "Your-Experiment-Name"
```
脚本会自动从 `results/` 目录加载数据，计算累积遗憾，并生成一张带有置信区间的对比图和一张最终性能总结表，保存在 `figures/` 目录下。

## 4. 已实现的算法

- `UCB1` (基线)
- `ETC` (Epsilon-Greedy, 用于环境验证)
- `UCB_SW` (滑动窗口)
- `UCB_D` (折扣因子)
- `UCB_DSW` (滑动窗口 + 折扣因子)

## 5. 依赖

本模块依赖于 `project_root` 内部的以下模块：
- `data/`
- `drift/`
- `bandits/`
以及可复用的 `evaluate.py` 脚本。 