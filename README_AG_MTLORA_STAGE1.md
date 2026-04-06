# AG-MTLoRA Stage-1

## 1. 方法概述

AG-MTLoRA Stage-1 是在当前 MTLoRA / UniPoRA 代码基础上实现的一个 grouped MTLoRA 版本。

与原始 MTLoRA 的区别：

- 原始 MTLoRA：所有任务共享一套全局 shared TA-LoRA，每个任务各自拥有自己的 TS-LoRA。
- AG-MTLoRA Stage-1：先做 task grouping，再让同一个 group 内的任务共享一套 group-specific TA-LoRA；TS-LoRA 保持逐任务独立不变。
- 当前版本只实现 Stage-1，也就是整网使用同一套 global grouping，不做 stage-wise 或 layer-wise grouping。
- 当前版本强制 partition 约束：每个 task 只能属于一个 group，不能重叠。

本实现参考 `./refer/ETAP` 的 prediction pipeline，而不是直接照搬其最终 search space。

与 ETAP 对齐的部分：

- directed pairwise affinity
- group-by-task proxy
- sampled training groups
- base predictor: affine calibration + spline/ridge 一维回归
- residual predictor: group multi-hot mask + ridge regression
- affinity-score 的多 epoch 在线累计方式

为了适配 grouped TA-LoRA 做的改动：

- ETAP 原始最终 search 不是严格 partition，允许重叠 group。
- AG-MTLoRA Stage-1 训练时必须给每个 task 一个唯一 group，才能路由到唯一的 shared TA-LoRA bank。
- 因此这里保留 ETAP 的 prediction pipeline，但把最后一步改成 partition-constrained exhaustive search。
- 当前实现仍保留显式 warmup 这段工程化适配，再进入 ETAP 风格的在线 affinity-score 累计阶段。

## 2. 代码改动说明

### 新增文件

- `ag_mtlora/config_utils.py`
  - grouping / partition 枚举、group rank 解析、artifact 路径解析。
- `ag_mtlora/stage1.py`
  - Stage-1 prepare 主流程：warmup、多 epoch directed affinity、group proxy、predictor chain、partition search、artifact 导出。
- `scripts/ag_mtlora_stage1_prepare.py`
  - 两步工作流中的 Step-1 入口脚本。
- `configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml`
  - PASCAL 四任务默认 Stage-1 配置。
- `configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud.yaml`
  - NYUD 四任务默认 Stage-1 配置。

### 关键修改

- `config.py`
  - 新增 `AFFINITY_WARMUP_EPOCHS` 和 `AFFINITY_SCORE_EPOCHS`。
  - 保留 `AFFINITY_COLLECT_EPOCHS` 作为兼容别名；如果没有显式设置 `AFFINITY_WARMUP_EPOCHS`，则回退到该旧字段。
- `models/lora.py`
  - 将单一 shared TA-LoRA 扩展为多个 group-specific TA-LoRA banks。
  - TS-LoRA 逻辑保持不变。
- `models/swin_transformer_mtlora.py`
  - 增加 AG 路由所需的 task stream 传递。
  - forward 中实现 `task -> group -> selected TA-LoRA`。
- `utils.py`
  - 支持把 baseline checkpoint 中的单一 shared TA-LoRA 自动复制到每个 group-specific TA bank。
- `README_AG_MTLORA_STAGE1.md`
  - 当前说明文档。

## 3. ETAP 风格 Stage-1 流程

### Step A. warmup + 多 epoch directed affinity

当前 Step-1 的 baseline 生成分为两段：

1. warmup 阶段

- 用 baseline MTLoRA 正常训练 `MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS`
- 这一步不累计 affinity-score
- 结束后保存 `warmup_checkpoint.pth`

2. affinity-score 阶段

- 在 warmup 结束后的同一个 baseline MTLoRA 上继续训练 `MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS`
- 这一步按 batch 在线累计 directed affinity
- 结束后保存 `post_affinity_checkpoint.pth`

这一步与 ETAP 对齐的地方是：

- affinity 来自 baseline 持续训练轨迹
- 从 post-warmup baseline 的第一个 batch 起在线累计
- 累计窗口覆盖多个 epoch，而不是 warmup 后单独 1 个 epoch

主 affinity 定义为：

`A[i->j] = mean_b( g_j^(b) · u_i^(b) / eta_b )`

其中：

- `g_t` 是 task `t` 在 shared TA-LoRA 参数上的梯度。
- `u_t` 是 task `t` 的 pseudo-update。
- `i->j` 表示 task `i` 的更新对 task `j` 的近似影响。

注意：

- 搜索使用的是 directed affinity。
- 对称矩阵 `0.5 * (A + A^T)` 只做可视化导出，不参与搜索。

### Step B. group-by-task proxy

对每个 candidate group `G`，先为 group 内每个 task 单独构造 proxy，而不是先把整个 group 压成一个分数。

对 `t in G`：

- 若 `|G| = 1`，则 `proxy_G[t] = 0`
- 否则 `proxy_G[t] = avg_{s in G, s != t} A[s->t]`

也就是：用 group 内其他成员指向当前 task 的 directed affinity 入边均值，作为这个 task 在该 group 下的 proxy。

### Step C. predictor chain

搜索之前先训练一个 ETAP 风格的 predictor chain。

1. sampled training groups

- 默认策略：`all_singletons + all_pairs + random_higher_order`
- 如果 budget 小于单例组数量，代码会自动保留所有 singleton 组，因为它们是 gain=0 的锚点
- 如果总 candidate groups 不大于 budget，则直接全部使用

2. base predictor

- 输入：group-by-task proxy
- 标签：短程 group 训练相对 singleton 训练的 per-task gain
- 默认实现：affine calibration + `SplineTransformer + Ridge`
- 可选实现：`knn`、`rf`

3. residual predictor

- 输入：group multi-hot mask
- 目标：`gt_gain - base_prediction`
- 默认实现：Ridge regression

4. final prediction

- `initial_predictions = base(proxy)`
- `residual_predictions = residual(mask)`
- `final_predictions = initial_predictions + residual_predictions`

后续 partition search 只读取 `final_predictions`，不会直接使用 raw affinity 或 raw proxy。

### Step D. partition-constrained search

对所有 candidate groups 先生成 group-by-task final predicted gains，然后再做 partition-constrained exhaustive search。

partition `P` 的得分为：

`score(P) = mean_{t in T} final_pred[group_of_P(t)][t]`

tie-break 顺序固定为：

- partition score 更高优先
- group 数更少优先
- group 字典序更小优先

## 4. Grouped TA-LoRA 训练侧实现

训练阶段保持以下约束：

- 每个 group 一套 group-specific TA-LoRA。
- 每个 task 仍保留自己的 TS-LoRA。
- forward 路由为 `task -> group -> selected TA-LoRA`。
- grouping 对整网所有 TA-LoRA 层统一生效。
- AG-MTLoRA 关闭时，原始 MTLoRA 行为保持不变。

baseline checkpoint 初始化规则：

- `warmup_checkpoint.pth` 是 warmup 结束后的中间检查点。
- `post_affinity_checkpoint.pth` 是多 epoch affinity-score 累计结束后的 baseline 检查点。
- Step-2 默认推荐从 `post_affinity_checkpoint.pth` 初始化。
- 加载时会自动把单一 shared TA 权重复制到每个 group-specific TA bank。

## 5. Shared rank 配置

`MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET` 只作为默认自动分配参考，不是硬约束。

支持两种方式：

- 默认自动分配：`GROUP_RANK_ALLOCATION: equal_split`
- 手动覆盖：`GROUP_SHARED_RANKS`

默认自动分配规则：

- 先按 `equal_split` 把 `TOTAL_SHARED_RANK_BUDGET` 平均分给各 group
- 余数按前几个 group 分摊
- 默认分配结果会显式写入 Step-1 生成的 `resolved_agmtlora_config.yaml`

手动覆盖规则：

- `GROUP_SHARED_RANKS` 优先级高于自动分配
- 可以写成一维列表，例如 `[32, 32]`
- 也可以写成按 stage 指定的二维列表，例如 `[[32,32,32,32], [48,48,48,48]]`
- 不要求各 group rank 之和等于 `TOTAL_SHARED_RANK_BUDGET`

如果直接用 `grouping.json` 训练且没有显式设置 `GROUP_SHARED_RANKS`，代码会优先读取 `grouping.json` 中保存的 rank 设置。

## 6. 关键配置项

主要配置位于 `MODEL.AGMTLORA`：

- `ENABLED`
- `STAGE`
- `MAX_GROUPS`
- `TOTAL_SHARED_RANK_BUDGET`
- `GROUP_SHARED_RANKS`
- `GROUP_RANK_ALLOCATION`
- `GROUPING_SOURCE`
- `GROUPING_JSON`
- `AFFINITY_WARMUP_EPOCHS`
- `AFFINITY_SCORE_EPOCHS`
- `AFFINITY_COLLECT_EPOCHS`
- `AFFINITY_SAVE_PATH`
- `GROUPING_SAVE_PATH`
- `SEARCH_OBJECTIVE`
- `PREDICTOR_TRAIN_GROUP_BUDGET`
- `PREDICTOR_TRAIN_GROUP_STRATEGY`
- `PREDICTOR_GROUP_TRAIN_EPOCHS`
- `BASE_PREDICTOR`
- `BASE_PREDICTOR_KWARGS`
- `RESIDUAL_PREDICTOR`
- `RESIDUAL_ALPHA`
- `VISUALIZE_SYMMETRIC_AFFINITY`

兼容规则：

- 推荐新配置使用 `AFFINITY_WARMUP_EPOCHS` 和 `AFFINITY_SCORE_EPOCHS`
- 旧配置如果只写了 `AFFINITY_COLLECT_EPOCHS`，则它会被解释为 warmup epoch 数

## 7. 运行流程

### 环境要求

环境依赖继续沿用仓库根目录 `README.md`。

额外要求：

- 安装 `scikit-learn`
- 准备好 backbone checkpoint，例如 `backbone/swin_tiny_patch4_window7_224.pth`

### Step-1: 生成 affinity / grouping

PASCAL 四任务示例：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

NYUD 四任务示例：

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/nyud/ag_mtlora_stage1_tiny_448_r64_scale4_pertask_nyud.yaml \
  --nyud /path/to/NYUD_MT \
  --tasks semseg,normals,depth,edge \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

Step-1 输出目录形如：

`output/<model_name>/<tag>/ag_mtlora_stage1_prepare/run_<timestamp>/`

### Step-2: 训练 grouped AG-MTLoRA

推荐直接使用 Step-1 自动生成的 `resolved_agmtlora_config.yaml`，并用 `post_affinity_checkpoint.pth` 做初始化。

PASCAL 四任务示例：

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

NYUD 四任务示例：

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --nyud /path/to/NYUD_MT \
  --tasks semseg,normals,depth,edge \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

### 直接指定已有 grouping json 训练

推荐方式：

- 复制一份 AG config
- 将 `MODEL.AGMTLORA.GROUPING_SOURCE` 改成 `fixed_json`
- 将 `MODEL.AGMTLORA.GROUPING_JSON` 改成已有 `grouping.json` 路径
- 如需手动改 rank，再设置 `MODEL.AGMTLORA.GROUP_SHARED_RANKS`
- 正式训练仍推荐从 `post_affinity_checkpoint.pth` 初始化

如果你直接使用 Step-1 生成的 `resolved_agmtlora_config.yaml`，这一步已经自动完成。

## 8. 输出文件说明

Step-1 至少会生成以下 artifacts：

- `affinity.json`
  - 多 epoch 聚合后的 directed affinity 主结果
- `affinity.csv`
  - directed affinity 可读矩阵
- `affinity_epoch_history.json`
- `affinity_epoch_history.csv`
  - 每个 affinity epoch 的 directed affinity 历史
- `affinity_symmetric.json`
- `affinity_symmetric.csv`
  - 对称化 affinity，可视化用途
- `group_proxy.json`
- `group_proxy.csv`
  - 每个 candidate group 对每个 task 的 proxy
- `predictor_train_groups.json`
- `predictor_train_groups.csv`
  - predictor 训练用 groups、singleton loss、ground-truth gains
- `initial_predictions.json`
- `initial_predictions.csv`
  - base predictor 输出
- `residual_predictions.json`
- `residual_predictions.csv`
  - residual predictor 输出
- `final_predictions.json`
- `final_predictions.csv`
  - final predicted gains
- `partition_search_results.json`
- `partition_search_results.csv`
  - 所有 partition 的得分、每 task predicted gain 和排序
- `grouping.json`
  - 最终分组结果
- `resolved_agmtlora_config.yaml`
  - 显式写入 `GROUP_SHARED_RANKS` 和 `GROUPING_JSON` 的训练配置
- `warmup_checkpoint.pth`
  - warmup 结束后的中间 checkpoint
- `post_affinity_checkpoint.pth`
  - Step-2 默认推荐初始化 checkpoint

正式训练日志和 checkpoint 继续使用原始训练目录格式：

`output/<model_name>/<tag>/run_<timestamp>/`

## 9. 最小实验示例

以 PASCAL 四任务为例：

1. 先做 Step-1

```bash
python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/mtlora/tiny_448/pascal/ag_mtlora_stage1_tiny_448_r64_scale4_pertask.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --resume-backbone backbone/swin_tiny_patch4_window7_224.pth
```

2. 再做 Step-2

```bash
python -m torch.distributed.launch --nproc_per_node 1 main.py \
  --cfg /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/resolved_agmtlora_config.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 8 \
  --epochs 300 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --resume /abs/path/to/output/ag_mtlora_stage1_prepare/run_xxx/post_affinity_checkpoint.pth
```

## 10. 注意事项与已知限制

- 当前只实现 Stage-1。
- 当前 grouping 是 global grouping，不是 stage-wise / layer-wise grouping。
- 当前实现贴合的是 ETAP 的 prediction pipeline 和多 epoch 在线 affinity-score 累计方式，不是 ETAP 原始最终 search space。
- 当前最终 search 强制 partition 约束，这是为了适配 grouped TA-LoRA 的唯一归组需求。
- 默认配置和当前实现主要面向 4-task PASCAL / NYUD 小任务数场景，因此 partition search 采用穷举。
- 当前 AG 路由重点覆盖 QKV、Proj、FC1、FC2 等 TA-LoRA 注入位置；默认配置中 `MODEL.MTLORA.DOWNSAMPLER_ENABLED=False`，这也是当前推荐设置。
- 如果后续扩展到 Stage-2 或 stage-wise grouping，优先修改 `models/swin_transformer_mtlora.py` 中的 task routing 和 `config.py` 中的 group rank / grouping 解析逻辑。
