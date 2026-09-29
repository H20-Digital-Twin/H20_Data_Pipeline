# SMPL-X → Robot Retargeting

本目录是从 `/data/GMR-master/scripts/robot_dataset_stage2` 整理出的独立生产
副本，将本工程生成的 canonical SMPL-X 30 FPS 动作重定向到 6 种机器人，
并精确抽取相同时间点的 15 FPS 轨迹。

## 保留内容

- `pipeline/`：单动作生成、515 条批处理、MuJoCo schema、轨迹后处理与验证；
- `general_motion_retargeting/`：生产流程实际 import 的 GMR/Mink 求解代码；
- `configs/`：6 种机器人的正式配置；
- `general_motion_retargeting/ik_configs/`：6 份 SMPL-X IK 配置；
- `assets/`：机器人资产与 SMPL-X body model 的软链接。

没有复制 audit、pilot、测试 shell、预览器、GUI 回放器、copy/debug 变体或
Python 缓存。

## 支持的机器人

```text
unitree_g1
booster_t1
stanford_toddy
fourier_n1
hightorque_hi
engineai_pm01
```

## 软链接

六个机器人资产目录分别链接到 `/data/GMR-master/assets/<robot_id>`，因此
XML、URDF 与 XML 引用的 mesh 都可用，但不会重复占用空间。

```text
assets/body_models/smplx -> /data/smpl_models/smplx
```

若这些源目录移动，必须重新建立对应软链接。生产 Python 代码本身不 import
`/data/GMR-master` 中的模块。

## 环境

已验证环境：

```text
/home/dell/anaconda3/envs/myenv/bin/python
```

主要依赖为 NumPy、SciPy、PyTorch、SMPL-X、MuJoCo、Mink、DAQP、PyYAML、
Matplotlib、Rich 和 tqdm。依赖名称同时记录在项目根目录 `requirements.txt`。

## 默认输入输出

```text
候选清单: /data/h2o_data_engine/selection/artifacts/v4/candidate_v4.jsonl
canonical: /data/h2o_data_engine/work/canonical/motions/<motion_id>/source/
PLY:       /data/h2o_data_engine/work/ply/<motion_id>/
输出:      /data/h2o_data_engine/work/robots/
```

如果 canonical、PLY 或输出位于外部数据盘，修改对应
`configs/robot_generation_*.yaml` 的 `paths` 即可。PLY 只用于检查机器人
15 FPS 与视觉帧是否对齐，不参与 IK 求解。

## 单动作生成

从项目根目录运行：

```bash
cd /data/h2o_data_engine
MPLCONFIGDIR=/tmp/robot-retarget-mpl \
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_one.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --motion-id <motion_id> \
  --resume
```

将配置文件换成其他机器人即可。仅在明确需要覆盖已有单动作结果时使用
`--overwrite`。

## 单机器人全量 515 条

```bash
MPLCONFIGDIR=/tmp/robot-retarget-mpl \
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_batch.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --all
```

依次对 6 份配置运行即可获得 6 种机器人。批处理内部对每条动作使用 resume
语义：完整且身份一致的结果验证后跳过；失败记录写入输出根目录 `failures/`。

## 验证已有结果

验证单个输出目录的全部 MuJoCo 帧：

```bash
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/validate_output.py \
  --robot-dir work/robots/motions/<motion_id>/robots/unitree_g1 \
  --all-mujoco-frames \
  --write-report
```

验证某机器人清单中的全部已有结果：

```bash
/home/dell/anaconda3/envs/myenv/bin/python \
  robot_retargeting/pipeline/generate_batch.py \
  --config robot_retargeting/configs/robot_generation_g1.yaml \
  --all \
  --validate-only \
  --all-mujoco-frames
```

## 输出

```text
work/robots/
  robot_schemas/<robot_id>.json
  manifests/robot_generation_inventory_<robot_id>.jsonl
  manifests/robot_generation_summary_<robot_id>.json
  motions/<motion_id>/robots/<robot_id>/
    trajectory_30fps.npz
    trajectory_15fps.npz
    diagnostics.npz
    metadata.json
    quality.json
    validation.json
```

15 FPS 数组严格等于 30 FPS 数组的 `0, 2, 4, ...` 帧，与 Avatar PLY、RGB、
TIFF Depth 和 Mask 使用同一组视觉帧编号。
