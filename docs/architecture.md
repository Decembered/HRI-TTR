# HRI-TTR 架构

## 本轮边界

HRI-TTR 是一个独立 Python 项目。运行时只从 `hri_ttr` 包导入代码，不依赖
Think-Then-React 或旧 `src/hri`。旧仓库只用于三件事：核对行为、生成金标准、
导入明确声明的基线权重。

本轮已经实现数据边界、几何变换、Human/G1 表示、两套因果 VQ、训练与评估入口、
token cache、SONIC 文件导出，以及未来语言训练的类型契约。本轮没有训练出可用的
新模型，也没有实现语言模型、在线学生或真机控制器。

## 数据流

```text
paired raw motion
├── Human joints22 ──> Human 262D ──> Human causal VQ ──> Human tokens
│                                      独立 encoder/codebook/decoder
└── G1 root+dof29 ──> G1 75D ─────> G1 causal VQ ────> G1 tokens
                                       独立 encoder/codebook/decoder

Human/G1 world pose ──> EpisodeFrame 中各自的 x/z/yaw space 序列

未来 Stage 3/4：Human tokens + G1 history + semantic ──> G1 target token
```

Human 与 G1 只共享无状态的基础组件，例如严格左填充的一维卷积类。两侧不会共享
normalizer、模型实例、codebook 参数或 checkpoint。虽然两侧 token 数字都在
`[0,255]`，但 `kind=human` 与 `kind=g1` 是两个不同的语义域；数字相同不代表动作
含义相同。

## 目录责任

- `data`：成对文件的严格读取、固定 split、尾帧补齐。
- `geometry`：Y-up/Z-up、`xyzw` 四元数、旋转和重采样。
- `representations/human`：官方兼容的 22J、262D、Human normalizer。
- `representations/g1`：EpisodeFrame、75D、G1 normalizer、接触检测。
- `tokenizers/human`、`tokenizers/g1`：两套独立因果 VQ。
- `training`：单卡/DDP、AMP、恢复和 checkpoint 元数据。
- `evaluation`：重建、codebook、因果一致性和图像。
- `language`：仅定义未来 teacher/student 的时间轴类型。
- `runtime`：仅保存在线因果前缀；不发机器人控制命令。
- `sonic`：把已解码 G1 动作转换为 SONIC 需要的离线数组。

训练续跑使用 `hri-ttr train human-vq|g1-vq --resume CHECKPOINT --max-steps N`。
输出目录可以沿用原目录，也可以用 `--output-dir` 指向新目录；两种情况都会先核对模型
类型、表示协议、tokenizer 配置以及 normalizer/split/source 哈希，任一不一致即拒绝加载。
通过后恢复训练步数、优化器、AMP scaler 和随机数状态；续跑不会再次应用 warm-start 权重。

## 已确认、待训练与基线

已确认的是代码契约和测试覆盖的确定性行为，例如 Human 金标准、G1 表示 roundtrip、
坐标转换、因果前缀一致性和 tiny smoke 流程。`checkpoints/baselines` 只承载旧模型对照，
不能说明新因果模型质量。`checkpoints/human_causal` 与 `checkpoints/g1_causal` 当前只是
训练输出位置；在完整训练与评估完成前，不应声称 Stage 2 的重建质量已经通过。
