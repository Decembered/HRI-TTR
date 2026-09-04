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
HRI-Datasets/filtered
├── humanl3d: HumanML3D Human/G1 pickle pairs
└── interx:   Inter-X Human/G1 retarget pairs
             │  same basename, same native timeline
             ▼
      20 Hz resampling + InteractionWorld coordinate conversion
             │
             ├── Human joints22 ──> Human frame-0 canonicalization
             │                       └─> Human 262D ──> Human causal VQ ──> Human tokens
             │                            独立 normalizer/encoder/codebook/decoder
             └── G1 root+dof29 ──> G1 frame-0 EpisodeFrame canonicalization
                                  └─> G1 75D ────> G1 causal VQ ────> G1 tokens
                                       独立 normalizer/encoder/codebook/decoder

Human/G1 world pose ──> InteractionWorld [x,z,yaw] space 序列

未来 Stage 3：按同一时间轴对齐的 Human tokens + G1 tokens ──> 跨 embodiment 语义映射
```

Bone-Seed 已从正式数据路径移除。数据层先冻结：filtered source、manifest、20 Hz
重采样、每个 embodiment 自己的第 0 帧 canonicalization、262D/75D 表示和各自
train-only normalizer 都固定后，才启动两套 tokenizer 训练。这里的“同一对动作”只
表示 Human 和 G1 时间对齐，并不表示它们共享 feature space 或 VQ codebook。

Human 与 G1 只共享无状态的基础组件，例如严格左填充的一维卷积类。两侧不会共享
normalizer、模型实例、codebook 参数或 checkpoint。虽然两侧 token 数字都在
`[0,255]`，但 `kind=human` 与 `kind=g1` 是两个不同的语义域；数字相同不代表动作
含义相同。

## 目录责任

- `data`：filtered paired 文件、固定 split、20 Hz corpus、尾帧补齐和 train-only
  normalizer。
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

训练的 seed 在每个 torchrun rank 创建模型前设置，保证 scratch encoder、decoder 和
初始 codebook 一致；训练循环也会重新设置 seed，resume 时再由 checkpoint 恢复 RNG。
EMA codebook 的 count/sum 在 DDP 中先做全局 all-reduce，dead-code 替换候选由一个
确定性 owner 广播给所有 rank，因此 codebook 不再由各卡的局部 batch 漂移。

## 训练顺序

1. 读取 filtered 的 HumanML3D/Inter-X paired pickle，固定 source manifest 和 group
   split；只在这个冻结输入上做 20 Hz、坐标转换、两侧独立 frame-0 canonicalization。
2. 写出未归一化的 Human 262D、G1 75D、world `[x,z,yaw]` 和质量审计；只用 train
   split 的 Human/G1 各自统计拟合两个 normalizer。padding 只在取 window 时复制，并
   通过 mask 排除 loss、perplexity 和 EMA。
3. 用同一 corpus 分别启动 Human 和 G1 两个训练进程组。每次只把一个 domain 的
   normalized feature 送进它自己的 causal VQ；20 FPS 输入按 4 帧一组，输出 5 Hz
   token。validation 不混入 train window，checkpoint 绑定 domain/config/normalizer/
   split/source hash。
4. 两侧 reconstruction、causality、codebook 使用率都达标后，Stage 3 才读取同一
   paired manifest 的 Human token/G1 token 序列训练跨 embodiment 映射；Stage 3 不
   回头共享或重训 Stage 2 的 VQ codebook。

## 已确认、待训练与基线

已确认的是代码契约和测试覆盖的确定性行为，例如 Human 金标准、G1 表示 roundtrip、
坐标转换、因果前缀一致性和 tiny smoke 流程。`checkpoints/baselines` 只承载旧模型对照，
不能说明新因果模型质量。`checkpoints/human_causal` 与 `checkpoints/g1_causal` 当前只是
训练输出位置；在完整训练与评估完成前，不应声称 Stage 2 的重建质量已经通过。
