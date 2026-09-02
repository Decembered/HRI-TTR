# 8×RTX 3090 部署与训练

## 已确认的目标环境

- SSH：`autovla@10.24.116.79 -p 1172`
- Ubuntu 22.04，8 张 RTX 3090 24 GiB，驱动 560.35.03
- 两个 NUMA 节点；GPU 0–3 靠近 NUMA 0，GPU 4–7 靠近 NUMA 1
- GPU 间只有 PCIe，没有 NVLink
- 项目、Python、虚拟环境、依赖缓存和训练产物都放在 `/data`

没有 NVLink 的实际含义是：8 卡之间的梯度同步走 PCIe，跨 GPU 0–3 与 4–7
还会经过 CPU 插槽之间的互连。因此 8 卡加速不会接近理想的 8 倍。首轮先用直接
DDP 测出吞吐，再决定是否增加梯度累积或按 NUMA 绑核；当前脚本不猜测网卡和 PCIe
拓扑参数。

## 固定路径

```text
/data/autovla/projects/HRI-TTR     项目
/data/autovla/envs/hri-ttr        Python 3.11 虚拟环境
/data/autovla/python               uv 管理的 Python
/data/autovla/cache/uv-hri-ttr    uv 依赖缓存
```

系统自带的 Python 3.10 和旧的 Torch 2.0.1 不用于本项目。项目锁定 Python 3.11+
与 Torch 2.7.1 cu126；部署后必须以 `torch.cuda.is_available()`、8 张设备枚举和
NCCL 集合通信实测为准。

## 部署

在本机项目根目录运行：

```bash
bash scripts/deploy_8x3090.sh
```

脚本不会使用 `rsync --delete`，也不会传输本地虚拟环境、训练 runs、token cache
或 checkpoint 二进制。目标 SSH、端口、用户和目录可用以下变量覆盖：

```bash
HRI_TTR_REMOTE_HOST=10.24.116.79 \
HRI_TTR_REMOTE_PORT=1172 \
HRI_TTR_REMOTE_USER=autovla \
bash scripts/deploy_8x3090.sh
```

远端 bootstrap 会在 `/data` 安装用户态 uv、Python 3.11 和独立环境，然后严格按
`uv.lock` 安装。不会修改系统 Python。

该服务器当前 shell 继承的 `127.0.0.1:7898` HTTP 代理会在下载大 wheel 时产生 TLS
EOF。bootstrap 只在执行 `uv sync` 时临时清除代理变量并走已实测可用的直连；不会
修改用户的 shell 配置或其他进程环境。

## 数据准备

完整 7,603 序列数据在服务器上的最终路径尚未确认，不能把猜测路径写进生产配置。
确认后先运行：

```bash
/data/autovla/envs/hri-ttr/bin/hri-ttr data audit --input-dir DATASET_DIR
/data/autovla/envs/hri-ttr/bin/hri-ttr data prepare \
  --input-dir DATASET_DIR --output /data/autovla/datasets/hri-ttr/prepared/SEQ.npz
```

normalizer、split 和 source artifact 都是 checkpoint 身份的一部分。替换其中任何一个，
恢复训练都会被拒绝，避免把不同 token space 混在一起。

## 8 卡训练

生产配置：

- Human：`configs/human_vq/causal_scratch_8x3090.json`
- G1：`configs/g1_vq/causal_scratch_8x3090.json`

两者都使用 FP16 AMP、20 FPS、4 帧一个 token、256 个 code。`batch_size` 是每张卡
的 batch；Human 为 32，G1 为 64。正式训练前根据实测显存和吞吐调整，不能只根据
24 GiB 容量推断。

启动 G1 示例：

```bash
cd /data/autovla/projects/HRI-TTR
bash scripts/launch_8x3090.sh \
  g1-vq configs/g1_vq/causal_scratch_8x3090.json \
  PREPARED.npz NORMALIZER.json configs/data/interx_human_g1_split_v1.json \
  SOURCE_ARTIFACT /data/autovla/projects/HRI-TTR/checkpoints/g1_causal/scratch_8x3090
```

脚本直接使用 `torchrun --standalone --nproc-per-node=8`。启动前会检查所有 GPU；默认
每张至少需要 16,000 MiB 空闲，否则退出，不会杀掉或挤占别人的任务。阈值可以通过
`HRI_TTR_MIN_FREE_MIB` 调整，但必须基于真实模型显存测量。

## 恢复、日志和产物

恢复时把 checkpoint 作为第八个参数：

```bash
bash scripts/launch_8x3090.sh g1-vq CONFIG PREPARED NORMALIZER SPLIT SOURCE \
  OUTPUT_DIR OUTPUT_DIR/interrupted.pt
```

等价的直接 CLI 形式是：

```bash
/data/autovla/envs/hri-ttr/bin/torchrun --standalone --nproc-per-node=8 \
  --module hri_ttr.cli train g1-vq \
  --config CONFIG --prepared PREPARED --normalizer NORMALIZER --split SPLIT \
  --source-artifact SOURCE --output-dir OUTPUT_DIR \
  --resume OUTPUT_DIR/interrupted.pt
```

建议由调用端记录日志：

```bash
mkdir -p runs/logs
bash scripts/launch_8x3090.sh ... 2>&1 | tee runs/logs/g1_$(date +%Y%m%d_%H%M%S).log
```

正常结束产生 `last.pt` 和 `best.pt`；收到 SIGINT/SIGTERM 时产生 `interrupted.pt`。
每次恢复必须继续使用相同的 domain、representation schema、tokenizer config、
normalizer、split 和 source hash。

## 不能从小测试推断的内容

8-rank 两步 smoke 只能证明 CUDA、NCCL、前向、反向、优化器和 checkpoint 通路可用。
它不能证明生产 batch 合适、8 卡扩展效率良好、codebook 不塌缩，也不能代替完整训练的
重建指标和动画检查。
