# 动作与空间表示协议

## 共同约定

- 时间：`data prepare` 将来源 FPS 确定性重采样为 20 FPS；每四帧形成一个 5 Hz
  motion token。
- InteractionWorld 与 EpisodeFrame：Y-up。
- 外部 G1/SONIC：Z-up。
- 所有公开四元数边界：标量在最后的 `xyzw`。
- 尾部不足四帧时复制最后一帧；真实帧由 mask 标记，补齐帧不进入 loss 或指标。
- `human_space[t]` 和 `g1_space[t]` 都是 `[x_m, z_m, yaw_rad]`，与 motion feature
  分开保存。
- 原始 pair 默认使用不含 object array 的严格 NPZ；旧 `.pkl` 只有显式确认可信本地
  来源后才允许读取。prepared metadata 保留 `source_fps`、`target_fps` 和
  `source_format`，其中 `trusted_pickle` 表示调用者做过明确的 pickle 信任选择。

## Human 262D

Human 输入是 `[T,22,3]` 的世界关节位置。兼容路径先将地面高度归零，去掉初始 root
的 XZ 平移，再把初始朝向旋到统一方向。262D 每帧布局是：

| slice | 维度 | 含义 |
|---|---:|---|
| `0:66` | 66 | 22 个归一化关节位置，米 |
| `66:132` | 66 | 相邻帧关节位移，米/帧 |
| `132:258` | 126 | 除 root 外 21 个关节的 6D 旋转 |
| `258:262` | 4 | 双脚四个接触标记 |

最后一帧的位移和接触沿用最后一个可计算值。这是历史 TTR/InterGen 行为，不应把
它误解为物理速度。金标准测试锁定了旧 TTR commit
`9b7e395f740a68cbd30c027b4952dedb0ebf8b6d` 上合成样例的逐元素结果。

完整 Human space 序列先把未归一化的关节世界姿态表达进同一个 G1 EpisodeFrame，
再计算每帧 root 的 XZ 与朝向。因此它既能描述双方初始交互布局，也能在后续辅助
预训练中提供完整轨迹监督。space 不送入 Human VQ。

## G1 75D V2

EpisodeFrame 的原点是 G1 第 0 帧 root 的地面投影；+Y 向上；+Z 沿 G1 第 0 帧
水平朝向。75D 布局的权威 schema ID 来自代码常量
`g1_canonical_75d_v2`：

| slice | 维度 | 名称 | 单位 |
|---|---:|---|---|
| `0:3` | 3 | `root_pos_episode_m` | m |
| `3:9` | 6 | `root_rot6d_episode` | unitless |
| `9:38` | 29 | `dof_pos_rad` | rad |
| `38:41` | 3 | `root_linear_vel_local_m_s` | m/s |
| `41:44` | 3 | `root_angular_vel_local_rad_s` | rad/s |
| `44:73` | 29 | `dof_vel_rad_s` | rad/s |
| `73:75` | 2 | `foot_contact_lr` | binary |

这里有一个容易混淆的点：root 位置和 6D 朝向是解码时的权威字段。局部线速度、
局部角速度和关节速度是辅助学习目标，不用于覆盖显式 root。这样表示自身的 roundtrip
不依赖长时间积分，积分只作为一致性诊断。

G1 space 也是 EpisodeFrame 中逐帧的 XZ 与 yaw，独立于 75D VQ 通道保存。75D
仍包含相对 EpisodeFrame 的完整 root 轨迹，这是为了让 motion token 能表达每个
四帧动作块的身体运动；space 通道用于布局与辅助监督，两者作用不同。

## SONIC 恢复

75D decode 加 EpisodeFrame anchor 先恢复 InteractionWorld 的 root 与 29DoF，再变换
回 G1 Z-up。导出字段为：

- `root_trans_offset`：Z-up root 位置。
- `root_rot`：Z-up、`xyzw` 四元数。
- `dof`：固定 G1 29DoF 顺序的关节角。
- `pose_aa`：root 旋转向量加 29 个固定 axis/order 的关节旋转向量。

该导出只是数据格式转换，不是 SONIC 控制策略，也不能代替真机安全检查。
