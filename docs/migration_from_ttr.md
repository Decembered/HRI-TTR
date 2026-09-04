# 从 Think-Then-React 迁移

## 迁移原则

新项目不是给旧 TTR 增加一层接口，而是独立重写。运行时不得导入旧源码。迁移采用
“固定输入、固定输出、哈希金标准”的方式：先用已知版本的旧代码生成期望结果，再让
新实现逐元素匹配。这样能迁移数学行为，而不会把旧目录结构和隐式依赖带进来。

Human 262D 的合成金标准绑定旧 TTR commit
`9b7e395f740a68cbd30c027b4952dedb0ebf8b6d`。G1 坐标、四元数、75D 和 SONIC
恢复由新项目自己的 roundtrip 测试锁定。

## 权重分类

| 位置 | 用途 | 能否代表新模型质量 |
|---|---|---|
| `checkpoints/baselines/ttr_human_noncausal` | 旧 Human non-causal 对照/显式导入源 | 不能 |
| `checkpoints/baselines/g1_73d_noncausal` | 旧 73D G1 对照 | 不能 |
| `checkpoints/human_causal` | 新 Human 因果训练输出 | 完整训练评估后才可以 |
| `checkpoints/g1_causal` | 新 75D G1 因果训练输出 | 完整训练评估后才可以 |

G1 的表示宽度和因果结构都改变，因此从零训练，不拼接 73D 权重。Human 有两条明确
路径：完全从零训练；或对官方权重做逐 key、逐 shape 的显式导入报告后 warm-start。
“文件存在”不等于“兼容导入成功”。每个 checkpoint 与 cache 都必须绑定配置、
schema、normalizer、split 和来源权重哈希。

## Stage 0 数据导入安全

新项目默认读取同名配对的 `*_actor.npz` / `*_reactor.npz`。NPZ 必须只含协议规定的
数值字段，读取时禁用 object array。旧 Stage 0 语料的 `.pkl` 仍可迁移，但 pickle
在读取时可能执行代码，因此默认拒绝。只有确认文件来自可信本地 Stage 0 语料时，才可
显式执行：

```sh
hri-ttr data audit --input-dir /trusted/stage0 --allow-trusted-pickle
hri-ttr data prepare --input-dir /trusted/stage0 --output prepared.npz \
  --allow-trusted-pickle
```

该标志不是“忽略检查”：反序列化后仍执行完整 shape、数值、四元数、FPS 和时间轴
校验。输出的 `source_format=trusted_pickle` 会保留这次信任决定，避免后续把它误报
成安全 NPZ 来源。下载文件或来源不明文件不得使用该标志。

## Token 不可混用

换成因果 encoder/codebook 后，即使 token ID 仍写作 `0..255`，其含义也已经改变。
以下内容都不能混用：

- non-causal Human cache 与新 Human causal checkpoint；
- 73D G1 cache 与新 75D G1 checkpoint；
- 不同 warm-start/scratch run 之间的 cache；
- Human 与 G1 的数值相同 token ID。

进入 Stage 3 前，必须先选定最终 Human/G1 checkpoint，再重新生成双方 cache 和 motion
embedding。旧 TTR LM embedding 不能直接解释新 token space。

## Stage 3/4 边界

本项目当前只实现下面两个可校验的时间轴契约，没有实现 LM：

```text
offline teacher: H[0:k] + G[0:k-1] + semantic -> G[k]
future online student: H[0:k] + G[0:k] + semantic -> G[k+1]
```

teacher 用同一时刻 Human 前缀预测当前 G1；student 在已经观察当前双方动作后预测下一
个四帧 G1 块。这个区别必须保留，不能只把模型名字换掉。在线 buffer 当前只负责构造
第二种前缀，不包含蒸馏、语言模型、SONIC 控制或网络通信。

## 当前不能声称的结果

本轮 smoke test 使用随机初始化或 tiny training，只验证路径可执行，不验证实际动作
质量。Human/G1 新因果模型的重建误差、codebook 使用率、四类 GT/reconstruction 动画
和真机跟踪效果，必须在后续完整训练与独立评估后报告。
