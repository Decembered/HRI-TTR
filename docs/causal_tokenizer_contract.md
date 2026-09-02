# 严格因果 Motion Tokenizer 契约

## 时间定义

输入是 20 FPS，`frames_per_token=4`，所以 token rate 是 5 Hz。token `k` 对应帧：

```text
[4k, 4k+1, 4k+2, 4k+3]
```

生成 token `k` 时可以读取到帧 `4k+3`，但不能读取更晚帧。这是“四帧块级因果”，
不是每来一帧就立即生成 token；在线端天然需要等满约 200 ms 的四帧块。

decoder 的第 `k` 个输出块只能依赖 `0..k` token。修改未来帧或未来 token 时，过去
latent、token ID 和已解码帧必须保持不变。

## 网络限制

- 时间卷积只允许左侧 padding。
- 禁止 BatchNorm、GroupNorm 等会跨时间汇总当前整段信息的归一化。
- Human 和 G1 拥有不同 encoder、decoder、EMA quantizer 与 codebook。
- codebook 固定 256 项，合法 ID 为 `[0,255]`。
- padding token 不更新 EMA，不计 perplexity；reconstruction loss 和指标由帧 mask
  排除复制的尾帧。

## 三种执行方式必须一致

1. 离线整段：一次输入完整已观察序列。
2. prefix：反复输入从第 0 帧到当前边界的前缀。
3. streaming：用 tokenizer state 逐块追加新帧。

对同一已观察前缀，三条路径要求 token ID 变化数为 0，latent 和 decoded motion 的
最大绝对差不超过 `1e-6`。state 是单次所有权对象；已消费的 state 或另一 tokenizer
创建的 state 不能复用。

当前 streaming 实现为了先保证一致性，会在 state 中保留完整历史前缀并重新计算。
它满足因果正确性，但还不是低内存、低算力的增量卷积缓存。后续部署优化可以把历史
替换为每层有限 receptive-field state，但必须继续通过同一一致性测试。

## 尾帧

不足四帧的尾部复制最后一帧补齐。帧 mask 区分真实帧和复制帧。只要一个四帧组含有
padding，该组不参与 codebook EMA/perplexity；真实帧的重建损失仍由逐帧 mask 独立
控制。不得静默删除原始尾帧。
