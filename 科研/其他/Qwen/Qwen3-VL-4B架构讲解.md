---
title: Qwen3-VL-4B 架构讲解
date: 2026-07-15
updated: 2026-07-29
tags:
  - 科研/其他
  - 深度学习/多模态大模型
  - LLM/Qwen
---

为了让您完整理解整个工作机制，下面基于官方技术报告、Qwen3-VL-4B 配置和 Transformers 实现，梳理 **Qwen3-VL-4B** 处理 `Text + Image` 的端到端技术流程，并配备对应的系统架构流向图。

本文以“一段文本 + 一张静态图片”的单轮推理为例。多图片和批量输入的基本机制相同，只是在视觉 Token 数量和序列组装方式上有所扩展。

关于各层可能保留的信息、DeepStack 的逐层影响，以及哪些判断只是机制性推断，参见 [[Qwen隐藏层信息分布]]。

先看整体结构。

从模块划分看，模型由视觉编码器、MLP 视觉—语言 Merger 和 Qwen3 语言模型
组成。DeepStack 不是第四个串行模块，而是从视觉编码器中间层通往语言模型
前几层的三条残差支路。

端到端流程可以概括为：

```text
文本 Token Embedding + Final Vision Merger 特征
→ 组装基础统一图文序列
→ Decoder Layer 1 → 注入 DeepStack 特征 1
→ Decoder Layer 2 → 注入 DeepStack 特征 2
→ Decoder Layer 3 → 注入 DeepStack 特征 3
→ Decoder Layers 4–36
→ Final RMSNorm → LM Head → 自回归生成
```

其中，视觉编码器会提前计算主视觉特征和三组 DeepStack 特征，但三组
DeepStack 特征不是在语言模型入口处一次性融合，而是在前三个 Decoder Layer
输出后依次加入。本文使用自然语言中的一基编号“第 1 层～第 36 层”；代码中的
对应索引为 `layer_idx=0～35`。

---

# 一、端到端详细技术流程

## 第一阶段：输入准备与预处理（Preprocessing Phase）

### 1. 输入接收

用户提供：

- 一段文本 $T$，例如“找出图中所有红色方框”；
- 一张原始图像 $I$，尺寸为：

$$
I \in \mathbb{R}^{H \times W \times 3}
$$

实际使用时，图片通常通过结构化消息传入：

```python
{
    "role": "user",
    "content": [
        {"type": "image", "url": image_url},
        {"type": "text", "text": "找出图中所有红色方框"}
    ]
}
```

上例采用当前官方模型卡展示的 URL 输入形式。若直接传入 PIL Image、Tensor 或本地文件，字段形式取决于具体的 Transformers 与 Processor 版本。这里的 `<image>` 主要是概念性表示，真正的视觉特殊 Token 通常由 `Processor` 和 Chat Template 自动生成。

### 2. 图像动态缩放（Dynamic Resolution）

Qwen3-VL 不会先把图片切成多个独立的大图块，也不是选择某种固定的“网格布局”。它首先在尽量保持原始宽高比的前提下，将图像缩放到尺寸 $H' \times W'$。由于高度和宽度会被取整到 32 的倍数，缩放后的宽高比可能与原图存在少量偏差。

缩放结果需要满足：

$$
H' \bmod 32 = 0, \qquad W' \bmod 32 = 0
$$

其中：
$$
32=\text{patch\_size}\times\text{spatial\_merge\_size}
=16\times2
$$

同时，处理器会根据配置的最小和最大像素预算，避免图像过小或生成过多视觉 Token。

### 3. 图像 Patch 化

Qwen3-VL-4B 的视觉参数为：

- 空间 Patch 大小：$16 \times 16$；
- 时间 Patch 大小：2；
- 空间合并比例：$2 \times 2$。

对于静态图片，当前处理实现会在时间维度形成长度为 2 的时间 Patch，等效于
使用两个相同帧。因此，每个展平后的输入 Patch 包含
$3 \times 2 \times 16 \times 16 = 1536$ 个数值。

定义：
$$
g_h=\frac{H'}{16},\qquad
g_w=\frac{W'}{16},\qquad
g_t=1
$$

处理器输出网格信息：
$$
\text{image\_grid\_thw}=(g_t,g_h,g_w)
=\left(1,\frac{H'}{16},\frac{W'}{16}\right)
$$

空间合并之前的 Patch 数量为：
$$
N_{\text{patch}} = g_t g_h g_w
$$

对于单张图片，可以将处理后的视觉输入概念性地表示为：
$$
\text{pixel\_values}
\in\mathbb{R}^{N_{\text{patch}}\times1536}
$$

### 4. 图像占位符生成

视觉编码器后续会把每个 $2 \times 2$ 的空间 Patch 合并为一个视觉 Token，因此进入语言模型的视觉 Token 数量是：
$$
N_{\text{visual}}
=\frac{N_{\text{patch}}}{2^2}
=\frac{N_{\text{patch}}}{4}
$$

对于静态图片：
$$
N_{\text{visual}}
=\frac{H'}{32}\times\frac{W'}{32}
$$

例如，对于缩放后的 $1024 \times 1024$ 图片：

$$
N_{\text{patch}} = 64 \times 64 = 4096
$$

$$
N_{\text{visual}} = 32 \times 32 = 1024
$$

处理器会在图片对应位置生成如下特殊 Token 序列：

```text
<|vision_start|>
<|image_pad|> × N_visual
<|vision_end|>
```

其中 `<|image_pad|>` 的数量必须与视觉 Merger 最终产生的视觉向量数量完全一致。

---

## 第二阶段：文本嵌入与视觉特征提取（Embedding & Vision Encoding Phase）

### 1. 文本路径（Text Embedding）

结构化消息经过 Processor、Chat Template 与 Tokenizer 协同处理，得到：

```text
input_ids
attention_mask
mm_token_type_ids
```

其中：

- 普通文本 Token 的类型为文本；
- `<|image_pad|>` 对应图片位置；
- `mm_token_type_ids` 用于区分文本、图片和视频位置。

所有 Token ID 首先通过语言模型的词嵌入表，得到：
$$
E_{\text{input}}
\in\mathbb{R}^{B\times L\times2560}
$$

这里：

- $B$ 是 Batch Size；
- $L$ 是包含文本、视觉占位符和特殊 Token 的总序列长度；
- 2560 是 Qwen3-VL-4B 的语言模型隐藏维度。

此时 `<|image_pad|>` 位置暂时仍是普通词嵌入向量，后续会被真实视觉特征替换。

### 2. 图像路径（SigLIP-2-based Vision Encoder）

Qwen3-VL 使用 SigLIP-2 架构作为视觉编码器，以官方预训练检查点初始化，并
针对动态输入分辨率继续训练。Qwen3-VL-4B 检查点中的具体视觉配置如下。

其主要配置为：

- Transformer Block 数量：24；
- 视觉隐藏维度：1024；
- 注意力头数量：16；
- FFN 中间维度：4096；
- Patch Size：16；
- Temporal Patch Size：2。

### 3. Patch Embedding

每个展平 Patch 首先被恢复为：
$$
[3, 2, 16, 16]
$$

然后经过一个三维卷积：

```text
Conv3D
kernel_size = (2, 16, 16)
stride      = (2, 16, 16)
```

映射关系为：
$$
3\times2\times16\times16
\longrightarrow 1024
$$

因此 Patch Embedding 的输出为：
$$
X_0 \in \mathbb{R}^{N_{\text{patch}} \times 1024}
$$

### 4. 视觉位置编码

Patch Embedding 会加入：

- 根据输入分辨率插值得到的绝对位置嵌入；
- 用于表达二维空间关系的视觉 RoPE。

随后，视觉 Token 依次进入 24 个视觉 Transformer Block。

官方实现中，每张图片内部执行的是非因果视觉自注意力，不是原文所述的固定局部 Window Attention。

### 5. 多层视觉特征提取

DeepStack 会提取以下视觉层的输出：

- 第 6 个视觉 Block；
- 第 12 个视觉 Block；
- 第 18 个视觉 Block。

在配置文件中对应零基索引：

```text
deepstack_visual_indexes = [5, 11, 17]
```

此外，第 24 个视觉 Block 的最终输出作为主视觉特征。

可以表示为：
$$
X^{(6)}, \quad X^{(12)}, \quad X^{(18)}, \quad X^{(24)}
$$

它们的基本形状均为：
$$
N_{\text{patch}} \times 1024
$$

这些层提供不同深度的视觉表示。通常可以把较早层理解为保留了更多局部信息，把较深层理解为包含更抽象的表示，但不能简单断言某一层只负责 OCR、边缘或物体语义。

---

## 第三阶段：视觉合并与序列组装（Visual Merging & Sequence Assembly）

### 1. 主视觉 Merger

第 24 个视觉 Block 的输出进入主 Visual Merger。

当前实现中的主 Visual Merger 依次执行：

1. 对每个 1024 维视觉 Token 做 LayerNorm；
2. 将空间上相邻的 $2 \times 2$ 个 Token 重排并组合为一个 4096 维向量；
3. 通过两层 MLP 映射到语言模型的 2560 维空间。

其中，$2 \times 2$ 组合后的维度为：

$$
4 \times 1024 = 4096
$$

两层 MLP 的映射为：
$$
4096
\longrightarrow4096
\longrightarrow2560
$$

中间使用 GELU 激活函数。

因此主视觉特征的变化为：
$$
X^{(24)} \in \mathbb{R}^{N_{\text{patch}} \times 1024} \xrightarrow{2 \times 2 \text{ Merger}} V_{\text{main}} \in \mathbb{R}^{N_{\text{visual}} \times 2560}
$$

其中：
$$
N_{\text{visual}} = \frac{N_{\text{patch}}}{4}
$$

Visual Merger 同时完成：

- 视觉序列长度压缩；
- 视觉维度到语言模型维度的映射。

因此，实现中不需要再单独设置一个 `1152 → 3072` 的 Projector。

### 2. DeepStack 专用 Merger

第 6、12、18 个视觉 Block 的输出分别进入三个独立的 DeepStack Merger：
$$
\begin{aligned}
X^{(6)} &\longrightarrow V_{\text{deep}}^{(1)} \\
X^{(12)} &\longrightarrow V_{\text{deep}}^{(2)} \\
X^{(18)} &\longrightarrow V_{\text{deep}}^{(3)}
\end{aligned}
$$

每个专用 Merger 都会进行 $2 \times 2$ 空间合并和维度映射，因此输出形状均为：
$$
V_{\text{deep}}^{(i)}
\in\mathbb{R}^{N_{\text{visual}}\times2560}
$$

与主 Merger 的归一化顺序不同，当前 DeepStack Merger 使用的是：

```text
2×2 Token 组合为 4096 维
→ LayerNorm(4096)
→ Linear(4096 → 4096)
→ GELU
→ Linear(4096 → 2560)
```

DeepStack 特征不会保持为原来的 $N_{\text{patch}}$ 长度，否则它们无法与语言模型中的视觉 Token 位置逐一对应。

### 3. 视觉占位符替换

模型根据 `input_ids` 找出所有 `<|image_pad|>` 位置，构造视觉位置掩码：
$$
M_{\text{visual}} \in \{0, 1\}^{B \times L}
$$

随后使用主视觉特征 $V_{\text{main}}$ 替换这些位置原有的 Token Embedding。

其过程可以写成：
$$
E_{\text{unified}}
=
\operatorname{Replace}
\left(
E_{\text{input}},
M_{\text{visual}},
V_{\text{main}}
\right)
$$

最终得到统一多模态序列：
$$
E_{\text{unified}}
\in\mathbb{R}^{B\times L\times2560}
$$

这里不是把全部视觉 Token 简单追加到文本尾部，而是在 Chat Template 指定的图片位置，用视觉向量替换 `<|image_pad|>` 的词嵌入。

需要特别区分：此时的 $E_{\text{unified}}$ 只包含主 Visual Merger 的输出。
三个 DeepStack Merger 的输出虽然已经计算完成，但尚未写入统一序列；它们会
在随后三个 Decoder Layer 的输出处依次残差加入。

---

## 第四阶段：语言模型前 3 层与 DeepStack 交错注入（Interleaved DeepStack Injection）

Qwen3-VL-4B 的语言模型包含 36 个 Decoder Layer。

主视觉特征 $V_{\text{main}}$ 已经位于语言模型输入序列中。三个 DeepStack
特征也已由视觉编码器提前计算完成，但它们不会在语言模型入口处一次性加入。
语言模型会先执行一个 Decoder Layer，再把对应的 DeepStack 特征残差加入
视觉 Token 位置：前三个 Decoder Layer 与三次 DeepStack 注入交错进行。

因此，不能把这一阶段理解成“先在语言模型外完成全部 DeepStack 融合，再开始
36 层语言推理”。更准确的流程是：

```text
Decoder Layer 1 → 注入 DeepStack 特征 1
→ Decoder Layer 2 → 注入 DeepStack 特征 2
→ Decoder Layer 3 → 注入 DeepStack 特征 3
→ Decoder Layers 4–36 继续推理
```

### 1. 第一层注入

统一序列首先经过第 1 个 Decoder Layer：
$$
H^{(1)}
=
\operatorname{DecoderLayer}_1(E_{\text{unified}})
$$

随后只在视觉 Token 位置加入第 6 个视觉 Block 对应的特征：
$$
H^{(1)}_{\text{visual}}
\leftarrow
H^{(1)}_{\text{visual}}
+
V_{\text{deep}}^{(1)}
$$

### 2. 第二层注入
$$
\begin{aligned}
H^{(2)} &= \operatorname{DecoderLayer}_2(H^{(1)}) \\
H^{(2)}_{\text{visual}} &\leftarrow H^{(2)}_{\text{visual}} + V_{\text{deep}}^{(2)}
\end{aligned}
$$

### 3. 第三层注入
$$
\begin{aligned}
H^{(3)} &= \operatorname{DecoderLayer}_3(H^{(2)}) \\
H^{(3)}_{\text{visual}} &\leftarrow H^{(3)}_{\text{visual}} + V_{\text{deep}}^{(3)}
\end{aligned}
$$

### 4. 后续解码层

注入完成后，序列继续通过剩余的第 4 至第 36 个 Decoder Layer。

也就是说，第 1 至第 3 个 Decoder Layer 同时属于语言模型推理，只是在每层
输出后额外进行一次视觉残差注入；从第 4 层开始，模型在三组 DeepStack 特征
已经全部融合的表示上继续推理。

DeepStack 的实现具有以下特点：

- 使用直接残差相加；
- 只修改序列中的视觉 Token 位置；
- 不使用独立 Cross-Attention；
- 不增加额外视觉 Token；
- 不增加语言模型上下文长度；
- 让语言模型在早期解码层中重新接收多层视觉信息。

---

## 第五阶段：完整语言解码机制与 Interleaved-MRoPE（LLM Decoding Mechanism）

本阶段说明的是贯穿全部 36 个 Decoder Layer 的共同机制，而不是在
DeepStack 完成后才启动的另一套解码器。Interleaved-MRoPE 和因果
Self-Attention 从第 1 层开始生效；第 4 至第 36 层只是不会再接收新的
DeepStack 残差。

### 1. Qwen3 Decoder

Qwen3-VL-4B 的文本解码器主要参数为：

- Decoder Layer 数量：36；
- 隐藏维度：2560；
- FFN 中间维度：9728；
- Query Attention Heads：32；
- Key/Value Heads：8；
- Head Dimension：128；
- 注意力形式：因果分组查询注意力，即 Causal GQA；
- 归一化：RMSNorm；
- MLP 激活函数：SiLU。

经过视觉替换后，文本 Token、视觉 Token 和特殊 Token 都位于同一个序列中，由同一组 Decoder Self-Attention 统一处理。

在因果掩码允许的方向上，后出现的文本 Token 可以关注前面出现的视觉 Token。开始生成后，每个新生成 Token 都可以关注完整的多模态提示序列。

Qwen3-VL 没有额外设置一个独立的文本—图像 Cross-Attention 模块，跨模态信息交换主要发生在统一序列的 Decoder Self-Attention 中。

### 2. 三维位置索引

Qwen3-VL 根据 `input_ids`、`mm_token_type_ids`、图像或视频的网格信息，以及可选的 `attention_mask`，为完整多模态序列构造三组位置索引：
$$
P_t, \quad P_h, \quad P_w
$$

分别对应：

- $t$：时间位置；
- $h$：图像高度网格位置；
- $w$：图像宽度网格位置。

对于普通文本 Token，三个位置轴使用相同的一维文本位置。

对于静态图片：

- 时间位置 $t$ 基本保持固定；
- 高度位置来自合并后的 $g_h / 2$ 网格；
- 宽度位置来自合并后的 $g_w / 2$ 网格。

这些位置是视觉 Token 的离散网格坐标，不是原始图片中的直接像素坐标。

从概念上看，视觉位置由上述 $t/h/w$ 三条轴描述。当前 Transformers 实现为了
同时服务因果掩码、KV Cache 与多模态 RoPE，会再保留一条普通文本序列位置，
把实现中的 `position_ids` 组织为：

$$
[\text{text position},\ P_t,\ P_h,\ P_w]
$$

其中第一条位置流用于常规文本顺序和缓存推进，后三条位置流用于多模态旋转位置
编码。这是实现接口上的四条流，不代表视觉空间多出了第四个坐标轴。

### 3. Interleaved-MRoPE

原始 MRoPE 将时间、高度和宽度频率分别放在连续的通道区间中。

Interleaved-MRoPE 改为把 $t / h / w$ 对应的旋转频率交错分布在注意力 Head 的不同频率通道中，使三个位置轴都能覆盖较低和较高的频率范围。

在 Decoder Self-Attention 中，Interleaved-MRoPE 作用于 Query 和 Key：
$$
\begin{aligned}
Q' &= \operatorname{RoPE}(Q, P_t, P_h, P_w) \\
K' &= \operatorname{RoPE}(K, P_t, P_h, P_w)
\end{aligned}
$$

随后计算：
$$
\begin{aligned}
\operatorname{Attention}(Q', K', V)
&= \operatorname{Softmax}\!\left(
\frac{Q'(K')^{\mathsf{T}}}{\sqrt{d}} + \text{CausalMask}
\right)V
\end{aligned}
$$

Interleaved-MRoPE 的作用是向模型提供更均衡的空间—时间位置表示。它本身不会直接输出检测框，也不能单独保证坐标预测精度。

如果模型在训练中学习了目标定位格式，它可以通过语言模型生成类似：

```text
[x1, y1, x2, y2]
```

的坐标文本，但这仍是自回归文本预测结果。

---

## 第六阶段：预测与自回归输出（Prediction Phase）

### 1. 输出隐藏状态

完成 36 个 Decoder Layer 后，序列经过最终 RMSNorm，得到：
$$
H_{\text{final}}
\in\mathbb{R}^{B\times L\times2560}
$$

### 2. LM Head

隐藏状态经过 LM Head：
$$
\text{LM Head}:
\mathbb{R}^{2560}
\rightarrow
\mathbb{R}^{151936}
$$

其中 151936 是 Qwen3-VL-4B 的词表大小。

该检查点配置为词嵌入与 LM Head 权重共享，因此这里的 $W_{\text{LM}}$ 与输入
Token Embedding 使用同一组权重。

对于下一个 Token 的预测，模型使用最后一个有效位置的隐藏状态计算，得到词表 Logits：
$$
z_{t+1} = W_{\text{LM}} h_t \in \mathbb{R}^{151936}
$$

### 3. Token 选择

根据推理配置，可以使用：

- Greedy Decoding；
- Temperature Sampling；
- Top-k Sampling；
- Top-p Sampling。

概念上可以将 Logits 转换为概率，然后选择下一个 Token：
$$
p(x_{t+1} \mid x_{\le t}, I) = \operatorname{Softmax}(z_{t+1})
$$

这里输出的是 Token，不一定等于单个字符；一个 Token 可能对应一个汉字、词片段、标点或特殊符号。

### 4. 自回归循环

新 Token 被添加到已有序列后，模型继续预测：
$$
x_{t+2}, x_{t+3}, \ldots
$$

推理过程中通常使用 KV Cache，避免每一步重新计算全部历史 Token。

第一次 Prefill 会处理完整的文本—图像提示，计算视觉特征、执行 DeepStack 注入，并建立各 Decoder 层的 KV Cache。后续自回归步骤通常只输入新生成的 Token，同时读取和更新 KV Cache；不会重新运行视觉编码器，也不会重复计算整段多模态提示。

生成持续到满足以下条件之一：

- 生成 `<|endoftext|>` 或其他结束 Token；
- 达到 `max_new_tokens`；
- 命中用户配置的停止条件。

最终输出为文本 Token 序列。即使执行目标检测或视觉定位任务，模型通常也是生成坐标或结构化内容的文本表示，而不是直接返回一个原生边界框张量。

---

# 二、系统架构流向图

本图展示了 Qwen3-VL-4B 从文本和静态图片输入，到视觉编码、DeepStack 注入和最终自回归输出的完整流动过程。

![Qwen3-VL结构](assets/Qwen3-VL-4B架构讲解/Qwen3-VL结构.canvas)

---

核验依据：

- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL-4B 官方配置（revision `ebb281ec`）](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json)
- [Qwen3-VL-4B 图像预处理配置（revision `ebb281ec`）](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/preprocessor_config.json)
- [Qwen3-VL Transformers 实现（commit `760d519a`）](https://github.com/huggingface/transformers/blob/760d519aa4d57378997b7929ed8dbf160bc11fac/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [图像 Patch 化与静态图时间维扩展实现（commit `760d519a`）](https://github.com/huggingface/transformers/blob/760d519aa4d57378997b7929ed8dbf160bc11fac/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)

核验日期：2026-07-29。
