---
title: "VLM-Loc: 项目代码精讲"
date: 2026-04-27
tags:
  - 论文/视觉语言
  - 论文/代码精讲
  - 任务/定位
  - 模态/点云
---

**原文**: [本地](../论文原件/VLM-Loc.pdf) [网络](https://arxiv.org/abs/2603.09826)

# VLM-Loc 项目代码精讲

> **VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models**
> CVPR 2026 | 南开大学 MCG 实验室
> 论文：[Arxiv](https://arxiv.org/abs/2603.09826)

---

## 目录

[[#1. 项目概述]]
[[#2. 项目结构总览]]
[[#3. 模块划分表]]
[[#4. 整体流程图]]
[[#5. 模块一：核心数据结构（imports.py）]]
[[#6. 模块二：全局常量与配置（utils.py）]]
[[#7. 模块三：方向计算与物体选择（select_ori.py）]]
[[#8. 模块四：Cell 创建与位姿描述生成（descriptions.py）]]
[[#9. 模块五：KITTI-360 数据准备（prepare_cityloc-k.py）]]
[[#10. 模块六：CityRefer 数据准备（prepare_cityloc-c.py）]]
[[#11. 模块七：坐标转换工具（data/utils.py）]]
[[#12. 模块八：数据集生成（dataset_generation_*.py）]]
[[#13. 模块九：训练与推理（train.sh / test.sh）]]
[[#14. 模块十：评估指标（recall.py）]]
[[#15. 模块十一：系统提示词（system_prompt.txt）]]
[[#16. 模块十二：可视化工具（drawing.py / rendering.py）]]
[[#17. 模块十三：空间邻域关系（add_relation.py）]]
[[#18. 模块十四：数据分割（split_data.py）]]
[[#19. 代码质量评估与优化建议]]

---

## 1. 项目概述

### 1.1 研究问题

**Text-to-Point-Cloud (T 2 P) 定位**：给定一段自然语言描述（如"目标位置位于一辆灰色汽车北方的一条道路上"），在 3D 点云地图中推断出精确的空间位置。

### 1.2 核心创新

VLM-Loc 利用**大型视觉语言模型（VLM）**的空间推理能力来解决 T 2 P 定位问题：

1. **BEV 图像 + 场景图**：将 3D 点云转换为鸟瞰图（BEV）图像和场景图，联合编码几何与语义信息
2. **部分节点分配机制**：显式地将文本线索与场景图节点关联，实现可解释的空间推理
3. **CityLoc 基准**：基于多源点云构建的细粒度 T 2 P 定位基准数据集

### 1.3 技术栈

| 组件 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| 深度学习框架 | PyTorch, ms-swift (ModelScope) |
| 基础模型 | Qwen 3-VL (2 B/4 B/8 B/32 B), InternVL 3.5-8 B |
| 训练策略 | LoRA (Low-Rank Adaptation) |
| 数据来源 | KITTI-360, CityRefer (Sensaturban) |
| 点云处理 | Open 3 D, NumPy |
| 可视化 | OpenCV, pptk |

### 1.4 两种运行状态

VLM-Loc 项目有**训练状态**和**推理（实际使用）状态**两种完全不同的工作模式，理解它们的区别至关重要：

> [!tip] 核心区别
> **训练状态**需要完整的离线数据准备流水线（点云 → Cell → BEV 图像 → 场景图 → 训练样本），而**推理状态**只需要一张 BEV 图像 + 一段文本描述即可输出定位坐标。

#### 训练状态（离线，一次性）

```
目标：让 VLM 学会"看 BEV 图 + 读场景图 → 推断位置"的能力

输入：原始 3D 点云数据集（KITTI-360 / CityRefer）
流程：点云 → 物体提取 → Cell 划分 → BEV 渲染 → 场景图构建
      → 自然语言描述生成 → LoRA 微调
输出：训练好的 LoRA 适配器权重（checkpoint）
涉及模块：M1 ~ M9（数据准备 + 数据集生成 + 训练）
运行频率：仅需运行一次
```

#### 推理状态（在线，可反复使用）

```
目标：给定文本描述，在点云地图中定位目标位置

输入：
  1. 一张 BEV 鸟瞰图（224×224 PNG）
  2. 一个场景图 JSON（物体节点列表）
  3. 一段自然语言描述（如"目标在灰色汽车北方"）
流程：BEV 图 + 场景图 + 文本 → VLM → JSON 输出
输出：{"assignments": [...], "point_2d": [x, y]}
涉及模块：M9（推理）、M10（评估）、M11（系统提示词）
运行频率：每次定位请求都需运行
```

#### 对比总览

| 维度 | 训练状态 | 推理状态 |
|------|----------|----------|
| **目的** | 训练模型参数 | 使用模型定位 |
| **输入** | 原始 3D 点云（GB 级） | BEV 图 + 场景图 + 文本 |
| **输出** | LoRA 权重文件 | 像素坐标 point_2 d |
| **计算资源** | 2× RTX 4090，数小时 | 单 GPU，秒级响应 |
| **前置步骤** | 数据准备 → 数据集生成 → 微调 | 仅需准备 BEV 图和场景图 |
| **涉及代码** | `prepare_*.py` → `dataset_generation_*.py` → `train.sh` | `test.sh` → `recall.py` |
| **是否需要原始点云** | ✅ 是 | ❌ 否 |

> [!note] 实际部署要点
> 在实际应用中，**训练只需做一次**。部署时只需：① 对新场景的点云生成 BEV 图和场景图（离线预处理），② 加载训练好的模型权重进行在线推理。

---

## 2. 项目结构总览

```
nku-3d-vision/
└── vlm-loc/
    ├── README.md                          # 项目说明文档
    ├── system_prompt.txt                  # VLM 系统提示词
    ├── train.sh                           # 训练脚本
    ├── test.sh                            # 推理脚本
    ├── recall.py                          # 评估指标计算
    ├── test.ipynb                         # 测试笔记本
    ├── split_data.py                      # CityRefer 数据分割
    ├── assets/
    │   └── figure_1.png                   # 论文示意图
    ├── data/
    │   ├── utils.py                       # 坐标转换工具
    │   ├── dataset_generation_semantics_cityloc-k.py  # KITTI-360 数据集生成
    │   └── dataset_generation_semantics_cityloc-c.py  # CityRefer 数据集生成
    └── datapreparation/
        ├── args.py                        # 命令行参数解析
        └── kitti360pose/
            ├── imports.py                 # 核心数据结构定义
            ├── utils.py                   # 全局常量与句子模板
            ├── descriptions.py            # Cell 创建与位姿描述生成
            ├── select_ori.py              # 方向计算与物体选择策略
            ├── drawing.py                 # 2D 可视化工具
            ├── rendering.py               # 3D 可视化工具
            ├── prepare_cityloc-k.py       # KITTI-360 数据准备主流程
            ├── prepare_cityloc-c.py       # CityRefer 数据准备主流程
            ├── add_relation.py            # Cell 邻域关系计算
            └── utils.py                   # 工具函数（已有独立版本）
```

---

## 3. 模块划分表

| 模块编号 | 模块名称           | 所属文件                         | 核心功能                                     |   运行阶段   |
| :--: | -------------- | ---------------------------- | ---------------------------------------- | :------: |
| M 1  | 核心数据结构         | `imports.py`                 | 定义 Object 3 d、Cell、Pose、Description 等基础类 |  🏋️ 训练  |
| M 2  | 全局常量与配置        | `utils.py`                   | 场景划分、类别映射、颜色系统、句子模板                      |  🏋️ 训练  |
| M 3  | 方向计算与物体选择      | `select_ori.py`              | 计算物体方位、多种物体选择策略                          |  🏋️ 训练  |
| M 4  | Cell 创建与描述生成   | `descriptions.py`            | 创建空间 Cell、生成位姿描述、映射到 Best Cell           |  🏋️ 训练  |
| M 5  | KITTI-360 数据准备 | `prepare_cityloc-k.py`       | 从 KITTI-360 提取物体、生成 BEV 图像               |  🏋️ 训练  |
| M 6  | CityRefer 数据准备 | `prepare_cityloc-c.py`       | 从 CityRefer 提取物体、生成 BEV 图像               |  🏋️ 训练  |
| M 7  | 坐标转换工具         | `data/utils.py`              | 世界坐标 ↔ 像素坐标双向转换                          | 🏋️🚀 两者 |
| M 8  | 数据集生成          | `dataset_generation_*.py`    | 生成 ms-swift 格式的训练/测试数据                   |  🏋️ 训练  |
| M 9  | 训练与推理          | `train.sh`, `test.sh`        | LoRA 微调（训练）/ 模型推理（使用）                    | 🏋️🚀 两者 |
| M 10 | 评估指标           | `recall.py`                  | 计算 R@5m , R@10m , R@15m                  |  🚀 推理   |
| M 11 | 系统提示词          | `system_prompt.txt`          | 定义 VLM 的推理规则和输出格式                        |  🚀 推理   |
| M 12 | 可视化工具          | `drawing.py`, `rendering.py` | 2 D/3 D 场景可视化                            |  🏋️ 训练  |
| M 13 | 邻域关系           | `add_relation.py`            | 计算 Cell 之间的空间邻域关系                        |  🏋️ 训练  |
| M 14 | 数据分割           | `split_data.py`              | CityRefer 数据集的实例级分割                      |  🏋️ 训练  |

---

## 4. 整体流程图

> [!important] 阅读指南
> 下图分为**上下两部分**：上半部分是「训练阶段」（离线，一次性），下半部分是「推理阶段」（在线，可反复使用）。两者通过 **LoRA 权重** 连接。

```
╔═══════════════════════════════════════════════════════════════════════╗
║  🏋️ 训练阶段（离线 · 一次性）                                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ┌──────────────┐    ┌──────────────┐                                 ║
║  │  KITTI-360   │    │  CityRefer   │   原始 3D 点云数据集              ║
║  │  (PLY 点云)  │     │ (Torch 张量) │                                 ║
║  └──────┬───────┘    └──────┬───────┘                                 ║
║         │                   │                                         ║
║         v                   v                                         ║
║  ┌──────────────┐    ┌──────────────┐                                 ║
║  │ prepare_     │    │ prepare_     │   Step 1: 数据准备               ║
║  │ cityloc-k.py │    │ cityloc-c.py │   (M5/M6)                       ║
║  └──────┬───────┘    └──────┬───────┘                                 ║
║         │                   │                                         ║
║         └───────┬───────────┘                                         ║
║                 │                                                     ║
║                 v                                                     ║
║  ┌──────────────────────────────┐                                     ║
║  │  中间数据 (pickle)           │                                      ║
║  │  - objects.pkl (3D 物体)     │                                     ║
║  │  - cells.pkl (空间 Cell)     │                                     ║
║  │  - poses.pkl (位姿+描述)     │                                      ║
║  │  - BEV 图像 (PNG)            │                                      ║
║  │  - centers_info (场景图数据)  │                                      ║
║  └──────────────┬───────────────┘                                     ║
║                 │                                                     ║
║                 v                                                     ║
║  ┌──────────────────────────────┐                                     ║
║  │  dataset_generation_*.py     │   Step 2: 数据集生成 (M8)            ║
║  │  - 场景图构建                 │                                     ║
║  │  - 节点匹配 (Grounding)       │                                     ║
║  │  - 自然语言描述生成           │                                      ║
║  │  - ms-swift JSON 格式输出     │                                     ║
║  └──────────────┬───────────────┘                                     ║
║                 │                                                     ║
║                 v                                                     ║
║  ┌──────────────────────────────┐                                     ║
║  │  训练数据 JSON                │                                     ║
║  │  messages: [                 │                                    ║
║  │    {role:"user",             │                                    ║
║  │     content: scene_graph},   │                                    ║
║  │    {role:"user",             │                                    ║
║  │     content: "<image> 描述"}, │                                   ║
║  │    {role:"assistant",        │                                    ║
║  │     content: JSON预测}        │                                    ║
║  │  ]                           │                                    ║
║  └──────────────┬───────────────┘                                    ║
║                 │                                                    ║
║                 v                                                    ║
║  ┌──────────────────────────────┐                                    ║
║  │  ms-swift LoRA 微调          │   Step 3: 模型训练 (M9)              ║
║  │  - 基座: Qwen3-VL-8B        │                                      ║
║  │  - LoRA rank=8, alpha=16    │                                     ║
║  │  - bf16, gradient ckpt      │                                     ║
║  └──────────────┬───────────────┘                                    ║
║                 │                                                    ║
║                 v                                                    ║
║         ┌───────────────┐                                            ║
║         │  LoRA 权重     │  ← 训练产物，连接两个阶段                     ║
║         │  (checkpoint)  │                                           ║
║         └───────┬───────┘                                            ║
╚═════════════════╪════════════════════════════════════════════════════╝
                  │
                  ▼
╔═══════════════════════════════════════════════════════════════════════╗
║  🚀 推理阶段（在线 · 可反复使用）                                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ┌─────────────────────────────────────────────┐                      ║
║  │  用户输入                                     │                     ║
║  │  ① BEV 鸟瞰图 (224×224 PNG)                 │                      ║
║  │  ② 场景图 JSON ({nodes: [{id,label,center}]})│                     ║
║  │  ③ 文本描述 ("目标在灰色汽车北方")             │                       ║
║  └──────────────┬──────────────────────────────┘                      ║
║                 │                                                     ║
║                 v                                                     ║
║  ┌──────────────────────────────┐                                     ║
║  │  VLM 推理 (swift infer)      │   加载基座模型 + LoRA 权重             ║
║  │  - 输入: BEV 图 + 场景图     │   (M9 推理部分)                        ║
║  │  - 系统提示词: system_prompt  │   (M11)                             ║
║  └──────────────┬───────────────┘                                    ║
║                 │                                                    ║
║                 v                                                    ║
║  ┌──────────────────────────────┐                                    ║
║  │  VLM 输出 (JSON)             │                                     ║
║  │  {                          │                                     ║
║  │    "assignments": [          │   部分节点分配结果                    ║
║  │      {"object_label":"car",  │   (可解释的推理过程)                  ║
║  │       "grounded":true,       │                                    ║
║  │       "matched_node":3}],    │                                    ║
║  │    "point_2d": [112, 56]     │   ← 定位结果（像素坐标）               ║
║  │  }                          │                                     ║
║  └──────────────┬───────────────┘                                    ║
║                 │                                                    ║
║                 v                                                    ║
║  ┌──────────────────────────────┐                                    ║
║  │  后处理 (可选)                │                                     ║
║  │  - 像素坐标 → 世界坐标转换    │   使用 pixels_to_world_from_center│    ║
║  │  - recall.py 评估 (M10)      │   R@5m, R@10m, R@15m                 ║
║  └──────────────────────────────┘                                     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

> [!note] 哪些模块属于哪个阶段？
> - **仅训练阶段使用**：M 1（数据结构）、M 2（常量）、M 3（方向计算）、M 4（描述生成）、M 5/M 6（数据准备）、M 7（坐标转换）、M 8（数据集生成）、M 12（可视化）、M 13（邻域关系）、M 14（数据分割）
> - **仅推理阶段使用**：M 9（推理部分）、M 10（评估）、M 11（系统提示词）
> - **两个阶段共用**：M 9（train.sh 用于训练，test.sh 用于推理）

---

## 5. 模块一：核心数据结构（imports.py）

**文件路径**：`datapreparation/kitti360pose/imports.py`

### 5.1 概述

这是整个项目的**数据结构基石**，定义了所有核心类。所有其他模块都依赖于此文件。理解这些数据结构是理解整个项目的前提。

### 5.2 全局常量

```python
COLOR_NAMES = [
    "dark-green", "gray", "gray-green", "bright-gray",
    "gray", "black", "green", "beige"
]
COLORS = [
    [0.05, 0.35, 0.05], [0.55, 0.55, 0.55], [0.35, 0.45, 0.30],
    [0.75, 0.75, 0.75], [0.45, 0.45, 0.45], [0.15, 0.15, 0.15],
    [0.15, 0.55, 0.15], [0.80, 0.75, 0.65]
]
```

这定义了 **8 种离散颜色名称**及其对应的 RGB 中心值（归一化到 [0,1]）。用于将物体颜色量化为自然语言可描述的颜色词。

### 5.3 Object 3 d 类

**用途**：表示场景中的一个 3D 物体实例。

```python
class Object3d:
    def __init__(self, id, instance_id, xyz, rgb, label):
        self.id = id                    # 物体 ID（仅在单个 cell 内唯一）
        self.instance_id = instance_id  # 原始实例 ID（可跨 cell 重复）
        self.xyz = xyz                  # 降采样后的点云坐标 (N, 3)
        self.xyz_raw = xyz.copy()       # 原始（未降采样）点云坐标
        self.rgb = rgb                  # 降采样后的 RGB 颜色 (N, 3), [0,1]
        self.rgb_raw = rgb.copy()       # 原始 RGB 颜色
        self.label = label              # 语义类别标签（字符串）
```

**关键方法详解**：

#### `get_color_rgb()` — 获取物体平均颜色
```python
def get_color_rgb(self):
    return np.mean(self.rgb, axis=0)
```
计算物体所有点的平均 RGB 值，用于后续颜色名称映射。

#### `get_color_text()` — 将颜色映射为文字
```python
def get_color_text(self):
    avg_color = self.get_color_rgb()
    # 计算与每个颜色中心的 L2 距离
    dists = [np.linalg.norm(avg_color - c) for c in COLORS]
    # 返回最近的颜色名称
    return COLOR_NAMES[np.argmin(dists)]
```
通过 L 2 距离将物体的平均颜色映射到最近的离散颜色名称。例如，一个平均颜色接近 `[0.05, 0.35, 0.05]` 的物体会被标记为 "dark-green"。

#### `get_center()` — 获取物体中心
```python
def get_center(self):
    return np.mean(self.xyz, axis=0)
```

#### `apply_downsampling(indices)` — 应用降采样
```python
def apply_downsampling(self, indices):
    self.xyz = self.xyz[indices]
    self.rgb = self.rgb[indices]
```
根据索引数组对点云进行降采样，保留指定索引的点。

#### `mask_points(mask, use_raw)` — 根据掩码过滤点
```python
def mask_points(self, mask, use_raw):
    if use_raw:
        xyz = self.xyz_raw[mask]
        rgb = self.rgb_raw[mask]
    else:
        xyz = self.xyz[mask]
        rgb = self.rgb[mask]
    return Object3d(self.id, self.instance_id, xyz, rgb, self.label)
```
根据布尔掩码过滤点云，返回新的 Object 3 d。`use_raw` 参数决定操作原始点云还是降采样后的点云。

#### `get_closest_point(anchor)` — 获取最近点
```python
def get_closest_point(self, anchor):
    dists = np.linalg.norm(self.xyz - anchor, axis=1)
    return self.xyz[np.argmin(dists)]
```
返回物体中距离锚点最近的点。用于计算位姿描述中的方向和偏移。

#### `merge(obj1, obj2)` — 合并两个物体（类方法）
```python
@classmethod
def merge(cls, obj1, obj2):
    xyz = np.vstack([obj1.xyz, obj2.xyz])
    rgb = np.vstack([obj1.rgb, obj2.rgb])
    return cls(obj1.id, obj1.instance_id, xyz, rgb, obj1.label)
```
将两个同类同 ID 的物体合并（拼接点云）。用于处理跨 PLY 文件的同一物体。

### 5.4 DescriptionPoseCell 类

**用途**：在 Pose Cell 上下文中对位姿的单条描述。

```python
class DescriptionPoseCell:
    def __init__(self, object_id, object_instance_id, object_label,
                 object_center, object_color_rgb, object_color_text,
                 direction, offset_center, offset_closest, closest_point):
        self.object_id = object_id              # 物体在 cell 内的 ID
        self.object_instance_id = object_instance_id  # 原始实例 ID
        self.object_label = object_label        # 语义标签（如 "car"）
        self.object_center = object_center      # 物体中心坐标
        self.object_color_rgb = object_color_rgb  # 物体平均 RGB
        self.object_color_text = object_color_text  # 颜色文字（如 "gray"）
        self.direction = direction              # 方向（"north"/"south"/"east"/"west"/"on-top"）
        self.offset_center = offset_center      # 物体中心到位姿的 2D 偏移
        self.offset_closest = offset_closest    # 最近点到位姿的 2D 偏移
        self.closest_point = closest_point      # 物体上距离位姿最近的点
```

**设计意图**：每条描述记录了"位姿相对于某个物体的空间关系"。例如，`direction="north"` 表示位姿在物体的北方。`offset_center` 和 `offset_closest` 提供了精确的空间偏移信息。

### 5.5 DescriptionBestCell 类

**用途**：在 Best Cell（数据库中最接近位姿的 cell）上下文中的描述。

```python
class DescriptionBestCell:
    def __init__(self, object_id, object_instance_id, object_label,
                 object_center, object_color_rgb, object_color_text,
                 direction, offset_center, offset_closest, closest_point,
                 best_offset_center, best_offset_closest, is_matched):
        # ... 与 DescriptionPoseCell 相同的字段 ...
        self.best_offset_center = best_offset_center  # 在 best cell 中的偏移
        self.best_offset_closest = best_offset_closest  # 在 best cell 中的最近点偏移
        self.is_matched = is_matched  # 是否在 best cell 中找到匹配
```

**关键类方法**：

- `from_matched(descr, object_id, best_closest_point, best_offset_center, best_offset_closest)`：从匹配成功的描述创建，保留原始方向，更新 best cell 中的空间信息
- `from_unmatched(descr)`：从未匹配的描述创建，设置 `is_matched=False`

**为什么需要两种 Description？** 因为描述最初是在 pose cell（以位姿为中心临时创建的 cell）中生成的，但最终需要映射到 best cell（数据库中预存的最近 cell）中使用。两者的坐标系不同，因此需要分别记录偏移信息。

### 5.6 Pose 类

**用途**：表示一个查询位姿及其所有描述。

```python
class Pose:
    def __init__(self, pose, pose_w, cell_id, scene_name, descriptions,
                 described_by, eval_cell_id, eval_scene_name):
        self.pose = pose                  # 在 best cell 中归一化到 [0,1] 的坐标
        self.pose_w = pose_w              # 世界坐标系中的坐标
        self.cell_id = cell_id            # best cell 的 ID
        self.scene_name = scene_name      # 场景名称
        self.descriptions = descriptions  # DescriptionBestCell 列表
        self.described_by = described_by  # 描述策略名称
        self.eval_cell_id = eval_cell_id  # 用于评估的 cell ID
        self.eval_scene_name = eval_scene_name  # 用于评估的场景名
```

**关键方法**：

- `get_text()`：将所有描述拼接为自然语言文本
- `get_number_unmatched()`：返回未匹配描述的数量（用于质量过滤）

### 5.7 Cell 类

**用途**：表示场景中的一个空间区域。

```python
class Cell:
    def __init__(self, scene_name, id, objects, cell_size, bbox_w):
        self.scene_name = scene_name  # 场景名称
        self.id = id                  # 唯一 ID，格式: {scene_name}_{idx:05d}
        self.objects = objects        # cell 内的所有 Object3d
        self.cell_size = cell_size    # 最长边长度（世界坐标）
        self.bbox_w = bbox_w          # 世界坐标系边界框 [xmin,ymin,zmin,xmax,ymax,zmax]
```

**设计意图**：将大场景划分为 50 m×50 m 的小区域（cell），每个 cell 包含该区域内的所有物体。BEV 图像也是以 cell 为单位生成的。

---

## 6. 模块二：全局常量与配置（utils.py）

**文件路径**：`datapreparation/kitti360pose/utils.py`

### 6.1 场景划分

```python
SCENE_NAMES = [
    "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync", "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync", "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync", "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]

SCENE_NAMES_TRAIN = [
    "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync", "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
]

SCENE_NAMES_VAL = ["2013_05_28_drive_0006_sync"]

SCENE_NAMES_TEST = [
    "2013_05_28_drive_0007_sync", "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]
```

KITTI-360 的 9 个驾驶序列被划分为 5 个训练、1 个验证、3 个测试场景。

### 6.2 类别映射

```python
KNOWN_CLASS = [
    "road", "sidewalk", "parking", "wall", "fence", "guard rail",
    "bridge", "tunnel", "vegetation", "terrain", "pole", "traffic sign",
    "traffic light", "polegroup", "billboard", "building", "garage",
    "car", "truck", "bus", "trailer", "motorcycle"
]
```

共 22 种已知类别。其中 **stuff 类别**（大面积区域）包括：

```python
STUFF_CLASSES = [
    "road", "sidewalk", "parking", "wall", "fence", "guard rail",
    "bridge", "tunnel", "vegetation", "terrain"
]
```

**stuff vs instance 的区别**：stuff 类别（如道路、植被）是连续的大面积区域，需要 DBSCAN 聚类将其分割为独立实例；instance 类别（如汽车、杆子）本身就是离散的独立物体。

### 6.3 最小点数阈值

```python
CLASS_TO_MINPOINTS = {
    "road": 2500, "sidewalk": 2500, "parking": 2500,
    "wall": 2500, "fence": 2500, "guard rail": 2500,
    "bridge": 2500, "tunnel": 2500, "vegetation": 2500,
    "terrain": 2500, "pole": 25, "traffic sign": 25,
    "traffic light": 25, "polegroup": 25, "billboard": 25,
    "building": 2500, "garage": 2500, "car": 250,
    "truck": 250, "bus": 250, "trailer": 250, "motorcycle": 250,
}
```

stuff 类别的最小点数（2500）远高于 instance 类别（25-250），反映了它们面积大小的差异。点数不足阈值的物体将被过滤掉。

### 6.4 句子模板

这是项目中非常巧妙的设计——通过**大量多样化的句子模板**实现数据增强，使模型能学习到丰富的语言表达。

**"on-top" 方向模板**（约 20 个变体）：
```python
sentence_style_t = [
    "on top of a {object}",
    "directly on top of a {object}",
    "situated on a {object}",
    "located on a {object}",
    "positioned on a {object}",
    "placed on a {object}",
    "{Object} is directly below the pose",
    "{Object} is right beneath the pose",
    ...
]
```

**"north" 方向模板**（约 40 个变体）：
```python
sentence_style_n = [
    "north of a {object}",
    "to the north of a {object}",
    "slightly north of a {object}",
    "just north of a {object}",
    "a short distance north of a {object}",
    "{Object} is to the south of the pose",
    "{Object} lies south of the pose",
    ...
]
```

每个方向都有类似的模板集。`{object}` 和 `{Object}` 分别用于主动语态和被动语态的句型。

**数据增强效果**：同一个空间关系（如"在汽车北方"）可以被表达为 40 种不同的句子，极大地增加了训练数据的语言多样性。

---

## 7. 模块三：方向计算与物体选择（select_ori.py）

**文件路径**：`datapreparation/kitti360pose/select_ori.py`

### 7.1 方向计算

#### `get_direction(obj, pose, on_top_thr=0.05)`

**核心算法**：判断位姿相对于物体的方向。

```python
def get_direction(obj, pose, on_top_thr=0.05):
    # 1. 获取物体上距离位姿最近的点
    closest = obj.get_closest_point(pose)

    # 2. 计算最近点到位姿的 2D 向量
    offset = pose[:2] - closest[:2]
    distance = np.linalg.norm(offset)

    # 3. 如果距离很近，判定为 "on-top"
    if distance < on_top_thr:
        return "on-top"

    # 4. 否则，计算物体中心到位姿的向量
    center = obj.get_center()
    dx = pose[0] - center[0]
    dy = pose[1] - center[1]

    # 5. 根据 |dx| 和 |dy| 的大小关系确定主方向
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "north" if dy > 0 else "south"
```

**关键细节**：
- 首先检查"on-top"（位姿几乎在物体上），使用最近点距离
- 方向判断使用物体中心到位姿的向量（而非最近点），更稳定
- `on_top_thr=0.05` 是归一化坐标下的阈值（cell 归一化后最大为 1）

#### `get_direction_noOntop(obj, pose)`

与 `get_direction` 类似，但**不判断 on-top**，始终基于物体中心计算方向。用于需要强制指定东南西北方向的场景。

### 7.2 物体选择策略

系统提供了 4 种从候选物体中选择用于描述的物体的策略：

#### 策略一：最近距离优先（closest）
```python
def select_objects_closest(objects, pose, num_mentioned):
    # 按最近点到位姿的距离排序，选择前 num_mentioned 个
    dists = [np.linalg.norm(obj.get_closest_point(pose)[:2] - pose[:2])
             for obj in objects]
    sorted_indices = np.argsort(dists)
    return [objects[i] for i in sorted_indices[:num_mentioned]]
```

#### 策略二：方向多样性优先（direction）
```python
def select_objects_direction(objects, pose, num_mentioned):
    # 按方向分组，轮流从每个方向组中选取
    # 确保描述中包含不同方向的物体
    direction_groups = {"north": [], "south": [], "east": [], "west": [], "on-top": []}
    for obj in objects:
        d = get_direction(obj, pose)
        direction_groups[d].append(obj)
    # 轮流从各组中选取...
```

**设计意图**：如果只选最近的物体，可能所有描述都是"北方"。方向多样性策略确保描述覆盖多个方向，提供更丰富的空间信息。

#### 策略三：类别多样性优先（class）
```python
def select_objects_class(objects, pose, num_mentioned):
    # 按类别分组，轮流从每个类别组中选取
    # 确保描述中包含不同类别的物体
```

#### 策略四：随机选择（random）
```python
def select_objects_random(objects, pose, num_mentioned):
    # 随机无放回选择 num_mentioned 个物体
    indices = np.random.choice(len(objects), num_mentioned, replace=False)
    return [objects[i] for i in indices]
```

**实际使用**：在 `args.py` 中通过 `--describe_by` 参数选择策略，默认为 `"all"`（使用所有策略的组合）。

---

## 8. 模块四：Cell 创建与位姿描述生成（descriptions.py）

**文件路径**：`datapreparation/kitti360pose/descriptions.py`

### 8.1 概述

这是数据准备流水线的**核心算法模块**，实现了三个关键功能：
1. 创建空间 Cell（含 DBSCAN 聚类）
2. 在 Pose Cell 中生成位姿描述
3. 将描述从 Pose Cell 映射到 Best Cell

### 8.2 create_cell — 创建空间 Cell

```python
def create_cell(cell_idx, scene_name, bbox_w, scene_objects,
                num_mentioned=6, inside_fraction=1/3,
                stuff_min=1000, all_cells=False, stuff_classes=None):
```

**详细流程**：

```
输入: bbox_w (世界坐标边界框), scene_objects (场景中所有物体)
  │
  ├─ 1. 遍历所有场景物体
  │   ├─ 对 stuff 类别:
  │   │   ├─ 用 bbox 裁剪点云
  │   │   ├─ DBSCAN 聚类 (eps=0.75m)
  │   │   └─ 过滤点数 < stuff_min 的簇
  │   └─ 对 instance 类别:
  │       ├─ 计算在 bbox 内的点比例
  │       └─ 仅保留比例 >= inside_fraction 的物体
  │
  ├─ 2. 坐标归一化
  │   └─ 将所有物体坐标归一化到 [0, 1]（基于 cell 最长边）
  │
  ├─ 3. 过滤
  │   └─ 如果物体数 < num_mentioned，返回 None
  │
  └─ 4. 重设 ID
      └─ 为所有物体分配递增的 cell 内唯一 ID
```

**为什么需要对 stuff 类别做 DBSCAN 聚类？**

stuff 类别（如道路、植被）在场景中是连续的大面积区域。当用 bbox 裁剪时，可能裁出多个不连续的片段。DBSCAN 聚类将这些片段分割为独立的实例，每个实例可以单独描述。

```python
def cluster_stuff_object(obj, stuff_min, eps=0.75, map_to_raw=True):
    # 1. 对降采样后的点云进行 DBSCAN 聚类
    db = DBSCAN(eps=eps, min_samples=1).fit(obj.xyz)
    labels = db.labels_

    # 2. 使用 KDTree 将聚类标签映射回原始点云
    tree = KDTree(obj.xyz)
    _, indices = tree.query(obj.xyz_raw)
    raw_labels = labels[indices]

    # 3. 为每个簇创建新的 Object3d
    for label_id in unique_labels:
        mask = (raw_labels == label_id)
        if np.sum(mask) >= stuff_min:
            new_obj = obj.mask_points(mask, use_raw=True)
            clustered.append(new_obj)
```

### 8.3 describe_pose_in_pose_cell — 生成位姿描述

```python
def describe_pose_in_pose_cell(pose_w, cell, select_by, num_mentioned,
                                max_dist=0.5, no_ontop=False):
```

**详细流程**：

```
输入: pose_w (世界坐标位姿), cell (空间 Cell)
  │
  ├─ 1. 将位姿归一化到 cell 坐标系
  │
  ├─ 2. 筛选候选物体
  │   └─ 仅保留距离位姿 <= max_dist * cell_size 的物体
  │
  ├─ 3. 选择物体
  │   └─ 根据 select_by 策略选择 num_mentioned 个物体
  │
  └─ 4. 为每个选中物体生成描述
      ├─ 计算方向 (get_direction)
      ├─ 找到最近点 (get_closest_point)
      ├─ 计算偏移向量
      ├─ 获取颜色信息
      └─ 创建 DescriptionPoseCell 对象
```

** `max_dist=0.5` 的含义**：只选择距离位姿不超过 cell 尺度 50% 的物体。对于 50 m 的 cell，即只选择 25 m 范围内的物体，确保描述的空间相关性。

### 8.4 ground_pose_to_best_cell — 映射描述到 Best Cell

```python
def ground_pose_to_best_cell(pose_w, pose_cell_descriptions, cell, all_cells=False):
```

**这是整个系统中最关键的函数之一。**

**问题背景**：描述是在 pose cell（以位姿为中心临时创建的 cell）中生成的，但训练/推理时使用的是 best cell（数据库中预存的最近 cell）。两者的物体集合和坐标系都不同。

**匹配算法**：

```
对于 pose cell 中的每条描述:
  │
  ├─ 1. 在 best cell 中查找相同 instance_id 的物体
  │
  ├─ 2. 如果找到:
  │   ├─ 计算在 best cell 中的最近点偏移
  │   ├─ 验证偏移差异 <= sqrt(2)/2
  │   │   (确保空间关系的一致性)
  │   └─ 创建 DescriptionBestCell.from_matched
  │
  └─ 3. 如果未找到:
      └─ 创建 DescriptionBestCell.from_unmatched
```

**偏移验证的直觉**：如果位姿相对于物体的空间关系在 pose cell 和 best cell 中差异太大，说明匹配不可靠。`sqrt(2)/2` 是归一化坐标下的阈值。

---

## 9. 模块五：KITTI-360 数据准备（prepare_cityloc-k.py）

**文件路径**：`datapreparation/kitti360pose/prepare_cityloc-k.py`

### 9.1 概述

这是 **CityLoc-K 数据集**的主数据准备脚本，从原始 KITTI-360 3D 语义点云出发，完成完整的处理流水线。

### 9.2 关键函数详解

#### load_points — 加载 PLY 点云

```python
def load_points(filepath):
    # 使用 Open3D 加载 PLY 文件
    pcd = o3d.io.read_point_cloud(filepath)
    xyz = np.asarray(pcd.points)          # (N, 3) 坐标
    rgb = np.asarray(pcd.colors)          # (N, 3) 颜色 [0, 1]
    lbl = ...  # 从文件名解析语义标签
    iid = ...  # 从文件名解析实例标签
    return xyz, rgb, lbl, iid
```

KITTI-360 的语义点云按标签和实例组织为多个 PLY 文件，文件名包含标签和实例 ID 信息。

#### extract_objects — 提取 3D 物体

```python
def extract_objects(xyz, rgb, lbl, iid):
    objects = []
    for (label, instance_id), mask in groupby(lbl, iid):
        obj = Object3d(
            id=len(objects),
            instance_id=instance_id,
            xyz=xyz[mask],
            rgb=rgb[mask],
            label=label
        )
        objects.append(obj)
    return objects
```

按语义标签和实例标签分组，为每个实例创建 Object 3 d。

#### gather_objects_both — 收集并处理物体

```python
def gather_objects_both(path_input, folder_name):
    # 1. 遍历场景中所有 PLY 文件
    for ply_file in glob.glob(os.path.join(path, "*.ply")):
        xyz, rgb, lbl, iid = load_points(ply_file)
        file_objects = extract_objects(xyz, rgb, lbl, iid)

        # 2. 按实例 ID 合并跨文件的同一物体
        for obj in file_objects:
            if obj.instance_id in merged:
                merged[obj.instance_id] = Object3d.merge(merged[obj.instance_id], obj)
            else:
                merged[obj.instance_id] = obj

    # 3. 按类别进行体素降采样
    for obj in merged.values():
        voxel_size = CLASS_TO_VOXELSIZE.get(obj.label)
        if voxel_size:
            indices = downsample_points(obj.xyz, voxel_size)
            obj.apply_downsampling(indices)

    # 4. 按最小点数过滤
    objects = [obj for obj in merged.values()
               if len(obj.xyz) >= CLASS_TO_MINPOINTS.get(obj.label, 0)]

    return objects
```

**关键步骤**：
1. **合并**：同一物体可能跨多个 PLY 文件，需要合并
2. **降采样**：使用 Open 3 D 体素降采样减少点数，加速后续处理
3. **过滤**：移除点数不足的小物体

#### generate_sem_bev_image — 生成 BEV 图像

```python
def generate_sem_bev_image(cell, image_size, bev_range, save_path=None):
    """
    将 cell 中的 3D 点云渲染为 2D 鸟瞰图。

    参数:
        cell: Cell 对象
        image_size: 输出图像尺寸 (224x224)
        bev_range: BEV 覆盖范围 (50m x 50m)
        save_path: 可选的保存路径

    返回:
        bev_image: (224, 224, 3) uint8 图像
        centers: 物体中心信息列表
    """
    # 1. 初始化空白图像
    bev_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # 2. 计算投影参数
    x_min, y_min = cell.bbox_w[0], cell.bbox_w[1]
    x_scale = image_size / bev_range
    y_scale = image_size / bev_range

    # 3. 先绘制 stuff 类别（背景层）
    for obj in cell.objects:
        if obj.label in STUFF_CLASSES:
            xy = obj.xyz_raw[:, :2]
            x_img, y_img, idx = project_points_to_pixels(
                xy, x_min, y_min, x_scale, y_scale, image_size - 1)
            color = to_uint8_rgb(obj.get_color_rgb())
            bev_image[y_img, x_img] = color

    # 4. 再绘制 non-stuff 类别（前景层）
    for obj in cell.objects:
        if obj.label not in STUFF_CLASSES:
            xy = obj.xyz_raw[:, :2]
            x_img, y_img, idx = project_points_to_pixels(
                xy, x_min, y_min, x_scale, y_scale, image_size - 1)
            color = to_uint8_rgb(obj.get_color_rgb())
            bev_image[y_img, x_img] = color

    # 5. 计算每个物体的像素质心
    centers = []
    for obj in cell.objects:
        xy = obj.xyz_raw[:, :2]
        x_img, y_img, idx = project_points_to_pixels(
            xy, x_min, y_min, x_scale, y_scale, image_size - 1)
        pcx, pcy = int(np.mean(x_img)), int(np.mean(y_img))
        # 反投影回世界坐标
        wx, wy = pixels_to_world_from_center(pcx, pcy, center, bev_range, image_size)
        centers.append({
            "label": obj.label,
            "pixel_center": [pcx, pcy],
            "world_center": [wx, wy]
        })

    return bev_image, centers
```

**绘制顺序的重要性**：先绘制 stuff（背景），再绘制 non-stuff（前景），确保小物体（如杆子、汽车）不会被大面积区域覆盖。

### 9.3 主流程

```python
if __name__ == "__main__":
    args = parse_arguments()

    for scene_name in SCENE_NAMES:
        # 1. 从轨迹文件采样 cell 位置
        locations = create_locations(args.path_in, scene_name, args.cell_dist)

        # 2. 加载并提取 3D 物体
        objects = gather_objects_both(args.path_in, scene_name)

        # 3. 过滤远离物体的位置
        locations = get_close_locations(locations, objects, args.cell_size)

        # 4. 创建 cell 并生成 BEV 图像
        success, cells = create_cells(objects, locations, scene_name, args.cell_size, args)

        # 5. 创建 pose（含描述）
        success, poses = create_poses(objects, locations, cells, args)

        # 6. 保存结果
        save_cells_without_raw(cells, path_cells, scene_name)
        # 保存 objects.pkl, cells.pkl, poses.pkl
```

---

## 10. 模块六：CityRefer 数据准备（prepare_cityloc-c.py）

**文件路径**：`datapreparation/kitti360pose/prepare_cityloc-c.py`

### 10.1 与 KITTI-360 版本的主要区别

| 方面 | KITTI-360 (cityloc-k) | CityRefer (cityloc-c) |
|------|----------------------|----------------------|
| 数据格式 | PLY 文件 | PyTorch tensor (.pth) |
| 颜色范围 | [0, 1] | [-1, 1] |
| 语义标签 | KITTI-360 编号 | CityRefer 编号 |
| 位置采样 | 轨迹采样 | 规则网格采样 |
| stuff 类别 | sidewalk, road, parking... | Ground, High Vegetation, Wall... |

### 10.2 convert_cityrefer_objects — 格式转换

```python
def convert_cityrefer_objects(objects, semantic_names):
    """将 CityRefer 的 dict 格式转换为 Object3d"""
    result = []
    for obj_dict in objects:
        # 颜色从 [-1, 1] 映射到 [0, 1]
        rgb = (obj_dict['colors'] + 1.0) / 2.0
        obj = Object3d(
            id=obj_dict['inst_labels'],
            instance_id=obj_dict['inst_labels'],
            xyz=obj_dict['coords'],
            rgb=rgb,
            label=semantic_names[obj_dict['sem_labels']]
        )
        result.append(obj)
    return result
```

### 10.3 sample_grid_points — 网格采样

```python
def sample_grid_points(objects, interval=5.0, target_semantics={0,4,5,7,10},
                       min_distance=None, margin=25.0):
    """
    在场景中生成规则网格采样点，仅保留落在目标语义类别上的点。

    参数:
        interval: 网格间距 (5m)
        target_semantics: 目标语义 ID（道路、人行道等可行走区域）
        min_distance: 最小点间距（贪心选择）
        margin: 边界余量 (25m)
    """
    # 1. 合并所有物体的坐标和语义标签
    all_xyz, all_sem = concat_objects(objects)

    # 2. 在 bbox 内生成规则网格
    xs = np.arange(bbox_min[0] + margin, bbox_max[0] - margin, interval)
    ys = np.arange(bbox_min[1] + margin, bbox_max[1] - margin, interval)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # 3. 用 KDTree 查询每个网格点的最近邻语义
    tree = KDTree(all_xyz[:, :2])
    _, indices = tree.query(grid_points)
    grid_sem = all_sem[indices]

    # 4. 仅保留目标语义类别的点
    mask = np.isin(grid_sem, list(target_semantics))
    result_points = grid_points[mask]

    # 5. 贪心选择确保最小间距
    if min_distance:
        result_points = _enforce_min_distance(result_points, min_distance)

    return result_points
```

**设计意图**：CityRefer 数据集没有车辆轨迹，因此使用规则网格在可行走区域（道路、人行道等）上均匀采样位姿位置。

---

## 11. 模块七：坐标转换工具（data/utils.py）

**文件路径**：`data/utils.py`

### 11.1 概述

提供世界坐标和像素坐标之间的**双向转换**，是 BEV 图像生成和定位评估的基础。

### 11.2 project_points_to_pixels — 世界坐标 → 像素坐标

```python
def project_points_to_pixels(xy, x_min, y_min, x_scale, y_scale, img_size_minus_1):
    """
    将世界坐标投影到 BEV 图像像素坐标。

    参数:
        xy: (N, 2) 世界坐标 (x_east, y_north)
        x_min, y_min: BEV 区域左下角的世界坐标
        x_scale, y_scale: 缩放因子 (pixels/meter)
        img_size_minus_1: 图像尺寸 - 1

    返回:
        x_img: (N,) int32 像素 x 坐标（列号）
        y_img_flipped: (N,) int32 像素 y 坐标（行号，已翻转）
        idx: (N,) int64 展平索引
    """
    W = int(img_size_minus_1) + 1
    H = W  # 正方形 BEV

    # 计算 BEV 覆盖范围
    bev_range_x = W / float(x_scale)
    bev_range_y = H / float(y_scale)

    # 格子左边界分箱
    eps = 1e-9  # 避免浮点误差
    i = np.floor(((xy[:, 0] - x_min) * W / bev_range_x) - eps).astype(np.int64)
    j0 = np.floor(((xy[:, 1] - y_min) * H / bev_range_y) - eps).astype(np.int64)

    # 裁剪到合法索引
    np.clip(i, 0, W - 1, out=i)
    np.clip(j0, 0, H - 1, out=j0)

    # 翻转 y 轴：图像行号自上而下 -> BEV y 向上（北）
    j = (H - 1) - j0

    return i.astype(np.int32), j.astype(np.int32), (j * W + i).astype(np.int64)
```

**关键设计决策**：
- **y 轴翻转**：图像坐标系 y 向下，但 BEV 坐标系 y 向北（上），需要翻转
- **左边界分箱**：使用 `floor` 函数确保每个世界坐标点映射到唯一的像素格子
- **eps 减法**：避免恰好在格子右边界的点被映射到下一个格子

### 11.3 pixels_to_world_from_center — 像素坐标 → 世界坐标

```python
def pixels_to_world_from_center(x_pix, y_pix, center_world, bev_range,
                                 image_size, use_center=True):
    """
    将像素坐标反投影到世界坐标（固定到像素中心）。

    保证: pixel -> world -> pixel 恒等（往返一致性）
    """
    W = int(image_size)
    cx_w, cy_w = float(center_world[0]), float(center_world[1])
    half = float(bev_range) / 2.0

    x_min = cx_w - half
    y_min = cy_w - half

    off = 0.5 if use_center else 0.0

    # 翻转回未翻转的行号
    j0 = (W - 1) - y_pix

    # 像素中心的世界坐标
    xw = x_min + ((x_pix + off) * bev_range) / W
    yw = y_min + ((j0 + off) * bev_range) / W
    return xw, yw
```

**往返一致性保证**：`project_points_to_pixels(pixels_to_world_from_center(p)) == p`。这是通过使用像素中心（`+0.5`）进行反投影实现的。

---

## 12. 模块八：数据集生成（dataset_generation_*.py）

**文件路径**：`data/dataset_generation_semantics_cityloc-k.py` 和 `data/dataset_generation_semantics_cityloc-c.py`

### 12.1 概述

这两个脚本将数据准备阶段生成的中间数据（pickle 文件）转换为 **ms-swift 训练格式**的 JSON 文件。它们是连接数据准备和模型训练的桥梁。

### 12.2 核心流程

```
中间数据 (pickle)
  │
  ├─ poses.pkl: 位姿 + 描述
  ├─ cells.pkl: 空间 Cell
  ├─ objects.pkl: 3D 物体
  └─ centers_info.pkl: 物体中心信息 + BEV 图像路径
  │
  v
create_text_image_pairs()
  │
  ├─ 1. 为每个 pose 构建场景图
  │   └─ create_scene_graph(centers_info)
  │
  ├─ 2. 匹配描述中的物体到场景图节点
  │   └─ ground_pose_to_image_scene_graph_v2()
  │
  ├─ 3. 生成自然语言查询
  │   └─ "The target location is north of a gray car and east of a dark-green vegetation."
  │
  ├─ 4. 投影位姿到像素坐标
  │   └─ project_points_to_pixels()
  │
  └─ 5. 构建 assistant 回复
      └─ {"assignments": [...], "point_2d": [x, y]}
  │
  v
create_train_info()
  │
  └─ 输出 ms-swift 格式 JSON:
      {
        "messages": [
          {"role": "user", "content": "<场景图 JSON>"},
          {"role": "user", "content": "<image> <自然语言描述>"},
          {"role": "assistant", "content": "<预测 JSON>"}
        ],
        "images": ["<BEV 图像路径>"]
      }
```

### 12.3 create_scene_graph — 构建场景图

```python
def create_scene_graph(centers_info):
    """
    从物体中心信息构建场景图。

    输入: centers_info = [
        {"label": "road", "pixel_center": [112, 56], "world_center": [x, y]},
        {"label": "vegetation", "pixel_center": [45, 180], "world_center": [x, y]},
        ...
    ]

    输出: {
        "nodes": [
            {"node_id": 0, "label": "road", "pixel_center": [112, 56]},
            {"node_id": 1, "label": "vegetation", "pixel_center": [45, 180]},
            ...
        ]
    }
    """
    raw_nodes = []
    for node_id, it in enumerate(centers_info):
        raw_nodes.append({
            "node_id": node_id,
            "label": str(it["label"]).lower(),
            "pixel_center": [int(it["pixel_center"][0]), int(it["pixel_center"][1])],
            "world_center": [float(it["world_center"][0]), float(it["world_center"][1])],
        })
    return {"nodes": raw_nodes}
```

**场景图的作用**：为 VLM 提供结构化的空间语义信息。每个节点代表 BEV 图像中的一个物体实例，包含其语义标签和像素位置。

### 12.4 ground_pose_to_image_scene_graph_v 2 — 节点匹配

```python
def ground_pose_to_image_scene_graph_v2(pose, scene_graph, objects_in_cell,
                                         object_instance_ids):
    """
    将描述中的物体匹配到场景图节点。

    匹配规则:
    1. 仅在同一语义标签的节点中匹配
    2. 计算物体中心到候选节点的距离
    3. 选择最近的节点，如果距离在阈值内则匹配成功

    阈值设置:
    - stuff 类别: 15m
    - instance 类别: 5m
    - road: 50m（道路面积大，允许更大偏差）
    """
    for desc in pose.descriptions:
        # 获取物体在 cell 中的世界坐标中心
        pose_obj_center_incell = desc.object_center[0:2]
        pose_obj_center_world = pose_obj_center_incell * BEV_RANGE + \
                                 np.array(pose.pose_w[0:2]) - BEV_RANGE / 2

        # 在同标签节点中找最近的
        cand_nodes = label2nodes.get(label, [])
        dists = [np.hypot(*(nd["world_center"] - pose_obj_center_world))
                 for nd in cand_nodes]
        best_i = np.argmin(dists)
        best_dist = dists[best_i]

        # 阈值判断
        thr = 15.0 if label in STUFF_CLASSES else 5.0
        if label == 'road':
            thr = 50.0

        if best_dist <= thr:
            matched["grounded"] = True
            matched["matched_node"] = best_nd["node_id"]
```

**这是"部分节点分配机制"的核心实现**：不是所有物体都能匹配到场景图节点（可能因为距离太远或标签不匹配），因此称为"部分"分配。未匹配的物体在训练中标记为 `"grounded": false`。

### 12.5 自然语言描述生成

```python
# 为每个描述生成短语
phrases = []
for hint in pose.descriptions:
    phrases.append(f"{hint.direction} of a {hint.object_color_text} {hint.object_label}")
    # 例如: "north of a gray car"

# 用逗号和 "and" 连接
natural_clause = _join_phrases_with_and(phrases)
# 例如: "north of a gray car, east of a dark-green vegetation, and south of a road"

# 构建完整查询
query_description = f"The target location is {natural_clause}."
```

### 12.6 两个版本的差异

| 方面 | cityloc-k | cityloc-c |
|------|-----------|-----------|
| 位姿字段 | `pose.cell_id` | `pose.eval_cell_id` |
| 并行处理 | ThreadPoolExecutor (5 workers) | 串行处理 |
| BEV_RANGE | 变量引用 | 硬编码 50 |
| 路径处理 | 直接使用 | 需要替换前缀 |

---

## 13. 模块九：训练与推理（train.sh / test.sh）

### 13.1 训练脚本（train.sh）

```bash
#!/bin/bash

# 环境配置
export NCCL_P2P_DISABLE=1    # 禁用 P2P（避免多 GPU 通信问题）
export NCCL_IB_DISABLE=1     # 禁用 InfiniBand
export CUDA_VISIBLE_DEVICES=0,1  # 使用 2 张 GPU

# ms-swift SFT 训练
swift sft \
    --system ./system_prompt.txt \           # 系统提示词
    --model /path/to/Qwen3-VL-8B-Instruct \ # 基座模型
    --logging_steps 10 \
    --attn_impl "flash_attn" \               # Flash Attention 加速
    --dataset dataset_items/CityLoc-K/vlmloc_training_data.json \  # 训练数据
    --load_from_cache_file true \             # 缓存加速
    --val_dataset dataset_items/CityLoc-K/vlmloc_val_data.json \   # 验证数据
    --train_type lora \                       # LoRA 微调
    --torch_dtype bfloat16 \                  # BF16 精度
    --num_train_epochs 5 \                    # 训练 5 个 epoch
    --per_device_train_batch_size 1 \         # 每卡 batch size = 1
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \                    # 学习率
    --lora_rank 8 \                           # LoRA 秩
    --lora_alpha 16 \                         # LoRA alpha
    --target_modules all-linear \             # 对所有线性层应用 LoRA
    --freeze_vit false \                      # 不冻结 ViT
    --freeze_aligner false \                  # 不冻结对齐层
    --gradient_accumulation_steps 8 \         # 梯度累积（等效 batch=16）
    --eval_steps 500 \                        # 每 500 步评估
    --save_steps 500 \                        # 每 500 步保存
    --output_dir output \
    --warmup_ratio 0.05 \                     # 5% warmup
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --save_total_limit 5 \                    # 最多保存 5 个 checkpoint
    --gradient_checkpointing true             # 梯度检查点（节省显存）
```

**关键配置解读**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_type` | lora | 使用 LoRA 进行参数高效微调 |
| `lora_rank` | 8 | LoRA 低秩矩阵的秩 |
| `lora_alpha` | 16 | LoRA 缩放因子（alpha/rank = 2） |
| `target_modules` | all-linear | 对所有线性层应用 LoRA |
| `freeze_vit` | false | 不冻结视觉编码器（关键！需要学习 BEV 图像特征） |
| `gradient_accumulation_steps` | 8 | 2 GPU × 1 batch × 8 累积 = 等效 batch 16 |
| `torch_dtype` | bfloat 16 | 使用 BF 16 混合精度训练 |

### 13.2 推理脚本（test.sh）

```bash
#!/bin/bash

NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /path/to/Qwen3-VL-8B-Instruct \        # 基座模型
    --adapters checkpoints/output/v0-.../checkpoint-3600 \  # LoRA 权重
    --infer_backend pt \                              # PyTorch 推理后端
    --val_dataset dataset_items/CityLoc-K/vlmloc_testing_data.json \  # 测试数据
    --max_new_tokens 2048 \                           # 最大生成长度
    --system ./system_prompt.txt \                    # 系统提示词
```

推理时加载基座模型 + LoRA 适配器权重，对测试数据集进行预测，输出 JSONL 格式的结果文件。

---

## 14. 模块十：评估指标（recall.py）

**文件路径**：`recall.py`

### 14.1 概述

计算定位的 ** Recall@K ** 指标，衡量预测位置与真实位置的偏差。

### 14.2 核心逻辑

```python
PX_PER_50M = 224.0
METERS_PER_PX = 50.0 / 224.0  # ≈ 0.2232 米/像素
```

**分辨率计算**：BEV 图像为 224×224 像素，覆盖 50 m×50 m 区域，因此每个像素约代表 0.223 米。

```python
def main():
    for line in jsonl_file:
        # 1. 解析预测和真实标签
        pred_obj = json.loads(row["response"])  # VLM 的预测输出
        gt_obj = json.loads(row["labels"])       # 真实标签

        # 2. 提取 2D 坐标
        pred_pt = pred_obj["point_2d"]  # [x, y] 像素坐标
        gt_pt = gt_obj["point_2d"]

        # 3. 计算像素距离并转换为米
        dpx = math.hypot(pred_pt[0] - gt_pt[0], pred_pt[1] - gt_pt[1])
        dm = dpx * METERS_PER_PX

        # 4. 统计不同阈值下的命中数
        if dm <= 5:  hits_5 += 1
        if dm <= 10: hits_10 += 1
        if dm <= 15: hits_15 += 1

    # 5. 输出 Recall 指标
    print(f"R@5m  = {hits_5/denom:.4f}")
    print(f"R@10m = {hits_10/denom:.4f}")
    print(f"R@15m = {hits_15/denom:.4f}")
```

### 14.3 指标含义

| 指标 | 含义 |
|------|------|
| R@5m | 预测位置与真实位置距离 ≤ 5 m 的比例 |
| R@10m | 预测位置与真实位置距离 ≤ 10 m 的比例 |
| R@15m | 预测位置与真实位置距离 ≤ 15 m 的比例 |

---

## 15. 模块十一：系统提示词（system_prompt.txt）

**文件路径**：`system_prompt.txt`

### 15.1 概述

系统提示词定义了 VLM 的**行为规范、推理规则和输出格式**。它是 VLM-Loc 框架的核心组件之一，指导模型如何理解 BEV 图像和场景图，以及如何进行空间推理。

### 15.2 逐段解析

#### 角色定义
```
You are a helpful assistant for spatial reasoning and grounding in BEV images.
You are also given a scene graph describing the environment as:
{node_id, label, pixel_center}
```
定义模型角色：BEV 图像中的空间推理和定位助手。同时告知模型会接收场景图输入。

#### 坐标系规则
```
In BEV pixel coordinates:
- Up    = North = y decreases
- Down  = South = y increases
- Left  = West  = x decreases
- Right = East  = x increases
```
明确 BEV 像素坐标系的方位约定。注意 y 轴方向与常规图像坐标相反（因为做了翻转）。

#### 推理目标
```
1) Parse the user's natural-language description
2) Extract all mentioned object phrases
3) For each object phrase:
   - Find the matching node in the scene graph
   - Determine whether grounding is successful
   - Record the matched node's ID
4) Finally, infer the 2D pixel coordinate of the target location
```
定义四步推理流程：解析文本 → 提取物体 → 匹配节点 → 推断位置。

#### 输出格式
```json
{
  "assignments": [
    {"object_label": "parking", "grounded": true, "matched_node": 0},
    ...
  ],
  "point_2d": [45, 135]
}
```
严格 JSON 格式输出，包含节点分配结果和最终预测坐标。

### 15.3 设计哲学

系统提示词体现了 VLM-Loc 的核心思想：
1. **结构化推理**：将定位问题分解为"物体匹配"和"坐标推断"两个子任务
2. **可解释性**：`assignments` 字段让模型的推理过程可追溯
3. **部分匹配**：允许部分物体无法匹配（`grounded: false`），增强鲁棒性

---

## 16. 模块十二：可视化工具（drawing.py / rendering.py）

### 16.1 drawing.py — 2D 可视化

提供多种 2D 俯视图绘制函数，用于调试和检查数据质量。

**核心函数**：

| 函数 | 用途 |
|------|------|
| `plot_objects` | 绘制物体俯视图，按类别着色 |
| `plot_cell` | 绘制 cell 的 BEV 图，支持多种着色模式 |
| `plot_matches_in_best_cell` | 可视化物体匹配结果（绿箭头=正确，红箭头=错误） |
| `plot_pose_in_best_cell` | 绘制位姿在 best cell 中的位置和描述箭头 |
| `plot_cells_and_poses` | 全局视图，显示所有 cell 和 pose |

**着色模式**：
- `use_rgb=True`：使用物体原始 RGB 颜色
- `use_instances=True`：按实例 ID 随机着色
- 默认：按语义类别着色（使用 `CLASS_TO_COLOR`）

### 16.2 rendering.py — 3D 可视化

提供基于 pptk 库的 3D 交互式查看工具。

**核心函数**：

| 函数 | 用途 |
|------|------|
| `create_viewer` | 创建 3D 点云交互式查看器 |
| `get_orientations_manually` | 交互式标注位姿朝向 |
| `show_street_centers` | 显示场景点云和 cell 中心标记 |

---

## 17. 模块十三：空间邻域关系（add_relation.py）

**文件路径**：`datapreparation/kitti360pose/add_relation.py`

### 17.1 概述

为 CityRefer 数据集中的每个 cell 计算**8 方向邻域关系**。

### 17.2 邻域判断算法

```python
def check_neighbor(cell_x, cell_y, neigh_x, neigh_y):
    """判断两个 cell 是否为邻居（x 和 y 方向差距都不超过 10m）"""
    return abs(cell_x - neigh_x) <= 10 and abs(cell_y - neigh_y) <= 10

def get_direction(cell_x, cell_y, neigh_x, neigh_y):
    """根据坐标差判断邻居方向"""
    dx = neigh_x - cell_x
    dy = neigh_y - cell_y
    # 假设 cell 间距为 10m 的整数倍
    if dx == 10 and dy == 0: return "east"
    if dx == -10 and dy == 0: return "west"
    if dx == 0 and dy == 10: return "north"
    if dx == 0 and dy == -10: return "south"
    # 对角线方向
    if dx == 10 and dy == 10: return "northeast"
    # ...
```

### 17.3 输出格式

```json
{
  "cell_id_1": {
    "east": "neighbor_cell_id",
    "west": null,
    "north": "neighbor_cell_id",
    "south": null,
    "northeast": null,
    "northwest": "neighbor_cell_id",
    "southeast": null,
    "southwest": null
  }
}
```

---

## 18. 模块十四：数据分割（split_data.py）

**文件路径**：`split_data.py`

### 18.1 概述

对 CityRefer/Sensaturban 数据集进行**实例级分割**，为数据准备提供 train/val/test 划分。

### 18.2 核心逻辑

```python
def preparePthFiles(args, files, split, outPutFolder, false_segments):
    for file in files:
        # 1. 读取 PLY 文件
        xyz, rgb, labels = DataProcessing.read_ply_data(file)

        # 2. 读取实例分割信息
        instance_db = json.load(open(seg_file))

        # 3. 处理每个实例
        for instance in instance_db["segGroups"]:
            # 过滤 Cars (9) 和 Bikes (11)
            if main_semantic in [9, 11]:
                continue

            # 过滤点数 < 500 的实例
            if instance_xyz.shape[0] < 500:
                continue

            # 颜色归一化到 [-1, 1]
            instance_colors = instance_rgb.astype(np.float32) / 127.5 - 1

            # 保存为 .pth 文件
            torch.save(scene_instances, outFilePath)
```

**过滤策略**：
- 移除汽车和自行车（动态物体，不适合作为定位参考）
- 移除点数不足 500 的小实例

---

## 19. 代码质量评估与优化建议

### 19.1 代码质量评估

| 维度 | 评分 | 说明 |
|------|:----:|------|
| 可读性 | ⭐⭐⭐⭐ | 函数命名清晰，逻辑分层合理 |
| 可维护性 | ⭐⭐⭐ | 存在代码重复（两个 dataset_generation 脚本高度相似） |
| 性能 | ⭐⭐⭐⭐ | 使用 NumPy 向量化操作和并行处理 |
| 文档 | ⭐⭐⭐ | 核心函数有注释，但部分缺少详细说明 |
| 测试 | ⭐⭐ | 仅有 test.ipynb，缺少单元测试 |

### 19.2 优化建议

#### 1. 消除代码重复（高优先级）

`dataset_generation_semantics_cityloc-k.py` 和 `dataset_generation_semantics_cityloc-c.py` 有约 80% 的代码重复。建议提取公共逻辑到基类或工具模块：

```python
# 建议重构为
class DatasetGenerator:
    def __init__(self, args, bev_range, image_size):
        ...

    def create_scene_graph(self, centers_info): ...
    def ground_pose_to_image_scene_graph_v2(self, ...): ...
    def create_text_image_pairs(self, ...): ...
    def create_train_info(self, ...): ...

class KITTIGenerator(DatasetGenerator):
    def get_pose_cell_id(self, pose): return pose.cell_id

class CityReferGenerator(DatasetGenerator):
    def get_pose_cell_id(self, pose): return pose.eval_cell_id
```

#### 2. 硬编码路径（高优先级）

多个文件中存在硬编码的绝对路径：
```python
# dataset_generation_semantics_cityloc-k.py:7
sys.path.append("/home/data_sata/vlmloc/vlm-loc")

# dataset_generation_semantics_cityloc-c.py:284
cell_image_path = '/data/kang/vlmloc/' + cell_image_path
```
建议使用配置文件或环境变量管理路径。

#### 3. 异常处理（中优先级）

`recall.py` 中的异常处理可以改进：
```python
# 当前代码
except:
    invalid += 1
    continue

# 建议
except (json.JSONDecodeError, KeyError, TypeError) as e:
    invalid += 1
    logger.warning(f"Skipping invalid line {i}: {e}")
    continue
```

#### 4. 类型注解（低优先级）

建议为所有公共函数添加完整的类型注解，提高代码可读性和 IDE 支持。

#### 5. 配置管理（中优先级）

建议将分散在各文件中的配置参数（BEV_RANGE=50, IMAGE_SIZE=224 等）统一到一个配置文件中。

---

## 附录：关键数据流示例

### A. 训练状态数据流（从点云到训练样本）

```
1. 原始数据加载
   KITTI-360 PLY 文件 → load_points() → (xyz, rgb, lbl, iid)
   CityRefer torch 文件 → convert_cityrefer_objects() → Object3d 列表

2. 物体提取与处理
   (xyz, rgb, lbl, iid) → extract_objects() → Object3d 列表
   Object3d 列表 → gather_objects_both() → 合并 + 降采样 + 过滤

3. Cell 创建与 BEV 渲染
   Object3d 列表 + bbox → create_cell() → Cell 对象
     ├─ stuff 物体 → DBSCAN 聚类 → 分割为独立实例
     └─ instance 物体 → 裁剪 + 比例过滤
   Cell 对象 → generate_sem_bev_image() → BEV 图像 (224×224 PNG) + centers_info

4. 位姿描述生成
   位姿位置 + Cell → describe_pose_in_pose_cell() → DescriptionPoseCell 列表
   DescriptionPoseCell + best Cell → ground_pose_to_best_cell() → DescriptionBestCell 列表
   → Pose 对象（含 6 条空间关系描述）

5. 训练样本组装
   Pose + centers_info → create_scene_graph() → 场景图 JSON
   Pose + 场景图 → ground_pose_to_image_scene_graph_v2() → assignments
   Pose.descriptions → 自然语言查询
   Pose.pose_w → project_points_to_pixels() → point_2d

6. 最终训练样本（ms-swift 格式）
   {
     "messages": [
       {"role": "user", "content": "{\"nodes\": [{\"node_id\": 0, \"label\": \"road\", ...}]}"},
       {"role": "user", "content": "<image>  The target location is north of a gray car and east of a dark-green vegetation."},
       {"role": "assistant", "content": "{\"assignments\": [{\"object_label\": \"car\", \"grounded\": true, \"matched_node\": 3}, ...], \"point_2d\": [112, 56]}"}
     ],
     "images": ["path/to/bev_image.png"]
   }

7. LoRA 微调 → 输出 checkpoint 权重
```

### B. 推理状态数据流（从用户输入到定位结果）

```
1. 用户输入（三条信息）
   ① BEV 鸟瞰图：一张 224×224 的 PNG 图片
   ② 场景图 JSON：
      {
        "nodes": [
          {"node_id": 0, "label": "road", "pixel_center": [112, 56]},
          {"node_id": 1, "label": "car", "pixel_center": [80, 30]},
          {"node_id": 2, "label": "vegetation", "pixel_center": [45, 180]}
        ]
      }
   ③ 自然语言描述："The target location is north of a gray car and east of a dark-green vegetation."

2. VLM 推理（加载 Qwen3-VL-8B + LoRA 权重）
   输入：BEV 图 + 场景图 + 文本描述 + system_prompt
   VLM 内部推理过程：
     ├─ 解析文本，提取物体短语："gray car", "dark-green vegetation"
     ├─ 在场景图中匹配节点：car → node 1, vegetation → node 2
     ├─ 基于空间关系推断目标位置
     └─ 输出 JSON

3. VLM 输出
   {
     "assignments": [
       {"object_label": "car", "grounded": true, "matched_node": 1},
       {"object_label": "vegetation", "grounded": true, "matched_node": 2}
     ],
     "point_2d": [95, 42]    ← 预测的像素坐标
   }

4. 后处理（可选）
   像素坐标 → 世界坐标：pixels_to_world_from_center(95, 42, center, 50, 224)
   → 得到真实世界中的米级坐标

5. 评估（仅测试时）
   预测 point_2d vs 真实 point_2d → 像素距离 × 0.2232 m/px → R@5m / R@10m / R@15m
```

> [!abstract] 一句话总结
> **训练状态**解决的是"如何从 3D 点云自动生成大量训练样本"的问题；**推理状态**解决的是"给定一张 BEV 图和一段文字，如何定位"的问题。两者通过训练好的 LoRA 权重桥接。

---

> 本文档由项目代码精讲师自动生成，基于对 VLM-Loc 项目所有源代码的逐行分析。

## 相关链接

- [[VLM-Loc|VLM-Loc]]
