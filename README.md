# PyTorch 多 GPU 同时训练操作指南

> 🧠 面向 **第一次尝试多 GPU 并行训练** 的完整教程  
> ✅ 帮助你理解每一步原理，并能轻松迁移到自己的其他项目

---

## 📁 项目结构
```text
├── Multi-GPU-simultaneous-training-operation-readme/   # 源码与示例代码实现
├── result_informer02_random10000GPU2/                  # 完整数据集运行结果（日志 / 模型 / 图表）
├── Multi-GPU-Training.png                              # 多 GPU 工作状态截图
├── datatry.csv                                         # 示例数据集（10,000 行 × 12 列）
├── informer.py                                         # Minimal Informer 多 GPU 训练脚本
└── README.md                                           # 本说明文件
```

---

## 🧱 目录

1. [背景知识](#背景知识)
2. [快速开始：5 个关键步骤](#快速开始5个关键步骤)
3. [通用模板](#通用模板)
4. [常见错误与排查](#常见错误与排查)
5. [训练口诀](#训练口诀)

---

## 🧠 背景知识

- 在 **PyTorch** 中，模型默认运行在单个 GPU（例如 `cuda:0`）
- 如果有多个 GPU，可**并行处理数据**以提升训练速度
- PyTorch 提供两种多卡训练方式：
  1. **`nn.DataParallel`**：单进程多线程，最易上手，适合中等规模训练
  2. **`torch.distributed` / `torchrun`**：多进程高效并行，适合大型集群训练

> 本文使用 `nn.DataParallel`，更易理解。如需更优性能，可进一步迁移到 `DistributedDataParallel`

---

## 🚀 快速开始：5 个关键步骤

> 以下示例假设已有模型 `MyModel` 和数据迭代器 `train_loader` / `test_loader`

### 1️⃣ 设定主设备为 `cuda:0`
```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 2️⃣ 封装模型为多 GPU 模型
```python
from torch import nn

model = MyModel()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs …")
    model = nn.DataParallel(model)   # 多卡封装
model = model.to(device)
```

### 3️⃣ 训练阶段：数据批也要 `.to(device)`
```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)  # DataParallel 自动分发数据和收集梯度
    # loss, backward, step …
```

### 4️⃣ 验证 / 推理阶段同样迁移到 GPU
```python
model.eval()
with torch.no_grad():
    for xb in test_loader:
        xb   = xb.to(device)
        pred = model(xb)
        # 评估 …
```

### 5️⃣ 可选：打印 GPU 状态信息
```python
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

---

## 🧩 通用模板

```python
import torch
from torch import nn

# 设定主设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建模型
model = MyModel()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs …")
    model = nn.DataParallel(model)
model = model.to(device)

# 训练循环
for xb, yb in dataloader:
    xb, yb = xb.to(device), yb.to(device)
    pred   = model(xb)
    # loss, backward, optimizer.step() …
```

---

## ❗ 常见错误与排查

| 报错 / 现象 | 可能原因 | 快速解决方案 |
|-------------|-----------|----------------|
| `RuntimeError: module must have its parameters on cuda:0 but got cuda:1` | 模型未在主设备上 | 封装完再调用 `.to(device)` |
| `Expected all tensors to be on the same device` | 输入数据或标签未 `.to()` | 所有输入数据都执行 `.to(device)` |
| 多 GPU 未提速 | 模型太小 / batch 太小 | 尝试增大 `batch_size` 或换用 DDP |
| `AttributeError: 'DataParallel' object has no attribute 'xxx'` | 访问了封装后的模型属性 | 用 `model.module.xxx` 来访问内部属性 |

---

## 🧠 训练口诀（记忆公式）

> 多卡训练五要素，口诀如下：

```text
设主卡 → device = cuda:0  
封模型 → nn.DataParallel(model)  
放主卡 → model.to(device)  
数据转 → xb, yb = xb.to(device), yb.to(device)  
调模型 → 正常 forward / backward
```

---

````markdown
# PyTorch 多 GPU 同时训练操作指南
> 面向 **第一次尝试多 GPU 并行训练** 的完整教程，帮助你理解每一步原理，并能轻松迁移到自己的其他项目。
---
## 项目结构
```text
├── Multi-GPU-simultaneous-training-operation-readme/   # 源码与示例代码实现
├── result_informer02_random10000GPU2/                  # 完整数据集运行结果可视化（日志 / 模型权重 / 图表）
├── Multi-GPU-Training.png                              # 多 GPU 工作显示
├── datatry.csv                                         # 示例数据集（10 000 行 × 12 列）
├── informer.py                                         # Minimal Informer 训练脚本（实现多GPU训练informer模型训练）
└── README.md                                           # 本说明文件
```
---
## 目录
1. [背景知识](#背景知识)
2. [快速开始：5 个关键步骤](#快速开始5-个关键步骤)
3. [通用模板](#通用模板)
4. [常见错误与排查](#常见错误与排查)
5. [训练口诀](#训练口诀)
---
## 背景知识
- 在 **PyTorch** 中，模型默认运行在单块 GPU（如 `cuda:0`）。
- 为了 **加速训练**，可以同时利用多块 GPU 处理不同批次数据。
- PyTorch 提供两种主流方案：  
  1. **`nn.DataParallel`**（单进程多线程）——易上手，代码改动最小；当 GPU 数量很多时效率一般。  
  2. **`torch.distributed`** / **`torchrun`**（多进程）——扩展性更佳，适合大规模集群；配置更复杂。
> 本文示例采用最容易上手的 **`nn.DataParallel`**；如需更高性能，可迁移到 `DistributedDataParallel`。
---
## 快速开始：5 个关键步骤  
> 假设已有模型 `MyModel` 和数据迭代器 `train_loader` / `test_loader`。

### 1️⃣ 设定主设备 `cuda:0`
```python
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
````
### 2️⃣ 封装模型为多 GPU 模型
```python
from torch import nn
model = MyModel()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs …")
    model = nn.DataParallel(model)           # 仅需这一行
model = model.to(device)
```
### 3️⃣ **训练阶段**：数据批也移动到主设备
```python
for xb, yb in train_loader:
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)           # DataParallel 自动切分数据并汇总梯度
    # loss, backward, step …
```
### 4️⃣ **验证 / 推理阶段** 同样 `.to(device)`
```python
model.eval()
with torch.no_grad():
    for xb in test_loader:
        xb   = xb.to(device)
        pred = model(xb)
        # 评估 …
```
### 5️⃣ （可选）打印 GPU 状态
```python
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```
---

## 通用模板

```python
import torch
from torch import nn
# ---- 设备 ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ---- 模型 ----
model = MyModel()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs …")
    model = nn.DataParallel(model)
model = model.to(device)
# ---- 训练循环 ----
for xb, yb in dataloader:
    xb, yb = xb.to(device), yb.to(device)
    pred   = model(xb)
    # loss, backward, optimizer.step() …
```
---

## 常见错误与排查

| 报错 / 现象                                                                  | 可能原因           | 快速解决                                                    |
| ------------------------------------------------------------------------ | -------------- | ------------------------------------------------------- |
| `RuntimeError: module must have its parameters on cuda:0 but got cuda:1` | 主设备不一致         | 确保 `device` 为 `cuda:0`，并在 **封装后** 调用 `model.to(device)` |
| `Expected all tensors to be on the same device`                          | 数据或标签未移动       | 对 **所有** 输入和标签执行 `.to(device)`                          |
| 多 GPU 未提速                                                                | 模型过小或 batch 太小 | 增大 `batch_size`、改用 `DistributedDataParallel`，或仅用单卡      |
| `AttributeError: 'DataParallel' object has no attribute 'xxx'`           | 直接访问封装后模型属性    | 用 `model.module.xxx` 访问内部模型                             |

---

## 训练口诀
> **多卡训练五要素**
>
> 1. **设主卡** → `device = cuda:0`
> 2. **封模型** → `nn.DataParallel(model)`
> 3. **放主卡** → `model.to(device)`
> 4. **数据转** → `xb, yb = xb.to(device), yb.to(device)`
> 5. **调模型** → 正常 `forward / backward`

