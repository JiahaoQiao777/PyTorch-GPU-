#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_informer_gpu.py
---------------------------------
• 输入特征: 前 10 列
• 输出目标: 后 2 列 (Pressure, SWAT)
• 训练集: 随机 10 000 行
• 测试集: 剩余行
• 自动检测 CUDA；优先使用 GPU1 (CUDA:1)；无 GPU 时退回 CPU
• 模型: 迷你 Informer 回归器（Encoder‑Only）
    ‑ 将 10 个数值特征视作长度为 10 的时间序列
    ‑ Embedding→位置编码→2× InformerEncoderLayer
    ‑ 每层: Multi‑Head Attention + Feed‑Forward (+ 可选 Conv1d Distilling)
    ‑ 全局平均池化→MLP 输出 2 维
NOTE: 为演示方便，ProbSparse 注意力近似以正常 MultiHeadAttention 代替（seq_len=10 影响可忽略）。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ===================== 0. 全局设置 =====================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

debug = False  # 设 True 可快速跑通（epoch 减少）

# ===================== 1. 结果目录 =====================
result_dir = "Multi-GPU-simultaneous-training-operation-results"
os.makedirs(result_dir, exist_ok=True)

# ===================== 2. 读取数据 =====================
df = pd.read_csv("datatry.csv", header=0)
X = df.iloc[:, :10].values.astype(np.float32)  # (N, 10)
y = df.iloc[:, -2:].values.astype(np.float32)  # (N, 2)
# X 对应的前 10 列列名
x_cols = df.columns[:10]
# y 对应的最后 2 列列名
y_cols = df.columns[-2:]
print("X 列名:", list(x_cols))
print("y 列名:", list(y_cols))

# ===================== 3. 标准化 =====================
scaler_x = StandardScaler(); scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X); y_scaled = scaler_y.fit_transform(y)

# ===================== 4. 划分训练 / 测试 =====================
idx_all = np.arange(len(X_scaled))
train_idx = np.random.choice(idx_all, size=10_000, replace=False)
test_idx = np.setdiff1d(idx_all, train_idx)
X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
X_test,  y_test  = X_scaled[test_idx],  y_scaled[test_idx]

# 转张量
X_train_t = torch.from_numpy(X_train); y_train_t = torch.from_numpy(y_train)
X_test_t  = torch.from_numpy(X_test)

# ===================== 5. 设备信息 =====================
# print(f"Available GPUs: {torch.cuda.device_count()}")
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else
#                       ("cuda:0" if torch.cuda.is_available() else "cpu"))
# print(f"Using device: {device}")
# ===================== 5. 设备信息 =====================
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ✅ 正确设定 device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== 6. 数据加载器 =====================
BATCH = 1024
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH, shuffle=True)

# ===================== 7. Positional Encoding =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()

# ===================== 8. Informer Encoder Layer =====================
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_ff=128, dropout=0.1, distil=True):
        super().__init__()
        self.distil = distil
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if distil:
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
            self.norm_conv = nn.LayerNorm(d_model)
    def forward(self, x):  # x: (B, L, D)
        attn_out,_ = self.mha(x,x,x, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        if self.distil and x.size(1) > 1:
            x = self.conv(x.transpose(1,2)).transpose(1,2)  # halve seq_len
            x = self.norm_conv(x)
        return x

# ===================== 9. Informer 回归模型 =====================
class InformerRegressor(nn.Module):
    def __init__(self, seq_len=10, d_model=64, nhead=4, e_layers=2, dim_ff=128, out_dim=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, dim_ff, dropout, distil=(i!=e_layers-1))
            for i in range(e_layers)
        ])
        self.head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, out_dim))
    def forward(self, x):  # x: (B, seq_len)
        x = x.unsqueeze(-1)                    # (B, L, 1)
        x = self.embed(x)                     # (B, L, D)
        x = self.pos(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)                     # 池化
        return self.head(x)

# model = InformerRegressor().to(device)
# ===== 多GPU支持 =====
model = InformerRegressor()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = nn.DataParallel(model)
model = model.to(device)
# print(model)
print(model)

# ===================== 10. 训练 =====================
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS = 300 if not debug else 5

print("\n开始训练 Informer 模型…")
for epoch in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        running += loss.item()*xb.size(0)
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{EPOCHS} | Train MSE: {running/len(train_loader.dataset):.6f}")

# ===================== 11. 预测 & 评估 =====================
model.eval()
TEST_BATCH = 4096
pred_chunks=[]
with torch.no_grad():
    for xb in DataLoader(X_test_t, batch_size=TEST_BATCH, shuffle=False):
        xb = xb.to(device)
        pred_chunks.append(model(xb).cpu())

y_pred_scaled = torch.cat(pred_chunks, dim=0).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

r2   = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
l2e  = np.linalg.norm(y_true - y_pred)

report = (
    f"Evaluation Metrics:\n"
    f"R² Score: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nL2 Error: {l2e:.4f}\n"
)
with open(os.path.join(result_dir, "evaluation_results.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("\n"+report)

# ===================== 12. 可视化 =====================
SAMPLE = 1000
idx = np.linspace(0, len(y_true)-1, SAMPLE, dtype=int)

plt.figure(figsize=(10,5));
plt.plot(idx, y_true[idx,0], label="True Pressure", ls="--", marker="o", ms=3)
plt.plot(idx, y_pred[idx,0], label="Predicted Pressure", ls="-", marker="x", ms=3)
plt.legend(); plt.title("True vs Predicted Pressure (Sampled)")
plt.savefig(os.path.join(result_dir, "sampled_true_vs_predicted_pressure.png")); plt.close()

plt.figure(figsize=(10,5));
plt.plot(idx, y_true[idx,1], label="True Swat", ls="--", marker="o", ms=3)
plt.plot(idx, y_pred[idx,1], label="Predicted Swat", ls="-", marker="x", ms=3)
plt.legend(); plt.title("True vs Predicted Swat (Sampled)")
plt.savefig(os.path.join(result_dir, "sampled_true_vs_predicted_swat.png")); plt.close()

plt.figure(figsize=(5,5)); plt.scatter(y_true[idx,0], y_pred[idx,0], alpha=0.6)
plt.xlabel("True Pressure"); plt.ylabel("Predicted Pressure")
plt.title("Scatter: Pressure (Sampled)")
plt.savefig(os.path.join(result_dir, "sampled_scatter_pressure.png")); plt.close()

plt.figure(figsize=(5,5)); plt.scatter(y_true[idx,1], y_pred[idx,1], alpha=0.6)
plt.xlabel("True Swat"); plt.ylabel("Predicted Swat")
plt.title("Scatter: Swat (Sampled)")
plt.savefig(os.path.join(result_dir, "sampled_scatter_swat.png")); plt.close()

print("所有结果已保存至 `Multi-GPU-simultaneous-training-operation-results`！")
