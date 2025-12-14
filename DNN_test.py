import torch
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib
import pickle

# ================== 随机种子 & 设备 ==================
seed = 42
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rmse(y_pred, y_true):
  return np.sqrt(np.mean((y_pred - y_true) ** 2))


def R_2(y_pred, y_true):
  return 1 - ((y_pred - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(2048, 2048),
      nn.GELU(),
      nn.Linear(2048, 1024),
      nn.GELU(),
      nn.Linear(1024, 1)
    )

  def forward(self, input1):
    output = self.model(input1)
    return output

  def predict(self, input_):
    return self.model(input_)

# ================== 数据读取 ==================
with open('D1A1_ECFP_input.pkl', 'rb') as f:
  data_list = pickle.load(f)

tasks, train_dataset, val_dataset, test_dataset = data_list
tasks.append('s1-t1')

# ---------- 构造 y: 增加 s1-t1 ----------
train_y = train_dataset[:, 2048:2057]
train_s1_t1 = train_y[:, 0] - train_y[:, 6]
train_y = np.hstack((train_y, train_s1_t1.reshape(len(train_s1_t1), 1)))

test_y = test_dataset[:, 2048:2057]
test_s1_t1 = test_y[:, 0] - test_y[:, 6]
test_y = np.hstack((test_y, test_s1_t1.reshape(len(test_s1_t1), 1)))

val_y = val_dataset[:, 2048:2057]
val_s1_t1 = val_y[:, 0] - val_y[:, 6]
val_y = np.hstack((val_y, val_s1_t1.reshape(len(val_s1_t1), 1)))

# ================== 划分 train/val/test 输入输出 ==================
X_train = train_dataset[:, 0:2048]
X_train = torch.Tensor(X_train).to(device)

X_val = val_dataset[:, 0:2048]
X_val = torch.Tensor(X_val).to(device)

X_test = test_dataset[:, 0:2048]
X_test = torch.Tensor(X_test).to(device)

y_train_dataset = torch.Tensor(train_y).to(device)
y_val_dataset = torch.Tensor(val_y).to(device)
y_test_dataset = torch.Tensor(test_y).to(device)

# ================== 超参数 ==================
LR = 1e-3
# BATCH_SIZE = 32  # 当前示例仍然是全量训练，如果需要 mini-batch 可再改
NUM_EPOCHS = 1000
EPOCHS = list(range(NUM_EPOCHS))

# 只训练前两个任务
target_tasks = tasks[1:3]

all_train_loss = [] # 每个元素是一个 list，对应一个 task 的所有 epoch 的 train_loss
all_val_loss = []
all_test_loss = []

# ================== 逐任务训练 ==================
for task_name in target_tasks:
  print("Training task:", task_name)
  start_time = time.time()

  # 当前任务索引
  task_id = tasks.index(task_name)

  # 取出当前任务的 y
  y_train = y_train_dataset[:, task_id].reshape(-1, 1)
  y_val = y_val_dataset[:, task_id].reshape(-1, 1)
  y_test = y_test_dataset[:, task_id].reshape(-1, 1)

  # 网络 & 优化器
  net_Adam = Net().to(device)
  optimizer = torch.optim.AdamW(net_Adam.parameters(), lr=LR)
  loss_func = nn.L1Loss()

  train_loss_list = []
  val_loss_list = []
  test_loss_list = []

  # ---------- 训练循环 ----------
  for epoch in range(NUM_EPOCHS):
    net_Adam.train()
    # 全训练
    output = net_Adam(X_train)
    train_loss = loss_func(output, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 评估集 & 测试集 loss
    net_Adam.eval()
    with torch.no_grad():
      val_output = net_Adam(X_val)
      val_loss = loss_func(val_output, y_val)

      test_output = net_Adam(X_test)
      test_loss = loss_func(test_output, y_test)

    train_loss_list.append(train_loss.item())
    val_loss_list.append(val_loss.item())
    test_loss_list.append(test_loss.item())

    # 打印一点训练信息（可注释掉）
    if (epoch + 1) % 50 == 0:
      print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
         f"Train Loss: {train_loss.item():.6f} "
         f"Val Loss: {val_loss.item():.6f} "
         f"Test Loss: {test_loss.item():.6f}")

  end_time = time.time()
  print(task_name, "training time:", end_time - start_time, "seconds")

  all_train_loss.append(train_loss_list)
  all_val_loss.append(val_loss_list)
  all_test_loss.append(test_loss_list)

# ================== 转成 numpy 数组 ==================
# shape: (num_epochs, num_tasks)
all_train_loss_array = np.array(all_train_loss).T
all_val_loss_array = np.array(all_val_loss).T
all_test_loss_array = np.array(all_test_loss).T

epochs_np = np.arange(NUM_EPOCHS)

# ================== 找每个任务的最优 epoch（基于 val_loss 最小） ==================
for task_idx, task_name in enumerate(target_tasks):
  val_losses = all_val_loss_array[:, task_idx]
  best_epoch = np.argmin(val_losses)
  best_loss = val_losses[best_epoch]
  print(f"{task_name} : 最佳模型 epoch: {best_epoch}, val_loss: {best_loss}")

# ================== 保存到 Excel ==================
# 每一行对应一个 epoch
data_dict = {
  "epoch": epochs_np
}
for task_idx, task_name in enumerate(target_tasks):
  data_dict[f"{task_name}_train_loss"] = all_train_loss_array[:, task_idx]
  data_dict[f"{task_name}_val_loss"] = all_val_loss_array[:, task_idx]
  data_dict[f"{task_name}_test_loss"] = all_test_loss_array[:, task_idx]

loss_df = pd.DataFrame(data_dict)
# 写到一个 Excel 文件里
loss_df.to_excel("losses_all_tasks.xlsx", index=False)

# ================== 画图并保存 ==================
# 1. 训练集 loss 图
plt.figure()
for task_idx, task_name in enumerate(target_tasks):
  plt.plot(epochs_np, all_train_loss_array[:, task_idx], label=task_name)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_loss.png", dpi=300)
# plt.show()

# 2. 验证集 loss 图
plt.figure()
for task_idx, task_name in enumerate(target_tasks):
  plt.plot(epochs_np, all_val_loss_array[:, task_idx], label=task_name)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_loss.png", dpi=300)
# plt.show()

# 3. 测试集 loss 图
plt.figure()
for task_idx, task_name in enumerate(target_tasks):
  plt.plot(epochs_np, all_test_loss_array[:, task_idx], label=task_name)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_loss.png", dpi=300)
# plt.show()