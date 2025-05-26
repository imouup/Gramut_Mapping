import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from matplotlib import font_manager


# 原数据加载和预处理函数保持不变
def load_and_preprocess(data_dir):
    """加载并预处理数据"""
    channels = ['R', 'G', 'B']
    tensor_3d = np.zeros((64, 64, 3, 3))  # 形状: [行, 列, LED通道, 颜色分量]

    for led_idx, led in enumerate(channels):
        for color_idx, color in enumerate(channels):
            df = pd.read_csv(f"{data_dir}/{led}_{color}.csv", header=None)
            tensor_3d[:, :, led_idx, color_idx] = df.values

    return tensor_3d / 255.0  # 归一化到[0,1]


# 新增神经网络相关函数
def prepare_training_data(tensor_3d):
    """准备训练数据"""
    # 输入为实际测量值，目标为理想响应
    X = tensor_3d.reshape(-1, 3)  # (64 * 64 * 3, 3)
    Y = np.zeros_like(X)

    # 构建目标矩阵（理想响应）
    for c in range(3):
        Y[c::3, c] = 220 / 255  # 每个颜色通道每隔3个样本设置目标值

    return X, Y


class CorrectionNet(nn.Module):
    """全连接神经网络模型"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


def train_model(X, Y, num_epochs=200, batch_size=1024):
    """训练神经网络模型"""
    # 转换数据为Tensor
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = CorrectionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 记录损失
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    return model, losses


def apply_neural_correction(tensor_3d, model):
    """应用神经网络校正"""
    corrected = np.zeros_like(tensor_3d)
    with torch.no_grad():
        for i in range(64):
            for j in range(64):
                for c in range(3):
                    input_data = tensor_3d[i, j, c, :]
                    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
                    output = model(input_tensor).numpy().squeeze()
                    corrected[i, j, c, :] = output
    return np.clip(corrected, 0, 1)


# 原损失计算和绘图函数保持不变
def calculate_loss(original_data, corrected_data):
    """计算整体颜色损失（MSE）"""
    target = np.zeros_like(original_data)
    target[:, :, 0, 0] = 220 / 255  # R
    target[:, :, 1, 1] = 220 / 255  # G
    target[:, :, 2, 2] = 220 / 255  # B

    original_loss = np.mean((original_data - target) ** 2)
    corrected_loss = np.mean((corrected_data - target) ** 2)
    return original_loss, corrected_loss


def plot_training_loss(losses, output_path="training_loss.png"):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training Process')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"训练损失图已保存至：{output_path}")

def set_chinese_font():
    """设置中文字体（补充到神经网络版本的代码中）"""

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False


def plot_correction_comparison(original, corrected, output_path="neural_correction_comparison.png"):
    """绘制矫正前后对比图（与原始矩阵方法通用）"""
    set_chinese_font()  # 确保已调用字体设置

    plt.figure(figsize=(10, 8))
    plt.suptitle("神经网络校正效果", y=0.95, fontsize=14)

    channels = ['Red', 'Green', 'Blue']
    for idx, ch in enumerate(channels):
        # 原始数据
        plt.subplot(3, 2, 2 * idx + 1)
        orig_img = original[:, :, idx, :].transpose(1, 0, 2) * 255
        plt.imshow(orig_img.astype(np.uint8))
        plt.title(f"原始 {ch} 通道输出")
        plt.axis('off')

        # 校正数据
        plt.subplot(3, 2, 2 * idx + 2)
        corr_img = corrected[:, :, idx, :].transpose(1, 0, 2) * 255
        plt.imshow(corr_img.astype(np.uint8))
        plt.title(f"校正后 {ch} 通道输出")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"对比图已保存至：{output_path}")

# 主程序流程调整
if __name__ == "__main__":
    # 加载数据
    data_dir = "D:/Pyproject/CSV"
    original_data = load_and_preprocess(data_dir)

    # 准备训练数据
    X, Y = prepare_training_data(original_data)

    # 训练神经网络
    model, train_losses = train_model(X, Y, num_epochs=200)

    # 保存模型
    torch.save(model.state_dict(), "correction_model.pth")

    # 应用校正
    corrected_data = apply_neural_correction(original_data, model)

    # 计算损失
    original_loss, corrected_loss = calculate_loss(original_data, corrected_data)
    print(f"\n原始损失: {original_loss:.6f} | 校正后损失: {corrected_loss:.6f}")

    # 保存结果
    os.makedirs("neural_output", exist_ok=True)
    channels = ['R', 'G', 'B']
    for ch_idx, ch in enumerate(channels):
        output = (corrected_data[:, :, ch_idx, ch_idx] * 255).astype(int)
        df = pd.DataFrame(output)
        df.to_csv(f"neural_output/corrected_{ch}.csv", header=False, index=False)

    # 绘制训练损失
    plot_training_loss(train_losses)

    # 绘制对比图（使用原函数）
    set_chinese_font()
    plot_correction_comparison(original_data, corrected_data, output_path="neural_correction_comparison.png")