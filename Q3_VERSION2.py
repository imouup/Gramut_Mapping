import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib import font_manager

def load_and_preprocess(data_dir):
    """加载并预处理数据"""
    channels = ['R', 'G', 'B']
    tensor_3d = np.zeros((64, 64, 3, 3))  # 形状: [行, 列, LED通道, 颜色分量]

    for led_idx, led in enumerate(channels):
        for color_idx, color in enumerate(channels):
            df = pd.read_csv(f"{data_dir}/{led}_{color}.csv", header=None)
            tensor_3d[:, :, led_idx, color_idx] = df.values

    return tensor_3d / 255.0  # 归一化到[0,1]


def calculate_calibration_matrix(tensor_3d):
    """计算校准矩阵并记录伪逆使用情况"""
    n, m, _, _ = tensor_3d.shape
    M = np.zeros((n, m, 3, 3))
    T = np.array([[220 / 255, 0, 0], [0, 220 / 255, 0], [0, 0, 220 / 255]])
    pseudo_pixels = []  # 记录需要伪逆的像素位置

    for i in range(n):
        for j in range(m):
            A = tensor_3d[i, j, :, :]
            try:
                inv_A = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                inv_A = np.linalg.pinv(A)
                pseudo_pixels.append((i, j))  # 记录坐标
            M[i, j, :, :] = inv_A @ T
    return M, pseudo_pixels


def calculate_loss(original_data, corrected_data):
    """计算整体颜色损失（MSE）"""
    target = np.zeros_like(original_data)
    target[:, :, 0, 0] = 220 / 255  # R
    target[:, :, 1, 1] = 220 / 255  # G
    target[:, :, 2, 2] = 220 / 255  # B

    original_loss = np.mean((original_data - target) ** 2)
    corrected_loss = np.mean((corrected_data - target) ** 2)
    return original_loss, corrected_loss


def apply_correction(input_data, calibration_matrix):
    """应用校准矩阵"""
    corrected = np.zeros_like(input_data)
    for i in range(64):
        for j in range(64):
            corrected[i, j, :, :] = calibration_matrix[i, j, :, :] @ input_data[i, j, :, :]
    return np.clip(corrected, 0, 1)


def save_correction_results(corrected_data, matrix_dir, output_dir):
    """保存结果并打印示例矩阵"""
    # 自动创建目录
    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 保存校准矩阵
    np.save(os.path.join(matrix_dir, "calibration_matrix.npy"), corrected_data)

    # 保存CSV并打印示例
    channels = ['R', 'G', 'B']
    for ch_idx, ch in enumerate(channels):
        output = (corrected_data[:, :, ch_idx, ch_idx] * 255).astype(int)
        df = pd.DataFrame(output)
        df.to_csv(os.path.join(output_dir, f"corrected_{ch}.csv"), header=False, index=False)


def set_chinese_font():
    """设置中文字体"""
    try:
        # 尝试使用系统自带的中文字体（根据系统选择合适字体）
        font_path = None

        # Windows系统常见中文字体
        if os.name == 'nt':
            font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi']
            for font in font_names:
                if font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    return

        # Linux/Mac系统常见中文字体
        else:
            font_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'  # 文泉驿微米黑
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
                return

        # 如果找不到系统字体，尝试使用思源字体
        font_path = os.path.join(os.path.dirname(__file__), 'SourceHanSansSC-Regular.otf')
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            return

        # 最后尝试默认字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统自带

    except Exception as e:
        print(f"字体设置失败: {str(e)}")
        print("请手动安装中文字体：")
        print("1. Windows系统推荐安装'微软雅黑'字体")
        print("2. Linux系统推荐安装文泉驿字体：sudo apt-get install fonts-wqy-microhei")
        print("3. Mac系统推荐使用系统自带的PingFang SC字体")


def plot_correction_comparison(original, corrected, output_path="correction_comparison.png"):
    """绘制矫正前后对比图（包含RGB三种输入情况）"""
    # 设置中文字体
    set_chinese_font()

    # 创建画布（保持原有代码不变）
    plt.figure(figsize=(10, 8))
    plt.suptitle("LED显示屏校正效果对比", y=0.95, fontsize=14)

    # 定义输入颜色名称和对应通道
    colors = ['Red', 'Green', 'Blue']
    led_channels = [0, 1, 2]  # R, G, B对应的LED通道索引

    # 生成每个子图
    for idx, (color_name, led_idx) in enumerate(zip(colors, led_channels)):
        # 提取原始和校正后的RGB分量
        orig_rgb = original[:, :, led_idx, :]  # 形状 (64, 64, 3)
        corr_rgb = corrected[:, :, led_idx, :]

        # 转换为0-255整数并转置为(H, W, C)格式
        orig_img = (orig_rgb * 255).astype(np.uint8).transpose(1, 0, 2)
        corr_img = (corr_rgb * 255).astype(np.uint8).transpose(1, 0, 2)

        # 原始图像子图
        plt.subplot(3, 2, 2 * idx + 1)
        plt.imshow(orig_img)
        plt.title(f"Original - {color_name} Output")
        plt.axis('off')

        # 校正后图像子图
        plt.subplot(3, 2, 2 * idx + 2)
        plt.imshow(corr_img)
        plt.title(f"Corrected - {color_name} Output")
        plt.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存至：{output_path}")

if __name__ == "__main__":
    # 加载数据
    data_dir = "D:/Pyproject/CSV"
    original_data = load_and_preprocess(data_dir)
    set_chinese_font()

    # 计算校准矩阵
    calibration_matrix, pseudo_pixels = calculate_calibration_matrix(original_data)

    # 报告伪逆使用情况
    if len(pseudo_pixels) > 0:
        print(f"需要伪逆矩阵的像素位置（共{len(pseudo_pixels)}个）:")
        for idx, (i, j) in enumerate(pseudo_pixels[:5]):  # 最多显示前5个
            print(f"({i}, {j})", end=" | " if (idx + 1) % 5 != 0 else "\n")
        if len(pseudo_pixels) > 5:
            print(f"...及其他{len(pseudo_pixels) - 5}个像素")
    else:
        print("所有像素均使用标准矩阵逆运算")

    # 应用校正
    corrected_data = apply_correction(original_data, calibration_matrix)

    # 计算并显示损失
    original_loss, corrected_loss = calculate_loss(original_data, corrected_data)
    print(f"\n原始损失: {original_loss:.6f} | 校正后损失: {corrected_loss:.6f}")

    # 保存结果并打印示例
    save_correction_results(corrected_data, "matrices", "output")

    # 检查越界
    for i, matrix in enumerate(calibration_matrix):
        # 将矩阵转换为numpy数组以便处理
        np_matrix = np.array(matrix)
        # 检查矩阵中的每个元素
        if np.all(np_matrix >= -2) and np.all(np_matrix <= 2):
            print(f"矩阵 {i + 1} 没有越界。")
        else:
            print(f"矩阵 {i + 1} 存在越界。")
            # 输出越界的元素
            out_of_bounds = np_matrix[(np_matrix < -2) | (np_matrix > 2)]
            print(f"越界元素: {out_of_bounds}")
    # 打印校准矩阵示例
    for i in range(0,64):
        for j in range(0,64):
            print("\n原矩阵:")
            print(original_data[i, j, :3, :3] * 255)
            print("\n校正后矩阵:")
            print(corrected_data[i, j, :3, :3] * 255)
            print("\n校准矩阵:")
            print(calibration_matrix[i, j, :3, :3])

    # 生成可视化
    plot_correction_comparison(original_data, corrected_data)
