#!/usr/bin/env python
# process_bt2020_image.py

import sys
import os
import torch
import numpy as np  # 确保导入 numpy
from PIL import Image

# 尝试从 Q1.py 导入 MLP 类和 M1 矩阵
try:
    # M1_q1 是 Q1.py 中定义的 NumPy BT.2020 to XYZ 矩阵
    from Q1 import MLP, M1 as M1_q1
except ImportError:
    print("错误：无法从 Q1.py 导入 'MLP' 类或 'M1' 矩阵。")
    print("请确保 Q1.py 与此脚本在同一目录中，或在您的 PYTHONPATH 中。")
    sys.exit(1)

# 导入 colour-science 工具库 (oetf_sRGB 将被手动定义)
try:
    from colour import XYZ_to_RGB, RGB_to_XYZ
    from colour.models import (
        oetf_inverse_BT2020,
        # oetf_sRGB, # <<< 从此处移除，我们将手动定义它
        RGB_COLOURSPACE_BT2020,
        RGB_COLOURSPACE_sRGB
    )
except ImportError:
    print("错误：未找到 'colour-science' 库或其核心组件。")
    print("请确保已正确安装： pip install colour-science")
    sys.exit(1)

# --- 全局配置和 PyTorch 设置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
M1_NP = M1_q1  # 使用从 Q1.py 导入的 M1 矩阵


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 开始：手动定义的 oetf_sRGB 函数 +++
def oetf_sRGB_manual(linear_rgb: np.ndarray) -> np.ndarray:
    """
    手动将 sRGB OETF (光电转换函数) 应用于线性 sRGB 数据。

    参数:
        linear_rgb: 一个 numpy 数组，包含线性 sRGB 值，通常在 [0, 1] 范围内。

    返回:
        一个 numpy 数组，包含非线性 sRGB 值。
    """
    L = np.asarray(linear_rgb)  # 确保输入是 numpy 数组

    # 定义 sRGB OETF 的两个部分条件
    condition = L <= 0.0031308

    # 计算 L <= 0.0031308 的部分
    part1 = 12.92 * L

    # 计算 L > 0.0031308 的部分
    # np.power 对于非整数指数，需要 L >= 0 以避免复数结果。
    # 由于 L > 0.0031308，这里 L 保证为正。
    part2 = 1.055 * np.power(L, 1.0 / 2.4) - 0.055

    # 根据条件应用计算
    non_linear_rgb = np.where(condition, part1, part2)

    return non_linear_rgb


# +++ 结束：手动定义的 oetf_sRGB 函数 +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# --- MLP 模型加载函数 ---
def load_mlp_model(ckpt_path: str, device: torch.device) -> MLP:
    """加载训练好的 MLP 模型。"""
    model = MLP().to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except FileNotFoundError:
        print(f"错误：模型检查点文件未在 {ckpt_path} 找到。")
        sys.exit(1)
    except Exception as e:
        print(f"错误：加载模型检查点时发生错误： {e}")
        sys.exit(1)
    model.eval()
    return model


# --- 图像处理函数 ---
def process_bt2020_to_srgb_image(image_path: str, model_path: str, output_path: str):
    """
    加载 BT.2020 图像，使用直接转换和 MLP（针对 OOG 像素）进行处理，
    并将其另存为 sRGB 图像。
    """
    print(f"正在处理图像: {image_path}")
    print(f"使用 MLP 模型: {model_path}")
    print(f"输出将保存至: {output_path}")
    print(f"使用设备: {DEVICE}")

    # 1. 加载图像
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：输入图像文件未在 {image_path} 找到。")
        sys.exit(1)
    except Exception as e:
        print(f"错误：打开图像时发生错误： {e}")
        sys.exit(1)

    img_np = np.array(img)
    original_shape = img_np.shape  # H, W, C
    if original_shape[2] != 3:
        print("错误：图像不包含3个颜色通道。")
        sys.exit(1)
    num_pixels = original_shape[0] * original_shape[1]

    # 2. 将像素值从 [0, 255] 归一化到 [0, 1]
    img_np_float = img_np.astype(np.float32) / 255.0

    # 3. 应用 BT.2020 的反 OETF (线性化 BT.2020 RGB)
    # 使用 colour.models 中的 oetf_inverse_BT2020
    # 在应用前裁剪到 [0,1] 以确保输入有效
    linear_bt2020_rgb = oetf_inverse_BT2020(np.clip(img_np_float, 0.0, 1.0))
    linear_bt2020_flat = linear_bt2020_rgb.reshape(num_pixels, 3)

    # --- CPU 路径进行初始色域检查 ---
    # 4. 将线性 BT.2020 RGB 转换为 XYZ
    # 使用从 Q1.py 导入的 M1_NP 矩阵
    xyz_pixels_np = np.dot(linear_bt2020_flat, M1_NP.T)

    # 5. 将 XYZ 转换为线性 sRGB (作为色域内像素的候选值)
    # 使用 colour 库的 XYZ_to_RGB 和 colour.models 的 RGB_COLOURSPACE_sRGB
    linear_srgb_candidate_np = XYZ_to_RGB(
        xyz_pixels_np,
        colourspace=RGB_COLOURSPACE_sRGB
    )

    # 6. 识别 sRGB 的色域外 (OOG) 像素
    # 如果任何 sRGB 通道 < 0 或 > 1 (允许微小容差)，则像素为 OOG
    oog_mask_np = np.any((linear_srgb_candidate_np < -1e-7) | (linear_srgb_candidate_np > 1.0 + 1e-7), axis=1)
    ingamut_mask_np = ~oog_mask_np

    num_oog_pixels = np.sum(oog_mask_np)
    print(f"总像素数: {num_pixels}, sRGB 色域外像素数: {num_oog_pixels}")

    # 准备最终的 sRGB 线性数组以存储结果
    final_linear_srgb_flat = np.zeros_like(linear_srgb_candidate_np)

    # 7. 处理色域内像素：使用直接转换的值，并裁剪到 [0,1]
    final_linear_srgb_flat[ingamut_mask_np] = np.clip(linear_srgb_candidate_np[ingamut_mask_np], 0.0, 1.0)

    # 8. 使用 MLP 处理色域外 (OOG) 像素
    if num_oog_pixels > 0:
        print(f"使用 MLP 映射 {num_oog_pixels} 个 OOG 像素...")
        oog_bt2020_linear_np = linear_bt2020_flat[oog_mask_np]
        oog_bt2020_linear_ts = torch.from_numpy(oog_bt2020_linear_np).float().to(DEVICE)

        model = load_mlp_model(model_path, DEVICE)
        with torch.no_grad():
            # MLP 模型直接接收线性 BT.2020 RGB 值
            mapped_srgb_linear_ts = model(oog_bt2020_linear_ts)
        mapped_srgb_linear_np = mapped_srgb_linear_ts.cpu().numpy()

        # MLP 输出预期在 [0,1] (因 Sigmoid 激活)。为安全起见进行裁剪。
        final_linear_srgb_flat[oog_mask_np] = np.clip(mapped_srgb_linear_np, 0.0, 1.0)
    else:
        print("未找到 OOG 像素。无需 MLP 映射。")

    # 9. 将扁平化的像素数组重塑为图像尺寸
    final_linear_srgb_image = final_linear_srgb_flat.reshape(original_shape)

    # 10. 应用 sRGB OETF (伽马校正以供显示)
    # 在应用 OETF 前裁剪到 [0,1] 是一个好习惯
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # 此处调用手动定义的函数
    final_nonlinear_srgb_image = oetf_sRGB_manual(np.clip(final_linear_srgb_image, 0.0, 1.0))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # 11. 从 [0, 1] 反归一化到 [0, 255] 并转换为 uint8
    final_image_np_uint8 = (np.clip(final_nonlinear_srgb_image, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 12. 保存最终的 sRGB 图像
    try:
        output_img = Image.fromarray(final_image_np_uint8)
        output_img.save(output_path)
        print(f"成功处理图像并保存至: {output_path}")
    except Exception as e:
        print(f"错误：保存输出图像时发生错误： {e}")
        sys.exit(1)


# --- 主执行块 ---
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\n用法: python process_bt2020_image.py <输入bt2020图像路径> <模型.pth路径> <输出srgb图像路径>\n")
        print("示例:")
        print("  python process_bt2020_image.py input.tif model/my_model_final.pth output.png\n")
        sys.exit(1)

    input_image_path_arg = sys.argv[1]
    model_ckpt_path_arg = sys.argv[2]
    output_image_path_arg = sys.argv[3]

    process_bt2020_to_srgb_image(input_image_path_arg, model_ckpt_path_arg, output_image_path_arg)