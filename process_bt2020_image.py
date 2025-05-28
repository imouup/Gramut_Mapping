#!/usr/bin/env python
# process_bt2020_image.py

import sys
import os
import torch
import numpy as np
from PIL import Image
import math  # For CIE2000 pi constant if not using torch.pi

# Import matplotlib for colormap. This is a new dependency.
import matplotlib.pyplot as plt

# --- PyTorch and Q1.py Imports/Setup ---
try:
    from Q1 import MLP, M1 as M1_q1
except ImportError:
    print("错误：无法从 Q1.py 导入 'MLP' 类或 'M1' 矩阵。")
    print("请确保 Q1.py 与此脚本在同一目录中，或在您的 PYTHONPATH 中。")
    sys.exit(1)

try:
    from colour import XYZ_to_RGB, RGB_to_XYZ
    from colour.models import (
        oetf_inverse_BT2020,
        RGB_COLOURSPACE_BT2020,
        RGB_COLOURSPACE_sRGB
    )
except ImportError:
    print("错误：未找到 'colour-science' 库或其核心组件。")
    print("请确保已正确安装： pip install colour-science")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
M1_NP = M1_q1

# --- PyTorch Matrices and Constants (from Q1.py or adapted) ---
M1_TS = torch.tensor(M1_NP, dtype=torch.float32, device=DEVICE)
SRGB_TO_XYZ_MAT_TS = torch.tensor([  # sRGB -> XYZ for PyTorch (from Q1.py)
    [0.41239080, 0.35758434, 0.18048079],
    [0.21263901, 0.71516868, 0.07219232],
    [0.01933082, 0.11919478, 0.95053215]
], dtype=torch.float32, device=DEVICE)

# D65 Whitepoint reference for XYZ to LAB conversion
REF_X_LAB = 0.95047
REF_Y_LAB = 1.00000
REF_Z_LAB = 1.08883

# Constants for XYZ2LAB
EPSILON_LAB = 216.0 / 24389.0
KAPPA_LAB = 24389.0 / 27.0


# --- Manually defined oetf_sRGB function (as per previous request) ---
def oetf_sRGB_manual(linear_rgb: np.ndarray) -> np.ndarray:
    L = np.asarray(linear_rgb)
    condition = L <= 0.0031308
    part1 = 12.92 * L
    part2 = 1.055 * np.power(L, 1.0 / 2.4) - 0.055
    non_linear_rgb = np.where(condition, part1, part2)
    return non_linear_rgb


# --- PyTorch Color Transformation Functions (adapted from Q1.py) ---
def bt2xyz_ts(bt2020_rgb_ts: torch.Tensor) -> torch.Tensor:
    return torch.matmul(bt2020_rgb_ts, M1_TS.T)


def srgb2xyz_ts(srgb_ts: torch.Tensor) -> torch.Tensor:
    return torch.matmul(srgb_ts, SRGB_TO_XYZ_MAT_TS.T)


def xyz2lab_ts(xyz_ts: torch.Tensor) -> torch.Tensor:
    xr = xyz_ts[:, 0] / REF_X_LAB
    yr = xyz_ts[:, 1] / REF_Y_LAB
    zr = xyz_ts[:, 2] / REF_Z_LAB

    fx = torch.where(xr > EPSILON_LAB, xr.pow(1.0 / 3.0), (KAPPA_LAB * xr + 16.0) / 116.0)
    fy = torch.where(yr > EPSILON_LAB, yr.pow(1.0 / 3.0), (KAPPA_LAB * yr + 16.0) / 116.0)
    fz = torch.where(zr > EPSILON_LAB, zr.pow(1.0 / 3.0), (KAPPA_LAB * zr + 16.0) / 116.0)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def cie2000_ts(lab1_ts: torch.Tensor, lab2_ts: torch.Tensor, kL: float = 1.0, kC: float = 1.0,
               kH: float = 1.0) -> torch.Tensor:
    L1, a1, b1 = lab1_ts[:, 0], lab1_ts[:, 1], lab1_ts[:, 2]
    L2, a2, b2 = lab2_ts[:, 0], lab2_ts[:, 1], lab2_ts[:, 2]

    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    C_mean = (C1 + C2) / 2.0

    # Use math.pi if torch.pi is not available (older PyTorch versions)
    try:
        pi_val = torch.pi
    except AttributeError:
        pi_val = math.pi

    pow_Cmean_7 = torch.pow(C_mean, 7)
    pow_25_7 = torch.pow(torch.tensor(25.0, device=C_mean.device, dtype=C_mean.dtype), 7)  # Ensure dtype matches
    G = 0.5 * (1 - torch.sqrt(pow_Cmean_7 / (pow_Cmean_7 + pow_25_7)))

    a1_pi_val = (1 + G) * a1
    a2_pi_val = (1 + G) * a2

    C1_pi = torch.sqrt(a1_pi_val * a1_pi_val + b1 * b1)
    C2_pi = torch.sqrt(a2_pi_val * a2_pi_val + b2 * b2)
    C_mean_pi = (C1_pi + C2_pi) / 2.0

    # Ensure h1_pi and h2_pi are in [0, 2*pi]
    h1_pi = torch.atan2(b1, a1_pi_val)
    h1_pi = torch.where(h1_pi >= 0, h1_pi, h1_pi + 2 * pi_val)
    h2_pi = torch.atan2(b2, a2_pi_val)
    h2_pi = torch.where(h2_pi >= 0, h2_pi, h2_pi + 2 * pi_val)

    # Calculate h_mean_pi
    abs_h_diff = torch.abs(h1_pi - h2_pi)
    sum_h = h1_pi + h2_pi

    h_mean_pi = torch.empty_like(abs_h_diff)

    # Condition 1: C1_pi * C2_pi == 0
    cond1 = (C1_pi * C2_pi) == 0
    h_mean_pi[cond1] = sum_h[
        cond1]  # Or sum_h[cond1] / 2, depending on desired behavior for achromatic. Original Q1 used sum_h.
    # Let's stick to original Q1 logic: h1_pi + h2_pi if C1C2=0
    # However, if sum_h is used, it might go out of range if not divided by 2
    # For safety if one C is 0, other h is used. If both are 0, sum is 0.
    # Sharma's paper implies if C1 or C2 is 0, then h_mean_pi = h1_pi + h2_pi.
    # Let's use (h1_pi + h2_pi) for this case for now.

    # Condition 2: abs_h_diff > pi
    cond2 = (~cond1) & (abs_h_diff > pi_val)
    h_mean_pi[cond2] = (sum_h[cond2] + 2 * pi_val) / 2.0

    # Condition 3: Default
    cond3 = (~cond1) & (~cond2)
    h_mean_pi[cond3] = sum_h[cond3] / 2.0

    T = (1
         - 0.17 * torch.cos(h_mean_pi - pi_val / 6.0)
         + 0.24 * torch.cos(2 * h_mean_pi)
         + 0.32 * torch.cos(3 * h_mean_pi + pi_val / 30.0)  # Q1.py was pi/30, not 6*pi/180
         - 0.20 * torch.cos(4 * h_mean_pi - 63.0 * pi_val / 180.0)
         )

    # Calculate delta_h_pi
    delta_h_pi = torch.empty_like(abs_h_diff)
    # Cond 1: C1_pi * C2_pi == 0
    delta_h_pi[cond1] = 0.0

    # Cond 2.1: abs_h_diff <= pi
    h_diff = h2_pi - h1_pi  # Use this instead of abs for sign
    cond21 = (~cond1) & (torch.abs(h_diff) <= pi_val)  # Original Q1 logic (h2pi_h1pi.abs() <= torch.pi)
    delta_h_pi[cond21] = h_diff[cond21]

    # Cond 2.2: h_diff > pi (original logic was based on h2pi_h1pi)
    cond22 = (~cond1) & (~cond21) & (
                h_diff > pi_val)  # Original Q1 was h2pi_h1pi - 2 * torch.pi * torch.sign(h2pi_h1pi)
    # which simplifies to h_diff - 2*pi if h_diff > pi
    delta_h_pi[cond22] = h_diff[cond22] - 2 * pi_val

    # Cond 2.3: h_diff < -pi
    cond23 = (~cond1) & (~cond21) & (h_diff < -pi_val)  # Original Q1 simplifies to h_diff + 2*pi if h_diff < -pi
    delta_h_pi[cond23] = h_diff[cond23] + 2 * pi_val

    dLp = L2 - L1
    dCp = C2_pi - C1_pi
    dHp = 2 * torch.sqrt(C1_pi * C2_pi) * torch.sin(delta_h_pi / 2.0)

    L_mean = (L1 + L2) / 2.0
    S_L = 1 + (0.015 * (L_mean - 50).pow(2)) / torch.sqrt(20 + (L_mean - 50).pow(2))
    S_C = 1 + 0.045 * C_mean_pi
    S_H = 1 + 0.015 * C_mean_pi * T

    delta_theta = (30.0 * pi_val / 180.0) * torch.exp(-((h_mean_pi * 180.0 / pi_val - 275.0) / 25.0).pow(2))

    R_C = 2 * torch.sqrt(pow_Cmean_7 / (pow_Cmean_7 + pow_25_7))
    R_T = -R_C * torch.sin(2 * delta_theta)

    delta_E = torch.sqrt(
        (dLp / (kL * S_L)).pow(2) +
        (dCp / (kC * S_C)).pow(2) +
        (dHp / (kH * S_H)).pow(2) +
        R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
    )
    return delta_E


# --- MLP Model Loading ---
def load_mlp_model(ckpt_path: str, device: torch.device) -> MLP:
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


# --- Heatmap Saving Function ---
def save_delta_e_heatmap(delta_e_values: np.ndarray, image_shape: tuple, filepath: str,
                         max_delta_e_scale: float = 10.0):
    """
    Generates and saves a heatmap image from Delta E values.
    """
    delta_e_image = delta_e_values.reshape(image_shape[0], image_shape[1])

    # Normalize Delta E values for colormap
    # Values are clipped to [0, max_delta_e_scale] then normalized to [0, 1]
    scaled_de = np.clip(delta_e_image / max_delta_e_scale, 0, 1)

    try:
        colormap = plt.get_cmap('viridis')  # Or 'plasma', 'magma', 'inferno'
        heatmap_rgb_float = colormap(scaled_de)[:, :, :3]  # Get RGB, drop alpha if present
    except Exception as e:
        print(f"错误：应用颜色映射时发生错误：{e}")
        print("请确保 Matplotlib 已正确安装。")
        return

    heatmap_uint8 = (heatmap_rgb_float * 255).astype(np.uint8)

    try:
        heatmap_img = Image.fromarray(heatmap_uint8)
        heatmap_img.save(filepath)
        print(f"Delta E 热图已保存至: {filepath} (最大 Delta E {max_delta_e_scale} 对应色彩范围顶端)")
    except Exception as e:
        print(f"错误：保存 Delta E 热图时发生错误： {e}")


# --- Image Processing Function ---
# Added heatmap_output_path argument
def process_bt2020_to_srgb_image(image_path: str, model_path: str, output_path: str, heatmap_output_path: str):
    print(f"正在处理图像: {image_path}")
    print(f"使用 MLP 模型: {model_path}")
    print(f"输出将保存至: {output_path}")
    print(f"Delta E 热图将保存至: {heatmap_output_path}")
    print(f"使用设备: {DEVICE}")

    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误：输入图像文件未在 {image_path} 找到。")
        sys.exit(1)
    except Exception as e:
        print(f"错误：打开图像时发生错误： {e}")
        sys.exit(1)

    img_np = np.array(img)
    original_shape = img_np.shape
    if original_shape[2] != 3:
        print("错误：图像不包含3个颜色通道。")
        sys.exit(1)
    num_pixels = original_shape[0] * original_shape[1]

    img_np_float = img_np.astype(np.float32) / 255.0
    linear_bt2020_rgb_np = np.clip(img_np_float, 0.0, 1.0)  # Clip before OETF inverse
    linear_bt2020_rgb_np = oetf_inverse_BT2020(linear_bt2020_rgb_np)
    linear_bt2020_flat_np = linear_bt2020_rgb_np.reshape(num_pixels, 3)

    xyz_pixels_np = np.dot(linear_bt2020_flat_np, M1_NP.T)
    linear_srgb_candidate_np = XYZ_to_RGB(
        xyz_pixels_np,
        colourspace=RGB_COLOURSPACE_sRGB
    )

    oog_mask_np = np.any((linear_srgb_candidate_np < -1e-7) | (linear_srgb_candidate_np > 1.0 + 1e-7), axis=1)
    ingamut_mask_np = ~oog_mask_np
    num_oog_pixels = np.sum(oog_mask_np)
    print(f"总像素数: {num_pixels}, sRGB 色域外像素数: {num_oog_pixels}")

    final_linear_srgb_flat_np = np.zeros_like(linear_srgb_candidate_np)
    final_linear_srgb_flat_np[ingamut_mask_np] = np.clip(linear_srgb_candidate_np[ingamut_mask_np], 0.0, 1.0)

    if num_oog_pixels > 0:
        print(f"使用 MLP 映射 {num_oog_pixels} 个 OOG 像素...")
        oog_bt2020_linear_np = linear_bt2020_flat_np[oog_mask_np]
        oog_bt2020_linear_ts = torch.from_numpy(oog_bt2020_linear_np).float().to(DEVICE)
        model = load_mlp_model(model_path, DEVICE)
        with torch.no_grad():
            mapped_srgb_linear_ts = model(oog_bt2020_linear_ts)
        final_linear_srgb_flat_np[oog_mask_np] = np.clip(mapped_srgb_linear_ts.cpu().numpy(), 0.0, 1.0)
    else:
        print("未找到 OOG 像素。无需 MLP 映射。")

    # --- Delta E Calculation ---
    print("正在计算 Delta E 2000 值...")
    linear_bt2020_ts = torch.from_numpy(linear_bt2020_flat_np).float().to(DEVICE)
    final_linear_srgb_ts = torch.from_numpy(final_linear_srgb_flat_np).float().to(DEVICE)

    xyz_bt2020_ts = bt2xyz_ts(linear_bt2020_ts)
    lab_bt2020_ts = xyz2lab_ts(xyz_bt2020_ts)

    xyz_srgb_ts = srgb2xyz_ts(final_linear_srgb_ts)
    lab_srgb_ts = xyz2lab_ts(xyz_srgb_ts)

    delta_e_flat_ts = cie2000_ts(lab_srgb_ts, lab_bt2020_ts)
    delta_e_np_flat = delta_e_flat_ts.cpu().numpy()

    # Save heatmap
    save_delta_e_heatmap(delta_e_np_flat, original_shape, heatmap_output_path)
    mean_delta_e = np.mean(delta_e_np_flat)
    max_delta_e = np.max(delta_e_np_flat)
    median_delta_e = np.median(delta_e_np_flat)
    percentile_95_delta_e = np.percentile(delta_e_np_flat, 95)
    print(
        f"Delta E 2000 - 平均值: {mean_delta_e:.4f}, 中位数: {median_delta_e:.4f}, 最大值: {max_delta_e:.4f}, 95百分位: {percentile_95_delta_e:.4f}")

    # --- Final Image Saving ---
    final_linear_srgb_image = final_linear_srgb_flat_np.reshape(original_shape)
    final_nonlinear_srgb_image = oetf_sRGB_manual(np.clip(final_linear_srgb_image, 0.0, 1.0))
    final_image_np_uint8 = (np.clip(final_nonlinear_srgb_image, 0.0, 1.0) * 255.0).astype(np.uint8)

    try:
        output_img = Image.fromarray(final_image_np_uint8)
        output_img.save(output_path)
        print(f"成功处理图像并保存至: {output_path}")
    except Exception as e:
        print(f"错误：保存输出图像时发生错误： {e}")
        sys.exit(1)


# --- 主执行块 ---
if __name__ == "__main__":
    if len(sys.argv) != 4:  # We'll derive heatmap path from output path
        print("\n用法: python process_bt2020_image.py <输入bt2020图像路径> <模型.pth路径> <输出srgb图像路径>\n")
        print("Delta E 热图将以 '_delta_e.png' 后缀自动保存在输出目录。\n")
        print("示例:")
        print("  python process_bt2020_image.py input.tif model/my_model_final.pth output.png\n")
        sys.exit(1)

    input_image_path_arg = sys.argv[1]
    model_ckpt_path_arg = sys.argv[2]
    output_image_path_arg = sys.argv[3]

    # Derive heatmap output path
    output_dir = os.path.dirname(output_image_path_arg)
    output_basename = os.path.basename(output_image_path_arg)
    output_name_no_ext, output_ext = os.path.splitext(output_basename)

    # Ensure output_dir is created if it's just a filename for current dir
    if not output_dir:
        output_dir = "."  # current directory
    else:
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    heatmap_path = os.path.join(output_dir, f"{output_name_no_ext}_delta_e{output_ext if output_ext else '.png'}")

    process_bt2020_to_srgb_image(
        input_image_path_arg,
        model_ckpt_path_arg,
        output_image_path_arg,
        heatmap_path  # Pass the derived heatmap path
    )