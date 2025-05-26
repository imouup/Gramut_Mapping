import numpy as np
from PIL import Image
import colour  # 依赖 colour-science 库的核心转换功能
import os


def load_image_to_float(image_path):
    """
    加载图像并将其像素值转换为[0, 1]范围内的浮点NumPy数组。
    手动实现整数到浮点数的转换。
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)

        if img_array.dtype == np.uint8:
            img_float = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.uint16:
            img_float = img_array.astype(np.float32) / 65535.0
        elif np.issubdtype(img_array.dtype, np.floating):
            # 如果已经是浮点数，我们假设它可能不在0-1范围，先进行裁切
            # 如果您确定输入的浮点图已经是0-1，可以移除或调整这里的裁切
            print(f"信息: 输入图像为浮点类型 ({img_array.dtype})。将裁切到 [0,1] 范围（如果需要）。")
            img_float = np.clip(img_array.astype(np.float32), 0.0, 1.0)
        else:
            raise ValueError(f"图像数据类型不支持: {img_array.dtype}。请提供 uint8, uint16 或浮点图像。")

        return img_float

    except Exception as e:
        print(f"加载或转换图像 '{image_path}' 时发生错误: {e}")
        return None


def save_float_image(image_array_float, output_path):
    """
    将[0, 1]范围内的浮点NumPy数组转换回uint8并保存为图像文件。
    手动实现浮点数到整数的转换。
    """
    if not isinstance(image_array_float, np.ndarray) or image_array_float.dtype != np.float32:
        # 确保输入是 float32 numpy 数组
        if isinstance(image_array_float, np.ndarray):
            image_array_float = image_array_float.astype(np.float32)
        else:
            print(f"错误: save_float_image 需要 NumPy 数组作为输入, 得到 {type(image_array_float)}")
            return

    # 1. 确保输入在 [0, 1] 范围内
    img_clipped_to_domain = np.clip(image_array_float, 0.0, 1.0)
    # 2. 缩放到 [0, 255]
    img_scaled = img_clipped_to_domain * 255.0
    # 3. 四舍五入到最接近的整数
    img_rounded = np.round(img_scaled)
    # 4. 再次裁切以确保值在 uint8 的有效范围内 [0, 255] 并转换类型
    img_int_uint8 = np.clip(img_rounded, 0, 255).astype(np.uint8)

    try:
        img = Image.fromarray(img_int_uint8)
        img.save(output_path)
        print(f"图像已保存到: {output_path}")
    except Exception as e:
        print(f"保存图像 '{output_path}' 时发生错误: {e}")


def bt2020_to_srgb_mappings(input_image_path, output_folder="output_images"):
    """
    将BT.2020图像转换为sRGB，使用绝对色度和简化的感知（相对色度）映射。
    计算每种映射结果与原始图像之间的平均Delta E00。
    不使用 'colour.RGB_gamut_fit' 和 'colour.utilities.to/from_domain_1'。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_bt2020_norm = load_image_to_float(input_image_path)
    if img_bt2020_norm is None:
        print("错误：无法加载或处理输入图像，脚本终止。")
        return None, None

    # 仍然需要使用 colour-science 的核心色彩空间定义和转换功能
    try:
        CS_BT2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
        CS_SRGB = colour.RGB_COLOURSPACES['sRGB']
    except Exception as e:
        print(f"错误: 无法从 'colour.RGB_COLOURSPACES' 加载色彩空间定义: {e}")
        print("     请确保您的 'colour-science' 库安装正确且版本兼容。")
        return None, None

    # --- 绝对色度映射 (Absolute Colorimetric) ---
    # BT.2020 (非线性 0-1) -> BT.2020 (线性)
    rgb_bt2020_linear = CS_BT2020.cctf_decoding(img_bt2020_norm)
    # BT.2020 (线性) -> XYZ (D65)
    xyz_representation_original = colour.RGB_to_XYZ(rgb_bt2020_linear, CS_BT2020)
    # XYZ (D65) -> sRGB (线性)
    rgb_srgb_linear_abs = colour.XYZ_to_RGB(xyz_representation_original, CS_SRGB)
    # sRGB (线性) -> sRGB (非线性 0-1), 同时裁切到 [0, 1] 色域
    img_srgb_absolute_nonlinear_raw = CS_SRGB.cctf_encoding(rgb_srgb_linear_abs)
    img_srgb_absolute = np.clip(img_srgb_absolute_nonlinear_raw, 0.0, 1.0)
    save_float_image(img_srgb_absolute, os.path.join(output_folder, "output_srgb_absolute.png"))

    # --- 感知映射 (简化为相对色度映射) ---
    print("\n信息: 由于用户要求不使用 'RGB_gamut_fit'，感知映射将简化为相对色度映射。")
    print("     这意味着超出sRGB色域的颜色将被直接裁切，而不是进行复杂的色域压缩。")
    print("     在此 BT.2020 (D65) -> sRGB (D65) 的情况下，感知映射的结果预计将与绝对色度映射的结果相同。")

    # XYZ (D65) -> sRGB (线性) - 与绝对色度路径中的相同步骤
    rgb_srgb_linear_unmapped_perc = colour.XYZ_to_RGB(xyz_representation_original, CS_SRGB)
    # 关键步骤：不使用RGB_gamut_fit，直接进行裁切 (clip)
    rgb_srgb_linear_perceptual_mapped_simplified = np.clip(rgb_srgb_linear_unmapped_perc, 0.0, 1.0)

    # sRGB (线性, 已裁切) -> sRGB (非线性 0-1)
    img_srgb_perc_nonlinear_raw = CS_SRGB.cctf_encoding(rgb_srgb_linear_perceptual_mapped_simplified)
    # 最终再次确保裁切 (通常上一步已完成)
    img_srgb_perceptual_simplified = np.clip(img_srgb_perc_nonlinear_raw, 0.0, 1.0)
    save_float_image(img_srgb_perceptual_simplified,
                     os.path.join(output_folder, "output_srgb_perceptual_simplified.png"))

    # --- Delta E00 计算 ---
    # 将原始BT.2020图像数据 (XYZ D65) 转换为CIELAB D65
    lab_ref = colour.XYZ_to_Lab(xyz_representation_original, CS_BT2020.whitepoint)

    # 1. 绝对映射sRGB结果转换为CIELAB D65
    rgb_srgb_absolute_linear_for_lab = CS_SRGB.cctf_decoding(img_srgb_absolute)
    xyz_from_srgb_abs = colour.RGB_to_XYZ(rgb_srgb_absolute_linear_for_lab, CS_SRGB)
    lab_abs = colour.XYZ_to_Lab(xyz_from_srgb_abs, CS_SRGB.whitepoint)

    # 2. 简化感知映射sRGB结果转换为CIELAB D65
    rgb_srgb_perceptual_simplified_linear_for_lab = CS_SRGB.cctf_decoding(img_srgb_perceptual_simplified)
    xyz_from_srgb_per_simplified = colour.RGB_to_XYZ(rgb_srgb_perceptual_simplified_linear_for_lab, CS_SRGB)
    lab_per_simplified = colour.XYZ_to_Lab(xyz_from_srgb_per_simplified, CS_SRGB.whitepoint)

    # 计算Delta E00
    delta_e_abs = colour.delta_E(lab_ref, lab_abs, method='CIE 2000')
    delta_e_per_simplified = colour.delta_E(lab_ref, lab_per_simplified, method='CIE 2000')

    avg_delta_e_abs = np.mean(delta_e_abs)
    avg_delta_e_per_simplified = np.mean(delta_e_per_simplified)

    print(f"\n--- 色差计算 (与原始BT.2020图像比较) ---")
    print(f"平均 Delta E00 (绝对色度映射): {avg_delta_e_abs:.4f}")
    print(f"平均 Delta E00 (感知映射 - 已简化为相对色度/裁切): {avg_delta_e_per_simplified:.4f}")
    if np.isclose(avg_delta_e_abs, avg_delta_e_per_simplified):
        print("注意: 由于感知映射被简化，其Delta E00与绝对色度映射的Delta E00可能非常接近或相同。")

    return avg_delta_e_abs, avg_delta_e_per_simplified


if __name__ == "__main__":
    input_file = './test/R2020-sRGB-color-ring.png'  # <--- 请修改这里

    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在。请提供一个有效的BT.2020图像文件路径。")
        print("正在创建一个虚拟的BT.2020图像用于演示...")

        dummy_height, dummy_width = 100, 100
        dummy_bt2020_array_uint8 = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
        dummy_bt2020_array_uint8[0:20, :, 0] = 204
        dummy_bt2020_array_uint8[20:40, :, 1] = 230
        dummy_bt2020_array_uint8[40:60, :, 2] = 178
        dummy_bt2020_array_uint8[60:80, :, :] = 128
        dummy_bt2020_array_uint8[80:100, :, 0] = 51
        dummy_bt2020_array_uint8[80:100, :, 1] = 153
        dummy_bt2020_array_uint8[80:100, :, 2] = 102

        img_temp = Image.fromarray(dummy_bt2020_array_uint8)
        # 保存为PNG，确保load_image_to_float可以处理其uint8数据类型
        img_temp.save("dummy_bt2020_input.png")
        input_file = "dummy_bt2020_input.png"
        print(f"已创建虚拟输入文件: {input_file}")

    # 运行转换和计算
    results = bt2020_to_srgb_mappings(input_file)
    if results is not None:
        print("\n脚本执行完毕。")
    else:
        print("\n脚本执行过程中出现错误。")