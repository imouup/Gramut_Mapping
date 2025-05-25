import numpy as np
import colour
import matplotlib.pyplot as plt
import matplotlib
import torch
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Qt5Agg')

def wavelength_2_xyz(wavelength: np.ndarray):
    XYZ = colour.wavelength_to_XYZ(wavelength)
    return XYZ

def plot_xyz_color_vectors(vectors, wavelength):
    """
    在三维空间中绘制 XYZ 色彩空间的颜色向量，并标记波长。

    参数:
        vectors (np.ndarray): shape 为 (n, 3) 的 XYZ 向量数组。
        wavelength (np.ndarray): 对应每个向量的波长（单位：nm）。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.zeros((vectors.shape[0], 3))
    X, Y, Z = origin[:, 0], origin[:, 1], origin[:, 2]
    U, V, W = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    # 将 XYZ 转换为 RGB，用于颜色显示
    rgb_colors = np.clip(colour.XYZ_to_sRGB(vectors), 0, 1)

    for i in range(vectors.shape[0]):
        ax.quiver(
            X[i], Y[i], Z[i], U[i], V[i], W[i],
            color=tuple(rgb_colors[i]),
            arrow_length_ratio=0.05, linewidth=1.5
        )

        # 添加文字标记
        ax.text(
            U[i], V[i], W[i],
            f'{int(wavelength[i])}nm',
            fontsize=9, color='black', ha='center'
        )

        # 如果需要标记端点球可以取消下面这行注释
        # ax.scatter(U[i], V[i], W[i], color=tuple(rgb_colors[i]), s=30)

    # 设置合理坐标轴范围
    max_range = np.max(vectors) * 1.1
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('XYZ Color Space Vectors (with Wavelength Labels)')

    plt.show()

def plot_gamut_on_chromaticity_diagram(vectors, wavelength):
    """
    绘制 xy 色度图（马蹄图），并在其上标出给定颜色围成的区域。
    """
    # 将 XYZ 向量转换为 xy 坐标
    xy = colour.XYZ_to_xy(vectors)

    # 获取色度图背景
    figure, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    # 绘制多边形区域
    polygon = plt.Polygon(xy, closed=True, fill=True, edgecolor='black', facecolor='none', linewidth=1.5)
    axes.add_patch(polygon)

    # 标记每个波长点
    rgb_colors = np.clip(colour.XYZ_to_sRGB(vectors), 0, 1)
    for i in range(len(xy)):
        axes.plot(xy[i, 0], xy[i, 1], 'o', color=rgb_colors[i], markersize=8)
        axes.text(xy[i, 0], xy[i, 1] + 0.01, f'{int(wavelength[i])}nm', fontsize=9, ha='center')

    plt.title("CIE 1931 Chromaticity Diagram with Custom Gamut")
    plt.grid(True)
    plt.show()


def GetPoints2020_lab(num_samples, device=None):
    """
    在 LAB 空间下的 BT.2020 色域范围内均匀采样 num_samples 个点，
    并返回这些点在 BT.2020 线性 RGB 空间下的坐标（NumPy 数组，形状 [num_samples, 3]）。

    参数：
        num_samples (int): 需要采样的点数量。
        device (str 或 torch.device, 可选): 指定运算设备，例如 "cuda" 或 "cpu"。如果为 None，会自动选择
            如果 CUDA 可用，则用 "cuda"，否则用 "cpu"。

    返回：
        np.ndarray，形状为 [num_samples, 3]，dtype 为 float32，对应于每个采样点在 BT.2020 线性 RGB 空间下的 (R, G, B)。
    """
    # ---------------------------------
    # 1. 设备（device）设置
    # ---------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ---------------------------------
    # 2. PyTorch 版的 LAB -> XYZ -> BT.2020 RGB 转换函数
    # ---------------------------------

    # D65 白点 (XYZ)
    Xn = 0.95047
    Yn = 1.0
    Zn = 1.08883

    # BT.2020 的 XYZ -> RGB 矩阵（线性转换，参考 SMPTE RP 177-1993 / Rec.2020 标准）
    M_xyz2rgb = torch.tensor([
        [ 1.7166511879712674, -0.3556707837763923, -0.25336628137365974],
        [-0.6666843518324891,  1.6164812366349395,  0.01576854581391113],
        [ 0.017639857445310783, -0.042770613257808524,  0.9421031212354738]
    ], dtype=torch.float32, device=device)  # shape: [3,3]

    # 下面给出一个从 LAB 到 XYZ，再到 BT.2020 线性 RGB 的完整向量化计算函数
    def lab_to_bt2020_rgb(L, a, b):
        """
        输入：
          L, a, b: 同形状的一组张量，表示在 CIELAB 空间中的 L*, a*, b* 分量。
        输出：
          rgb_lin: 形状为 [..., 3] 的张量，表示对应 BT.2020 线性 RGB 值 (R, G, B)。
        """
        # ---- 2.1. LAB --> XYZ ----
        # 按照 CIE 1976 定义：
        #   f_y = (L* + 16) / 116
        #   f_x = a* / 500 + f_y
        #   f_z = f_y - b* / 200
        f_y = (L + 16.0) / 116.0
        f_x = a / 500.0 + f_y
        f_z = f_y - b / 200.0

        # 对应的 XYZ 相对量 xr, yr, zr：
        #   若 f_i^3 > 0.008856，则对应 i = f_i^3；否则 i = (116 * f_i - 16) / 903.3
        eps = 0.008856  # CIE 标准阈值 = (6/29)^3 ≈ 0.008856
        kappa = 903.3

        # 先计算 fx^3, fy^3, fz^3
        f_x3 = f_x ** 3
        f_y3 = f_y ** 3
        f_z3 = f_z ** 3

        # xr
        xr = torch.where(f_x3 > eps, f_x3, (116.0 * f_x - 16.0) / kappa)
        # yr
        # 注意：CIE 76 定义里，yr 也要区分 L* 阈值，即当 L* > kappa * eps 的时候，yr = f_y^3，否则 yr = L* / kappa
        # kappa * eps = 903.3 * 0.008856 ≈ 7.9996
        yr = torch.where(L > (kappa * eps), f_y3, L / kappa)
        # zr
        zr = torch.where(f_z3 > eps, f_z3, (116.0 * f_z - 16.0) / kappa)

        # 还原到绝对 XYZ (参照白点 D65)
        X = xr * Xn
        Y = yr * Yn    # Yn == 1.0
        Z = zr * Zn

        # 合并到 shape [...,3]
        xyz = torch.stack([X, Y, Z], dim=-1)  # 最后一个维度为 3

        # ---- 2.2. XYZ --> BT.2020 线性 RGB ----
        # xyz 形状 [...,3]，把最后一维与 M_xyz2rgb 相乘
        # 为了用矩阵乘法，需要把 xyz 扁平到形状 (-1, 3)，然后再重塑回来
        orig_shape = xyz.shape
        flat_xyz = xyz.reshape(-1, 3).t()  # shape: [3, N]
        flat_rgb = torch.matmul(M_xyz2rgb, flat_xyz)  # shape: [3, N]
        rgb = flat_rgb.t().reshape(*orig_shape)       # shape: [..., 3]

        return rgb  # 线性 RGB 值，可能超出 [0,1] 需要后续裁剪/筛选

    # ---------------------------------
    # 3. 拒绝采样主循环：不断在 LAB 边界内随机生成点，筛选出属于 BT.2020 色域（RGB 三分量都在 [0,1]）的
    # ---------------------------------
    # CIELAB 的大致边界：
    #   L*: [0, 100]
    #   a*: [-128, +127]
    #   b*: [-128, +127]
    #
    # 采样思路：
    #   1) 每次尝试采样 batch_size 个 LAB 随机点（均匀分布在上述立方体内部）。
    #   2) 把这批点转换到 BT.2020 线性 RGB，筛选出 0<=R<=1, 0<=G<=1, 0<=B<=1 的有效点。
    #   3) 把有效点累加到结果队列，当数量 >= num_samples 时退出循环；否则继续采样，直到凑够。
    #
    samples_collected = 0
    rgb_list = []

    # 一次批量取样的数量；可以根据经验调大或调小。
    # 如果你发现采样效率（在 LAB 立方体里命中的比例）很低，可以适当加大批次，减少循环次数。
    batch_size = max(num_samples * 5, 16384)

    while samples_collected < num_samples:
        # ---------------------------------
        # 3.1. 在 LAB 立方体内随机生成 batch_size 个点
        # ---------------------------------
        # L* 均匀分布在 [0, 100]
        L_rand = torch.rand(batch_size, device=device) * 100.0
        # a*, b* 均匀分布在 [-128, 127]
        a_rand = (torch.rand(batch_size, device=device) * 255.0) - 128.0
        b_rand = (torch.rand(batch_size, device=device) * 255.0) - 128.0

        # ---------------------------------
        # 3.2. 从 LAB 转到 BT.2020 线性 RGB
        # ---------------------------------
        rgb_lin = lab_to_bt2020_rgb(L_rand, a_rand, b_rand)  # shape: [batch_size, 3]

        # ---------------------------------
        # 3.3. 筛选出 R,G,B 都在 [0,1] 范围内的点
        # ---------------------------------
        mask = (rgb_lin >= 0.0) & (rgb_lin <= 1.0)  # shape [batch_size, 3] bool
        valid_mask = mask.all(dim=-1)               # shape [batch_size], bool
        valid_rgb = rgb_lin[valid_mask]             # shape [M, 3]，其中 M 是这批次命中的数量

        # ---------------------------------
        # 3.4. 把筛到的有效点加入结果队列
        # ---------------------------------
        if valid_rgb.numel() > 0:
            # 如果当前累积数量还没满，就尽量取出前面的部分
            remain = num_samples - samples_collected
            if valid_rgb.shape[0] > remain:
                valid_rgb = valid_rgb[:remain]
            rgb_list.append(valid_rgb)
            samples_collected += valid_rgb.shape[0]

        # 如果不够，循环会自动继续；直到 samples_collected >= num_samples
    # end while

    # ---------------------------------
    # 4. 拼接最终结果，并转为 NumPy
    # ---------------------------------
    rgb_all = torch.cat(rgb_list, dim=0)  # shape [num_samples, 3]
    rgb_all = rgb_all.clamp(0.0, 1.0)     # 保证数值在 [0,1]（虽然筛过了，但为保险做 clamp）

    # 最后把 Tensor 拷贝回 CPU 并转为 NumPy：
    return rgb_all.cpu().numpy()
