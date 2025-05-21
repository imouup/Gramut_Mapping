import numpy as np
import colour
import matplotlib.pyplot as plt
import matplotlib
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
