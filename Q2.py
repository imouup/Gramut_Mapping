import numpy as np
import colour
import matplotlib.pyplot as plt
import matplotlib
from toolkits import wavelength_2_xyz, plot_xyz_color_vectors, plot_gamut_on_chromaticity_diagram
matplotlib.use('Qt5Agg')

'''
这里是两个示例的四基色和五基色空间，为了方便直接用了波长来创建
实际使用时请使用XYZ坐标
'''
four_bases = np.array([430, 480, 550, 625])
five_bases = np.array([440, 490, 530, 580,610])
XYZ4 = wavelength_2_xyz(four_bases)
XYZ5 = wavelength_2_xyz(five_bases)
# plot_xyz_color_vectors(XYZ5, five_bases)
# plot_gamut_on_chromaticity_diagram(XYZ5,five_bases)

def xyY2XYZ(xyY):
    '''
    该函数讲xyY坐标转为XYZ坐标

    :param xyY: xyY坐标，(n,3) numpy数组
    :return: XYZ坐标，(n,3) numpy数组
    '''
    XYZ = colour.xyY_to_XYZ(xyY)
    return XYZ

def four2five(cordinate, XYZ4, XYZ5):
    '''
    利用伪逆，将四基色下的坐标转为五基色下的坐标

    :param cordinate: (n,4) 的矩阵，每一行为一个四基色下的坐标
    :param XYZ4: 四基色转换矩阵 (3,4)
    :param XYZ5: 五基色转换矩阵 (3,5)
    :return: P_flags 标记后的五基色坐标
    '''
    # 先将输入的四基色颜色在XYZ下表示
    cordinate = cordinate.T
    mat4 = XYZ4.T
    xyz = mat4 @ cordinate # cordiante 是(4,)的列向量

    # 求五基色矩阵的伪逆
    mat5 = XYZ5.T
    mat5_inv = np.linalg.pinv(mat5)
    P_5 = mat5_inv @ xyz
    return P_5.T   # (n,5) 最后一列为flag: 0表示未越界，1表示越界

def flag(points4, points5):
    '''
    标记函数

    :param points4: (n,4) 四基色下的点
    :param points5: (n,5) 求伪逆后五基色下的点
    :return: 第一个返回的标记后的points4，第二个是标记后的points5
    '''
    flag = np.any((points5 < 0) | (points5 > 1), axis=1)
    flag = flag[:,np.newaxis]  # 把一维数组转为二维列向量
    P_flags5 = np.hstack((points5,flag))
    P_flags4 = np.hstack((points4,flag))
    return P_flags4, P_flags5


def GetPoints4(n_samples=1000, seed=233):
    '''
    在四基色下均匀取点

    :return: 一个n_samples行三列的矩阵,返回的点在BT2020定义的坐标系下
    '''
    np.random.seed(seed)
    points = np.random.rand(n_samples, 4)
    return points


def filter(points):
    '''
    分离flag = 0与flag =1的点

    :param points:  four2five数返回的点
    :return: 第一个返回是flag = 0的点，第二个是flag = 1的点
    '''
    flags = points[:,-1]
    flags = flags[:,np.newaxis]
    # 创建掩码 (axis=1表示是对行进行判断，每行对应一个向量)
    mask0 = np.any((flags == 0), axis=1)  # flag = 0
    mask1 = np.any((flags == 1), axis=1)  # flag = 1
    points = points[:,0:-1]
    # 布尔索引，得到超出目标色域边界的点
    return points[mask0], points[mask1]


# 取点
points4 = GetPoints4()
# 使用求伪逆的方法
points5 = four2five(points4,XYZ4,XYZ5)
_, pt_flag = flag(points4,points5)
a,b = filter(pt_flag)
print(a,b)

