import numpy as np
import colour
import matplotlib.pyplot as plt
import matplotlib
from toolkits import wavelength_2_xyz, plot_xyz_color_vectors, plot_gamut_on_chromaticity_diagram
matplotlib.use('Qt5Agg')
import torch
from datetime import datetime
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

'''
这里是两个示例的四基色和五基色空间，为了方便直接用了波长来创建
实际使用时请使用XYZ坐标
'''

def xyY2XYZ(xyY):
    '''
    该函数讲xyY坐标转为XYZ坐标

    :param xyY: xyY坐标，(n,3) numpy数组
    :return: XYZ坐标，(n,3) numpy数组
    '''
    XYZ = colour.xyY_to_XYZ(xyY)
    return XYZ

def p42XYZ(p4):
    return p4 @ XYZ4

def p52XYZ(p5):
    return p5 @ XYZ5

def four2five(cordinate, XYZ4, XYZ5):
    '''
    利用伪逆，将四基色下的坐标转为五基色下的坐标

    :param cordinate: (n,4) 的矩阵，每一行为一个四基色下的坐标
    :param XYZ4: 四基色转换矩阵 (3,4)
    :param XYZ5: 五基色转换矩阵 (3,5)
    :return: P_5.T   # (n,5)的利用最小二乘法映射后的无五基色下的坐标
    '''
    # 先将输入的四基色颜色在XYZ下表示
    cordinate = cordinate.T
    mat4 = XYZ4.T
    xyz = mat4 @ cordinate # cordiante 是(4,)的列向量

    # 求五基色矩阵的伪逆
    mat5 = XYZ5.T
    mat5_inv = np.linalg.pinv(mat5)
    P_5 = mat5_inv @ xyz
    return P_5.T   # (n,5)

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

    return points[mask0], points[mask1]


def p42XYZ_ts(p4: torch.Tensor) -> torch.Tensor:
    '''
    在pytorch中实现的从五基色坐标到XYZ的转换

    :param p5: 输入的五基色坐标坐标
    :return: XYZ下的坐标
    '''
    return torch.matmul(p4, XYZ4_ts)

def p52XYZ_ts(p5: torch.Tensor) -> torch.Tensor:
    '''
    在pytorch中实现的从五基色坐标到XYZ的转换

    :param p5: 输入的五基色坐标坐标
    :return: XYZ下的坐标
    '''
    return torch.matmul(p5, XYZ5_ts)


def XYZ2LAB(xyz: torch.Tensor) -> torch.Tensor:
      '''
      输入XYZ下坐标得到LAB下坐标

      :param xyz: 输入XYZ下坐标
      :return: LAB下的坐标
      '''
      # 求x_r, y_r, z_r
      xr = xyz[:,0] / ref_X
      yr = xyz[:,1] / ref_Y
      zr = xyz[:,2] / ref_Z

      epsilon = 216.0 / 24389.0
      kappa = 24389.0 / 27.0

      fx = torch.where(
            xr > epsilon, xr.pow(1.0 / 3.0),
            (kappa * xr + 16.0) / 116.0
      )

      fy = torch.where(
            yr > epsilon, yr.pow(1.0 / 3.0),
            (kappa * yr + 16.0) / 116.0
      )

      fz = torch.where(
            zr > epsilon, zr.pow(1.0 / 3.0),
            (kappa * zr + 16.0) / 116.0
      )

      # 计算L*, a*, b*
      L = (116.0 * fy) - 16.0
      a = 500.0 * (fx - fy)
      b = 200.0 * (fy - fz)
      lab = torch.stack([L,a,b],dim=-1) # dim=-1表示沿着最后一个维度拼接

      return lab

# 测试代码
# xyz_white = torch.tensor([[0.95047, 1.00000, 1.08883]], dtype=torch.float32)
# lab_white = XYZ2LAB(xyz_white)
# print("D65 白点 → Lab:", lab_white.cpu().numpy())

def CIE2000(Lab1: torch.Tensor, Lab2: torch.Tensor, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> torch.Tensor:
      # 取出数据
      L1, a1, b1 = Lab1[:,0], Lab1[:,1], Lab1[:,2]
      L2, a2, b2 = Lab2[:,0], Lab2[:,1], Lab2[:,2]

      # 计算C1,C2,C平均
      C1 = torch.sqrt(a1 * a1 + b1 * b1)  # (B,)
      C2 = torch.sqrt(a2 * a2 + b2 * b2)
      C_mean = (C1 + C2) / 2.0

      # 计算G
      pow_Cmean_7 = torch.pow(C_mean, 7)
      pow_25_7 = torch.pow(torch.tensor(25.0, device=C_mean.device), 7)
      G = 0.5 * (1 - torch.sqrt(pow_Cmean_7 / (pow_Cmean_7 + pow_25_7)))

      # 计算a1', a2'
      a1_pi = (1 + G) * a1
      a2_pi = (1 + G) * a2

      # 计算C1', C2',C平均‘
      C1_pi = torch.sqrt(a1_pi * a1_pi + b1 * b1)
      C2_pi = torch.sqrt(a2_pi * a2_pi + b2 * b2)
      C_mean_pi = (C1_pi+C2_pi) /2.0

      # 计算h1', h2'
      h1_pi = torch.where(
            torch.atan2(b1, a1_pi) >=0,torch.atan2(b1, a1_pi),
            torch.atan2(b1, a1_pi) + 2 * torch.pi
            )
      h2_pi = torch.where(
            torch.atan2(b2, a2_pi) >=0,torch.atan2(b2, a2_pi),
            torch.atan2(b2, a2_pi) + 2 * torch.pi
            )

      # 计算h平均’
      h1pi_h2pi = h1_pi - h2_pi
      h_mean_pi = torch.where(
            (C1_pi * C2_pi) == 0, h1_pi + h2_pi,
            torch.where(
                  h1pi_h2pi.abs() > torch.pi,(h1_pi + h2_pi + 2 * torch.pi) / 2,
                  (h1_pi + h2_pi) / 2)
            )

      # 计算T
      T = (1
           - 0.17 * torch.cos(h_mean_pi - torch.pi / 6)
           + 0.24 * torch.cos(2 * h_mean_pi)
           + 0.32 * torch.cos(3 * h_mean_pi + torch.pi / 30)
           - 0.20 * torch.cos(4 * h_mean_pi - 63 * torch.pi / 180)
           )

      # 计算delta h'
      h2pi_h1pi = h2_pi - h1_pi
      delta_h_pi = torch.where(
            (C1_pi * C2_pi) == 0,torch.zeros_like(h2pi_h1pi),
            torch.where(
                  h2pi_h1pi.abs() <= torch.pi, h2pi_h1pi,
                  h2pi_h1pi - 2 * torch.pi * torch.sign(h2pi_h1pi)
             )
      )

      dLp = L2 - L1
      dCp = C2_pi - C1_pi
      dHp = 2 * torch.sqrt(C1_pi * C2_pi) * torch.sin(delta_h_pi / 2.0)

      # 计算加权函数 SL, SC, SH
      L_mean = (L1 + L2) / 2.0
      S_L = 1 + (0.015 * (L_mean - 50) ** 2) / torch.sqrt(20 + (L_mean - 50) ** 2)
      S_C = 1 + 0.045 * C_mean_pi
      S_H = 1 + 0.015 * C_mean_pi * T

      # 计算delta theta
      delta_theta = (30 * torch.pi / 180) * torch.exp(-((h_mean_pi * 180 / torch.pi - 275) / 25) ** 2)

      # 计算R_C, R_T
      R_C = 2 * torch.sqrt(pow_Cmean_7 / (pow_Cmean_7 + pow_25_7))
      R_T = -R_C * torch.sin(2 * delta_theta)

      # 最终计算 delta E
      delta_E = torch.sqrt(
            (dLp / (kL * S_L)) ** 2 +
            (dCp / (kC * S_C)) ** 2 +
            (dHp / (kH * S_H)) ** 2 +
            R_T * (dCp / (kC * S_C)) * (dHp / (kH * S_H))
      )

      return delta_E

def four2five_ts(cordinate: torch.Tensor, XYZ4_ts, XYZ5_ts):
    '''
    在pytorch中，利用伪逆，将四基色下的坐标转为五基色下的坐标

    :param cordinate: (n,4) 的矩阵，每一行为一个四基色下的坐标
    :param XYZ4: 四基色转换矩阵 (3,4)
    :param XYZ5: 五基色转换矩阵 (3,5)
    :return: P_flags 标记后的五基色坐标
    '''
    # 先将输入的四基色颜色在XYZ下表示
    cordinate = cordinate.T
    mat4 = XYZ4_ts.T.detach().clone()
    xyz = torch.matmul(mat4,cordinate)  # cordiante 是(4,)的列向量

    # 求五基色矩阵的伪逆
    mat5 = XYZ5_ts.T.detach().clone()
    mat5_inv = torch.linalg.pinv(mat5)
    P_5 = torch.matmul(mat5_inv, xyz)
    return P_5.T

# # 你的函数名为 CIE2000，这里为了统一测试封装一下
# def test_CIE2000():
#     # 测试用例来自 Sharma et al. (2005)
#     test_cases = [
#         ([50.0000, 2.6772, -79.7751], [50.0000, 0.0000, -82.7485], 2.0425),
#         ([50.0000, 3.1571, -77.2803], [50.0000, 0.0000, -82.7485], 2.8615),
#         ([50.0000, 2.8361, -74.0200], [50.0000, 0.0000, -82.7485], 3.4412),
#     ]
#
#     Lab1 = torch.tensor([x[0] for x in test_cases], dtype=torch.float32)
#     Lab2 = torch.tensor([x[1] for x in test_cases], dtype=torch.float32)
#     expected = torch.tensor([x[2] for x in test_cases], dtype=torch.float32)
#
#     # 调用你的函数
#     results = CIE2000(Lab1, Lab2)
#
#     # 打印对比结果
#     for i, (res, exp) in enumerate(zip(results, expected)):
#         diff = abs(res.item() - exp.item())
#         print(f"Test {i+1}: ΔE00 = {res:.4f}, Expected = {exp:.4f}, Diff = {diff:.4f}")
#
#     # 判断是否全部通过
#     assert torch.allclose(results, expected, atol=1e-3), "测试未通过，请检查实现"
#     print("✅ 所有测试通过！")
#
# # 运行测试
# test_CIE2000()

# 定义数据集
class GamutMp_Dataset(Dataset):
      def __init__(self, bt2020_pts: np.ndarray):
            """
            bt2020_pts: np.ndarray, shape=(N,3)，线性化 BT.2020 RGB，均在 [0,1] 范围
            """
            super().__init__()
            self.x = torch.from_numpy(bt2020_pts).float()   # (N,3), dtype=torch.float32

      def __len__(self):
            return self.x.shape[0]

      def __getitem__(self, idx):
            # 返回 BT.2020 RGB
            return self.x[idx]

# 定义神经网络
class MLP_withflags(nn.Module):
      def __init__(self):
            super(MLP_withflags,self).__init__()
            self.proj = nn.Linear(5, 256)  # 1.投影层
            self.net = nn.Sequential(
                  nn.Linear(256, 256),  # 2.隐藏层1 256
                  nn.LayerNorm(256),
                  nn.GELU(),
                  # 隐藏层 256 → 256
                  nn.Linear(256, 256),  # 3.隐藏层2 256
                  nn.LayerNorm(256),
                  nn.GELU(),
                  # 隐藏层 256 → 128
                  nn.Linear(256, 128),  # 4.隐藏层3 128
                  nn.LayerNorm(128),
                  nn.GELU(),
                  # 隐藏层 128 → 64
                  nn.Linear(128, 64),   # 5.隐藏层4 64
                  nn.LayerNorm(64),
                  nn.GELU(),
                  # 隐藏层 64 → 64
                  nn.Linear(64, 64),    # 6.隐藏层5 64
                  nn.LayerNorm(64),
                  nn.GELU(),
                  # 输出层 64 → 3，并限制到 [0,1]
                  nn.Linear(64, 5),
                  nn.Sigmoid()
            )
      def forward(self,x):
            out = self.proj(x)
            out = self.net(out)
            return out

class MLP_oss_only(nn.Module):
      def __init__(self):
            super(MLP_oss_only,self).__init__()
            self.proj = nn.Linear(4, 256)  # 1.投影层
            self.net = nn.Sequential(
                  nn.Linear(256, 256),  # 2.隐藏层1 256
                  nn.LayerNorm(256),
                  nn.GELU(),
                  # 隐藏层 256 → 256
                  nn.Linear(256, 256),  # 3.隐藏层2 256
                  nn.LayerNorm(256),
                  nn.GELU(),
                  # 隐藏层 256 → 128
                  nn.Linear(256, 128),  # 4.隐藏层3 128
                  nn.LayerNorm(128),
                  nn.GELU(),
                  # 隐藏层 128 → 64
                  nn.Linear(128, 64),   # 5.隐藏层4 64
                  nn.LayerNorm(64),
                  nn.GELU(),
                  # 隐藏层 64 → 64
                  nn.Linear(64, 64),    # 6.隐藏层5 64
                  nn.LayerNorm(64),
                  nn.GELU(),
                  # 输出层 64 → 3，并限制到 [0,1]
                  nn.Linear(64, 5),
                  nn.Sigmoid()
            )
      def forward(self,x):
            out = self.proj(x)
            out = self.net(out)
            return out

# 训练主流程
huber = nn.HuberLoss(delta=1.0)
mse = torch.nn.MSELoss()
alpha = 0.8

def train_mlp_withflag(
    n_samples: int = 4096,
    batch_size: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
      '''
      这个是同时将越界与非越界点都打上标记送入MLP的训练过程

      :param n_samples: 取样个数，默认4096
      :param batch_size: 批次大小，默认1024
      :param epochs: 训练轮次，默认20
      :param lr: 学习率，默认1e-
      :param device: 使用的设备，默认自动判别
      :return: model
      '''
      ## 数据集部分
      # 生成数据集
      print(f'▶ 正在从四基色色彩空间中选取 {n_samples} 个点')

      # 取点
      points4 = GetPoints4(n_samples=n_samples,seed=233)
      # 使用求伪逆的方法
      points5 = four2five(points4, XYZ4, XYZ5)
      pt_flags4, pt_flags5 = flag(points4, points5)
      points_in5, points_oos5 = filter(pt_flags5) # 对points5进行分类
      points_in4, points_oos4 = filter(pt_flags4) # 对points4进行分类


      print(f'  共有{points_in5.shape[0]}个点映射后仍在五基色空间内，有{points_oos5.shape[0]}个点映射后超出空间范围')
      print('▶ 接下来将尝试将标记的四基色坐标送入MLP')



      # 加载训练集
      perm = np.random.permutation(pt_flags4.shape[0])
      points_all = pt_flags4[perm]
      n_train = int(0.8 * points_all.shape[0]) #训练集大小
      train_pts = points_all[:n_train]
      val_pts = points_all[n_train:]

      train_dts = GamutMp_Dataset(train_pts) # 训练集
      val_dts = GamutMp_Dataset(val_pts) #测试集

      train_loader = DataLoader(train_dts, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
      val_loader = DataLoader(val_dts, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

      # -------------------------------

      ## 定义模型、学习率、优化器
      model = MLP_withflags().to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

      # -------------------------------

      # 记录损失，用于绘图
      train_losses = []
      val_losses = []
      fig, ax = plt.subplots()
      plt.ion()

      ## 训练部分
      for e in range(1, epochs+1):
            model.train()
            running_loss = 0.0

            for p4 in train_loader:
                  p4 = p4.to(device) # 移到gpu
                  p4_noflag = p4[:, 0:-1]  # 先把标记去掉，用于进行坐标变换
                  p4_noflag = p4_noflag.to(device)

                  # 先求出p4（原始的四基色下的坐标）的LAB坐标
                  XYZ_org = p42XYZ_ts(p4_noflag)
                  LAB_org = XYZ2LAB(XYZ_org)
                  # 求出若直接投影到边界的坐标
                  direct_pro_t = torch.clamp(four2five_ts(p4_noflag,XYZ4_ts, XYZ5_ts),0,1)
                  direct_pro_t_xyz = p52XYZ_ts(direct_pro_t)
                  direct_pro_t_lab = XYZ2LAB(direct_pro_t_xyz)

                  # 正向传播,得到(batch_size,5)的包含映射后的五基色下坐标的矩阵
                  p5_lin = model(p4)

                  # 转换p5坐标为LAB
                  XYZ_p5 = p52XYZ_ts(p5_lin)
                  LAB_p5 = XYZ2LAB(XYZ_p5)


                  # 计算 CIEDE2000 delta E
                  delta_E = CIE2000(LAB_p5,LAB_org)
                  loss_de = huber(delta_E,torch.zeros_like(delta_E))
                  loss_m =mse(LAB_p5,direct_pro_t_lab)
                  loss = torch.mean(alpha * loss_de + (1 - alpha) * loss_m)

                  # 反向传播
                  optimizer.zero_grad() # 清除已有梯度
                  loss.backward()
                  optimizer.step() # 更新参数

                  running_loss += loss.item() * p4.size(0) # 单个batch的loss总和


            train_e_loss = running_loss / len(train_dts)
            train_losses.append(train_e_loss)

            ## 验证部分
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                  for p4 in val_loader:
                        p4 = p4.to(device)
                        p4_noflag = p4[:, 0:-1]  # 先把标记去掉，用于进行坐标变换
                        p4_noflag = p4_noflag.to(device)

                        # 先求出p4（原始的四基色下的坐标）的LAB坐标
                        XYZ_org = p42XYZ_ts(p4_noflag)
                        LAB_org = XYZ2LAB(XYZ_org)
                        # 求出若直接投影到边界的坐标
                        direct_pro_v = torch.clamp(four2five_ts(p4_noflag, XYZ4_ts, XYZ5_ts), 0, 1)
                        direct_pro_v_xyz = p52XYZ_ts(direct_pro_v)
                        direct_pro_v_lab = XYZ2LAB(direct_pro_v_xyz)

                        # 正向传播,得到(batch_size,5)的包含映射后的五基色下坐标的矩阵
                        p5_lin = model(p4)

                        # 转换p5坐标为LAB
                        XYZ_p5 = p52XYZ_ts(p5_lin)
                        LAB_p5 = XYZ2LAB(XYZ_p5)


                        # 计算 CIEDE2000 delta E
                        delta_E = CIE2000(LAB_p5, LAB_org)
                        loss_de_val = huber(delta_E, torch.zeros_like(delta_E))
                        loss_m_val = mse(LAB_p5, direct_pro_v_lab)
                        loss_val = torch.mean(alpha * loss_de_val + (1 - alpha) * loss_m_val)
                        val_running_loss += loss_val.item() * p4.size(0)

            val_e_loss = val_running_loss / len(val_dts)
            val_losses.append(val_e_loss)
            scheduler.step(val_e_loss)

            print(f'▶ Epoch [{e:02d}/{epochs:02d}]  Train ΔE₀₀: {train_e_loss:.4f}   Val ΔE₀₀: {val_e_loss:.4f}')
            # 绘图
            x_vals = list(range(1, len(train_losses) + 1))
            ax.clear()
            ax.plot(x_vals,train_losses, label='Train Loss', color='blue')
            ax.plot(x_vals,val_losses, label='Validation Loss', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True)
            plt.pause(0.3)  # 暂停0.3秒模拟训练时间

      plt.ioff()
      plt.show()

      return model

def train_mlp_oos_only(
    n_samples: int = 4096,
    batch_size: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
      '''
      这个是仅将越界点（flag = 1）送入MLP的训练过程

      :param n_samples: 取样个数，默认4096
      :param batch_size: 批次大小，默认1024
      :param epochs: 训练轮次，默认20
      :param lr: 学习率，默认1e-
      :param device: 使用的设备，默认自动判别
      :return: model
      '''
      ## 数据集部分
      # 生成数据集
      print(f'▶ 正在从四基色色彩空间中选取 {n_samples} 个点')

      # 取点
      points4 = GetPoints4(n_samples=n_samples,seed=233)
      # 使用求伪逆的方法
      points5 = four2five(points4, XYZ4, XYZ5)
      pt_flags4, pt_flags5 = flag(points4, points5)
      points_in5, points_oos5 = filter(pt_flags5) # 对points5进行分类
      points_in4, points_oos4 = filter(pt_flags4) # 对points4进行分类


      print(f'  共有{points_in5.shape[0]}个点映射后仍在五基色空间内，有{points_oos5.shape[0]}个点映射后超出空间范围')
      print('▶ 接下来将只将映射后越界的四基色坐标送入MLP')



      # 加载训练集
      perm = np.random.permutation(points_oos4.shape[0])
      points_oos = points_oos4[perm]
      n_train = int(0.8 * points_oos.shape[0]) #训练集大小
      train_pts = points_oos[:n_train]
      val_pts = points_oos[n_train:]

      train_dts = GamutMp_Dataset(train_pts) # 训练集
      val_dts = GamutMp_Dataset(val_pts) #测试集

      train_loader = DataLoader(train_dts, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
      val_loader = DataLoader(val_dts, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

      # -------------------------------

      ## 定义模型、学习率、优化器
      model = MLP_oss_only().to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

      # -------------------------------

      # 记录损失，用于绘图
      train_losses = []
      val_losses = []
      fig, ax = plt.subplots()
      plt.ion()

      ## 训练部分
      for e in range(1, epochs+1):
            model.train()
            running_loss = 0.0

            for p4 in train_loader:
                  p4 = p4.to(device) # 移到gpu

                  # 先求出p4（原始的四基色下的坐标）的LAB坐标
                  XYZ_org = p42XYZ_ts(p4)
                  LAB_org = XYZ2LAB(XYZ_org)
                  # 求出若直接投影到边界的坐标
                  direct_pro_t = torch.clamp(four2five_ts(p4,XYZ4_ts, XYZ5_ts),0,1)
                  direct_pro_t_xyz = p52XYZ_ts(direct_pro_t)
                  direct_pro_t_lab = XYZ2LAB(direct_pro_t_xyz)

                  # 正向传播,得到(batch_size,5)的包含映射后的五基色下坐标的矩阵
                  p5_lin = model(p4)

                  # 转换p5坐标为LAB
                  XYZ_p5 = p52XYZ_ts(p5_lin)
                  LAB_p5 = XYZ2LAB(XYZ_p5)


                  # 计算 CIEDE2000 delta E
                  delta_E = CIE2000(LAB_p5,LAB_org)
                  loss_de = huber(delta_E,torch.zeros_like(delta_E))
                  loss_m =mse(LAB_p5,direct_pro_t_lab)
                  loss = torch.mean(alpha * loss_de + (1 - alpha) * loss_m)

                  # 反向传播
                  optimizer.zero_grad() # 清除已有梯度
                  loss.backward()
                  optimizer.step() # 更新参数

                  running_loss += loss.item() * p4.size(0) # 单个batch的loss总和

            train_e_loss = running_loss / len(train_dts)
            train_losses.append(train_e_loss)

            ## 验证部分
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                  for p4 in val_loader:
                        p4 = p4.to(device)

                        # 先求出p4（原始的四基色下的坐标）的LAB坐标
                        XYZ_org = p42XYZ_ts(p4)
                        LAB_org = XYZ2LAB(XYZ_org)
                        # 求出若直接投影到边界的坐标
                        direct_pro_v = torch.clamp(four2five_ts(p4, XYZ4_ts, XYZ5_ts), 0, 1)
                        direct_pro_v_xyz = p52XYZ_ts(direct_pro_v)
                        direct_pro_v_lab = XYZ2LAB(direct_pro_v_xyz)

                        # 正向传播,得到(batch_size,5)的包含映射后的五基色下坐标的矩阵
                        p5_lin = model(p4)

                        # 转换p5坐标为LAB
                        XYZ_p5 = p52XYZ_ts(p5_lin)
                        LAB_p5 = XYZ2LAB(XYZ_p5)


                        # 计算 CIEDE2000 delta E
                        delta_E = CIE2000(LAB_p5, LAB_org)
                        loss_de_val = huber(delta_E, torch.zeros_like(delta_E))
                        loss_m_val = mse(LAB_p5, direct_pro_v_lab)
                        loss_val = torch.mean(alpha * loss_de_val + (1 - alpha) * loss_m_val)
                        val_running_loss += loss_val.item() * p4.size(0)

            val_e_loss = val_running_loss / len(val_dts)
            val_losses.append(val_e_loss)
            scheduler.step(val_e_loss)

            print(f'▶ Epoch [{e:02d}/{epochs:02d}]  Train ΔE₀₀: {train_e_loss:.4f}   Val ΔE₀₀: {val_e_loss:.4f}')
            # 绘图
            x_vals = list(range(1, len(train_losses) + 1))
            ax.clear()
            ax.plot(x_vals,train_losses, label='Train Loss', color='blue')
            ax.plot(x_vals,val_losses, label='Validation Loss', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True)
            plt.pause(0.3)  # 暂停0.3秒模拟训练时间

      plt.ioff()
      plt.show()

      return model


def train():
      print(f'✨使用设备: {device}')
      print(f'▶ 训练开始')

      n_samples = 4096000
      batch_size = 10240
      epochs = 20
      lr = 0.001

      print('1.所有点标记后送入MLP   2.仅送入越界点')
      choose = int(input('请选择一种训练方式: [1/2]'))
      if choose == 1:
            model = train_mlp_withflag(
                  n_samples=n_samples,
                  batch_size=batch_size,
                  epochs=epochs,
                  lr=lr,
                  device=device
            )

            # 保存模型
            file_name = f'models/Q2/all/{datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")}.pth'
            torch.save(model.state_dict(), file_name)

      elif choose == 2:
            model = train_mlp_oos_only(
                  n_samples=n_samples,
                  batch_size=batch_size,
                  epochs=epochs,
                  lr=lr,
                  device=device
            )

            # 保存模型
            file_name = f'models/Q2/oss/{datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")}.pth'
            torch.save(model.state_dict(), file_name)


def project_mlp(ckpt_path, p4, network):
      '''
      此函数用于加载模型并推理

      :param ckpt_path: 模型的路径
      :param p4: 输入四基色下的坐标
      :param nn: 传入神经网络的类名
      :return: 五基色下的坐标
      '''
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model = network().to(device)
      # 创建 Huber 与mse 损失函数，不进行聚合
      huber = nn.SmoothL1Loss(reduction='none')
      mse = nn.MSELoss(reduction='none')

      checkpoint_pth = ckpt_path
      state_dict = torch.load(checkpoint_pth, map_location=device)
      model.load_state_dict(state_dict)
      p4 = torch.from_numpy(p4).float()
      p4 = p4.to(device)


      model.eval()
      with torch.no_grad():
            project_p5_t = model(p4)
            project_p5 = project_p5_t.cpu().numpy()  # 转为numpy数组

            # 计算loss
            if p4.shape[1] == 5:
                  p4 = p4[:,0:-1]
            XYZ_org = p42XYZ_ts(p4)  # 将要映射的四基色下的坐标转为XYZ坐标
            LAB_org = XYZ2LAB(XYZ_org)  # 将要映射的XYZ坐标转为LAB坐标
            direct_pro_v = torch.clamp(four2five_ts(p4, XYZ4_ts, XYZ5_ts), 0, 1)  # 求直接映射的坐标
            direct_pro_v_xyz = p52XYZ_ts(direct_pro_v)
            direct_pro_v_lab = XYZ2LAB(direct_pro_v_xyz)
            XYZ_p5 = p52XYZ_ts(project_p5_t)  # 映射后五基色下的坐标转为XYZ坐标
            LAB_p5 = XYZ2LAB(XYZ_p5)  # 映射后XYZ坐标转为LAB坐标
            delta_E = CIE2000(LAB_p5, LAB_org)  # 求CIEDE2000
            loss_de = huber(delta_E, torch.zeros_like(delta_E))
            loss_m = mse(LAB_p5, direct_pro_v_lab).mean(dim=1)
            loss = alpha * loss_de + (1 - alpha) * loss_m  # loss由CIEDE2000与MSE加权求得
            loss = loss.cpu().numpy()

      return project_p5, loss

def project(p_num=1000):

      proj_pts = GetPoints4(p_num)  # 先取点
      ## 有标记 -----------<begin>-----------
      print("🚩 方法1：带标记的纯MLP映射方法")
      ### 标记，划分越界与非越界点
      proj_pts_pinv = four2five(proj_pts, XYZ4, XYZ5)  # 最小二乘法
      proj_pts_flags4, proj_pts_flags5 = flag(proj_pts, proj_pts_pinv)  # 标记

      flag0_pts, flag1_pts = filter(proj_pts_flags4)  # 需要传入mlp的四基色点
      print(f'共有{flag1_pts.shape[0]}个点越界，')
      ckpt_path = "models/Q2/all/20250524_160416.pth"  # 模型路径
      ### 映射
      pjt_mlp_flags, loss_mlp = project_mlp(ckpt_path, proj_pts_flags4, MLP_withflags)  # 越界点的五基色坐标
      ### 求moss
      #### 只有MLP的loss
      print('🍰以下是MLP的loss，也是整体的loss')
      loss_mlp_95 = np.percentile(loss_mlp, 95, 0)
      loss_mlp_99 = np.percentile(loss_mlp, 99, 0)
      loss_mlp_mean = np.mean(loss_mlp, axis=0)
      # print("❤️ 映射结果:\n", pjt_mlp)
      print(f'整体的loss值的95分位数为: {loss_mlp_95}')
      print(f'整体的loss值的99分位数为: {loss_mlp_99}')
      print(f'整体的平均loss为: {loss_mlp_mean}')
      ## 有标记 -----------<end>-----------

      ## 无标记 -----------<begin>-----------
      print("🚩 方法2：最小二乘法+MLP映射方法")
      ### 标记，划分越界与非越界点
      proj_pts_pinv = four2five(proj_pts, XYZ4, XYZ5)  # 最小二乘法
      proj_pts_flags4, proj_pts_flags5 = flag(proj_pts, proj_pts_pinv)  # 标记

      _, mlp_pts = filter(proj_pts_flags4)  # 需要传入mlp的四基色点
      print(f'共有{mlp_pts.shape[0]}个点越界')
      ckpt_path = "models/Q2/oss/20250524_162828.pth"  # 模型路径
      ### 映射
      direct_pts, _ = filter(proj_pts_flags5)  # 没有越界的点的五基色下的坐标
      pjt_mlp, loss_mlp = project_mlp(ckpt_path, proj_pts, MLP_oss_only)  # 越界点的五基色坐标
      ### 求moss
      #### 求最小二乘法的loss
      xyz_direct_pts = p52XYZ(direct_pts)  # 转为XYZ
      lab_direct_pts = colour.XYZ_to_Lab(xyz_direct_pts)  # 转为LAB
      direct_origin, _ = filter(proj_pts_flags4)  # 映射前的坐标
      xyz_direct_origin = p42XYZ(direct_origin)  # 映射前的坐标
      lab_direct_origin = colour.XYZ_to_Lab(xyz_direct_origin)  # 映射前的坐标
      deltaE_direct = colour.difference.delta_E_CIE2000(lab_direct_pts, lab_direct_origin)
      loss_direct = alpha * deltaE_direct  # 直接映射过程中MSE = 0，因为此处的MSE是直接映射与使用MLP映射直接的MSE

      #### 仅关注MLP的loss
      print('🍰1.以下是MLP的loss')
      loss_mlp_95 = np.percentile(loss_mlp, 95, 0)
      loss_mlp_mean = np.mean(loss_mlp, axis=0)
      # print("❤️ 映射结果:\n", pjt_mlp)
      print(f'MLP的loss值的95分位数为: {loss_mlp_95}')
      print(f'MLP的平均loss为: {loss_mlp_mean}{',可认为是0' if loss_mlp_mean < 1e-6 else ''}')

      print('🍰2.以下是映射后没有越界的点的loss（这些点直接使用最小二乘法）')
      print(f'未越界点的平均loss为: {np.mean(loss_direct, axis=0)}')

      print('🍰3.以下是加上没有越界的点的整体loss')
      loss_all = np.concatenate([loss_direct, loss_mlp])
      loss_all_mean = np.mean(loss_all, axis=0)
      loss_all_95 = np.percentile(loss_all, 95, 0)
      print(f'总体的loss值的95分位数为: {loss_all_95}')
      print(f'总体的平均loss为: {loss_all_mean}')

      ## 无标记 -----------<end>-----------


# main
if __name__ == "__main__":
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      four_bases = np.array([430, 480, 550, 625])
      five_bases = np.array([440, 490, 530, 580, 610])
      XYZ4 = wavelength_2_xyz(four_bases)
      XYZ5 = wavelength_2_xyz(five_bases)
      # plot_xyz_color_vectors(XYZ5, five_bases)
      # plot_gamut_on_chromaticity_diagram(XYZ5,five_bases)

      # D65 白点的 XYZ 值（归一化后 Yn=1.0）
      ref_X = 0.95047
      ref_Y = 1.00000
      ref_Z = 1.08883

      XYZ4_ts = torch.tensor(XYZ4,dtype=torch.float32).to(device)
      XYZ5_ts = torch.tensor(XYZ5,dtype=torch.float32).to(device)

      # 训练
      # train()

      # 推理
      project(p_num=1000)



