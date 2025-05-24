from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from colour import XYZ_to_RGB
from colour.models import oetf_inverse_BT2020, RGB_COLOURSPACE_sRGB, RGB_Colourspace, RGB_COLOURSPACES, XYZ_to_Lab, RGB_to_XYZ
matplotlib.use('Qt5Agg')

M1 = np.array([[  6.36958048e-01,   1.44616904e-01,   1.68880975e-01],
      [  2.62700212e-01,   6.77998072e-01,   5.93017165e-02],
      [  4.99410657e-17,   2.80726930e-02,   1.06098506e+00]])
GOAL = 'sRGB' # 目标色彩空间


def BT2XYZ(rgb_2020):
      '''
      将BT2020下的坐标转到XYZ下

      输入的rgb_2020是颜色在BT2020下的坐标(向量)，为 1x3 的numpy数组
      '''
      # 转换为numpy数组
      if isinstance(rgb_2020,np.ndarray) != True:
            rgb_2020 = np.array(rgb_2020)
      # 归一化
      if np.all((rgb_2020 >= 0) | (rgb_2020 <=1)) != True:
            rgb_2020 = rgb_2020 / 255.0

      # 线性化 EOTF操作(如果输入的是图片时才需要)
      # rgb_2020 = oetf_inverse_BT2020(rgb_2020)

      xyz = M1 @ rgb_2020.T
      return xyz.T

def GetPoints(n_samples=1000,seed=233):
      '''
      在BT2020上均匀取点
      
      :return: 一个n_samples行三列的矩阵,返回的点在BT2020定义的坐标系下
      '''
      np.random.seed(seed)
      points = np.random.rand(n_samples,3)
      return points

def filter(points):
      '''
      剔除在目标色域（例如是RGB）内的点

      :param points:  GetPoints函数返回的点
      :return: 第一个返回是在BT2020下的坐标，第二个是在目标色域下的坐标
      '''
      xyz = BT2XYZ(points) # 先把GetPoints函数取出的在BT2020下的点的坐标转换为其在XYZ下的坐标
      rgb_goal = XYZ_to_RGB(xyz, RGB_COLOURSPACE_sRGB)
      # 创建掩码，屏蔽在目标色域内的点(axis=1表示是对行进行判断，每行对应一个向量)
      mask_oos = np.any((rgb_goal < 0) | (rgb_goal > 1), axis=1)  # 返回了一个包含布尔值的数组

      # 布尔索引，得到超出目标色域边界的点
      return points[mask_oos],rgb_goal[mask_oos]


# points = GetPoints()
# print(points)
# print(filter(points))

# ----------------------------------------
# pytorch部分
# ----------------------------------------


def BT2XYZ_ts(bt2020: torch.Tensor) -> torch.Tensor:
      '''
      在pytorch中实现的从BT2020到XYZ的转换

      :param srgb: 输入的BT2020坐标
      :return: XYZ下的坐标
      '''
      return torch.matmul(bt2020,M1_ts.T)

def sRGB2XYZ_ts(srgb: torch.Tensor) -> torch.Tensor:
      '''
      在pytorch中实现的从sRGB到XYZ的转换

      :param srgb: 输入的sRGB坐标
      :return: XYZ下的坐标
      '''
      return torch.matmul(srgb,srgb_to_xyz_mat.T)

def XYZ2sRGB(xyz: torch.Tensor) -> torch.Tensor:
      '''
      在pytorch中实现的从XYZ到sRGB的转换

      :param xyz: 输入的XYZ坐标
      :return: sRGB下的坐标
      '''
      return torch.matmul(xyz,M2_ts.T)


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
class MLP(nn.Module):
      def __init__(self):
            super(MLP,self).__init__()
            self.proj = nn.Linear(3, 256)  # 1.投影层
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
                  nn.Linear(64, 3),
                  nn.Sigmoid()
            )
      def forward(self,x):
            out = self.proj(x)
            out = self.net(out)
            return out

# 训练主流程
huber = nn.HuberLoss(delta=1.0)
mse = torch.nn.MSELoss()
alpha = 1.0
beta_L = 0.005
beta_hue = 0.05

def train_mlp(
    n_samples: int = 4096,
    batch_size: int = 1024,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
      ## 数据集部分
      # 生成数据集
      print(f'▶ 正在从BT.2020色彩空间中选取 {n_samples} 个点，\n并筛选出超出目标色域的点')

      points_org = GetPoints(n_samples=n_samples,seed=233)
      points_oos = filter(points_org)[0]
      print(f'  共采集到了{points_oos.shape[0]}个点')

      # 加载训练集
      perm = np.random.permutation(points_oos.shape[0])
      points_oos = points_oos[perm]
      n_train = int(0.8 * points_oos.shape[0]) #训练集大小
      train_pts = points_oos[:n_train]
      val_pts = points_oos[n_train:]

      train_dts = GamutMp_Dataset(train_pts) # 训练集
      val_dts = GamutMp_Dataset(val_pts) #测试集

      train_loader = DataLoader(train_dts, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
      val_loader = DataLoader(val_dts, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

      # -------------------------------

      ## 定义模型、学习率、优化器
      model = MLP().to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

      # -------------------------------

      # 记录损失，用于绘图
      train_losses = []
      train_deltaE = []
      val_losses = []
      val_deltaE = []
      fig, ax = plt.subplots()
      plt.ion()

      ## 训练部分
      for e in range(1, epochs+1):
            model.train()
            running_loss = 0.0
            running_deltaE = 0.0

            for bt in train_loader:
                  bt = bt.to(device) # 移到gpu

                  # 先求出bt（原始的BT2020坐标）的LAB坐标
                  XYZ_org = BT2XYZ_ts(bt)
                  sRGB_org = XYZ2sRGB(XYZ_org)
                  LAB_org = XYZ2LAB(XYZ_org)
                  # 求出若直接投影到边界的sRGB坐标
                  # direct_pro_t = torch.clamp(sRGB_org,0,1)
                  # direct_pro_t_xyz = sRGB2XYZ_ts(direct_pro_t)
                  # direct_pro_t_lab = XYZ2LAB(direct_pro_t_xyz)
                  L_org = LAB_org[:, 0]  # 目标值的 L*
                  a_org, b_org = LAB_org[:, 1], LAB_org[:, 2]
                  h_org = torch.atan2(b_org, a_org)

                  # 正向传播,得到(batch_size,3)的包含映射后的sRGB坐标的矩阵
                  srgb_lin = model(bt)

                  # 转换srgb_lin为LAB
                  XYZ_srgb = sRGB2XYZ_ts(srgb_lin)
                  LAB_srgb = XYZ2LAB(XYZ_srgb)
                  L_srgb = LAB_srgb[:, 0]  # 预测值的 L*
                  a_srgb, b_srgb = LAB_srgb[:, 1], LAB_srgb[:, 2]
                  h_srgb = torch.atan2(b_srgb, a_srgb)

                  # 计算 CIEDE2000 delta E
                  delta_E = CIE2000(LAB_srgb,LAB_org)
                  loss_de = huber(delta_E,torch.zeros_like(delta_E))
                  loss_L = torch.mean((L_srgb - L_org) ** 2)

                  delta_h = torch.remainder(h_srgb - h_org + torch.pi, 2 * torch.pi) - torch.pi
                  loss_hue = torch.mean(delta_h ** 2)
                  loss  = (
                            alpha * loss_de +
                            beta_L * loss_L +
                            beta_hue * loss_hue
                        )

                  # 反向传播
                  optimizer.zero_grad() # 清除已有梯度
                  loss.backward()
                  optimizer.step() # 更新参数

                  running_loss += loss.item() * bt.size(0) # 单个batch的loss总和
                  running_deltaE += torch.mean(delta_E) * bt.size(0)

            train_e_loss = running_loss / len(train_dts)
            train_e_deltaE = running_deltaE / len(train_dts)
            train_losses.append(train_e_loss)
            train_deltaE.append(train_e_deltaE)


            ## 验证部分
            model.eval()
            val_running_loss = 0.0
            val_running_deltaE = 0.0
            with torch.no_grad():
                  for bt in val_loader:
                        bt = bt.to(device)
                        XYZ_org = BT2XYZ_ts(bt)
                        sRGB_org = XYZ2sRGB(XYZ_org)
                        LAB_org = XYZ2LAB(XYZ_org)

                        # direct_pro_v = torch.clamp(sRGB_org, 0, 1)
                        # direct_pro_v_xyz = sRGB2XYZ_ts(direct_pro_v)
                        # direct_pro_v_lab = XYZ2LAB(direct_pro_v_xyz)
                        L_org = LAB_org[:, 0]  # 目标值的 L*
                        a_org, b_org = LAB_org[:, 1], LAB_org[:, 2]
                        h_org = torch.atan2(b_org, a_org)

                        srgb_lin = model(bt)
                        XYZ_srgb = sRGB2XYZ_ts(srgb_lin)
                        LAB_srgb = XYZ2LAB(XYZ_srgb)
                        L_srgb = LAB_srgb[:, 0]  # 预测值的 L*
                        a_srgb, b_srgb = LAB_srgb[:, 1], LAB_srgb[:, 2]
                        h_srgb = torch.atan2(b_srgb, a_srgb)

                        delta_E = CIE2000(LAB_srgb, LAB_org)
                        loss_de_val = huber(delta_E, torch.zeros_like(delta_E))

                        loss_L = torch.mean((L_srgb - L_org) ** 2)
                        delta_h = torch.remainder(h_srgb - h_org + torch.pi, 2 * torch.pi) - torch.pi
                        loss_hue = torch.mean(delta_h ** 2)
                        loss_val = (
                            alpha * loss_de_val +
                            beta_L * loss_L +
                            beta_hue * loss_hue
                        )
                        val_running_loss += loss_val.item() * bt.size(0)
                        val_running_deltaE += torch.mean(delta_E) * bt.size(0)

            val_e_loss = val_running_loss / len(val_dts)
            val_e_deltaE = val_running_deltaE / len(val_dts)
            val_losses.append(val_e_loss)
            val_deltaE.append(val_e_deltaE)
            scheduler.step(val_e_loss)

            print(f'▶ Epoch [{e:02d}/{epochs:02d}]  Train ΔE₀₀: {train_e_deltaE:.4f}   Val ΔE₀₀: {val_e_deltaE:.4f}')
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

def project(ckpt_path,bt):
      '''
      此函数用于加载模型并推理

      :param ckpt_path: 模型的路径
      :param bt: 输入BT2020坐标
      :return: sRGB坐标与包含loss的
      '''
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model = MLP().to(device)
      # 创建 Huber 与mse 损失函数，不进行聚合
      huber = nn.SmoothL1Loss(reduction='none')
      mse = nn.MSELoss(reduction='none')

      checkpoint_pth = ckpt_path
      state_dict = torch.load(checkpoint_pth, map_location=device)
      model.load_state_dict(state_dict)
      bt = torch.from_numpy(bt).float()
      bt = bt.to(device)

      model.eval()
      with torch.no_grad():
            project_srgb_t = model(bt)
            XYZ_org = BT2XYZ_ts(bt)
            sRGB_org = XYZ2sRGB(XYZ_org)
            project_srgb = project_srgb_t.cpu().numpy() # 转为numpy数组



            # 计算loss
            XYZ_org = BT2XYZ_ts(bt)  # 将要映射的BT2020坐标转为XYZ坐标
            LAB_org = XYZ2LAB(XYZ_org)  # 将要映射的XYZ坐标转为LAB坐标

            L_org = LAB_org[:, 0]  # 目标值的 L*
            a_org, b_org = LAB_org[:, 1], LAB_org[:, 2]
            h_org = torch.atan2(b_org, a_org)


            direct_pro_v = torch.clamp(sRGB_org, 0, 1) # 求直接映射的坐标
            direct_pro_v_xyz = sRGB2XYZ_ts(direct_pro_v)
            direct_pro_v_lab = XYZ2LAB(direct_pro_v_xyz)
            XYZ_srgb = sRGB2XYZ_ts(project_srgb_t) # 映射后sRGB坐标转为XYZ坐标
            LAB_srgb = XYZ2LAB(XYZ_srgb) # 映射后XYZ坐标转为LAB坐标

            L_srgb = LAB_srgb[:, 0]  # 预测值的 L*
            a_srgb, b_srgb = LAB_srgb[:, 1], LAB_srgb[:, 2]
            h_srgb = torch.atan2(b_srgb, a_srgb)

            delta_E = CIE2000(LAB_srgb, LAB_org) # 求CIEDE2000
            loss_de = huber(delta_E, torch.zeros_like(delta_E))
            # loss_m = mse(LAB_srgb, direct_pro_v_lab).mean(dim=1)
            loss_L = torch.mean((L_srgb - L_org) ** 2)
            delta_h = torch.remainder(h_srgb - h_org + torch.pi, 2 * torch.pi) - torch.pi
            loss_hue = torch.mean(delta_h ** 2)
            loss = (
                    alpha * loss_de +
                    beta_L * loss_L +
                    beta_hue * loss_hue
            )  # loss由CIEDE2000与MSE加权求得
            loss = loss.cpu().numpy()

      return project_srgb, loss, delta_E.cpu().numpy()

def train():
      print(f'✨使用设备: {device}')
      print(f'▶ 训练开始')

      model = train_mlp(
            n_samples=4096000,
            batch_size=10240,
            epochs=20,
            lr=1e-3,
            device=device
      )

      # 保存模型
      file_name = f'models/Q1/{datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")}.pth'
      torch.save(model.state_dict(), file_name)


# main
if __name__ == "__main__":
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

      # BT2020 → XYZ变换矩阵
      M1_ts = torch.tensor(M1, dtype=torch.float32)
      # XYZ → sRGB变换矩阵
      M2_ts = torch.tensor([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.204 ,  1.057 ]
      ], dtype=torch.float32)
      # sRGB → XYZ变换矩阵
      srgb_to_xyz_mat = torch.tensor([
            [0.41239080, 0.35758434, 0.18048079],
            [0.21263901, 0.71516868, 0.07219232],
            [0.01933082, 0.11919478, 0.95053215]
      ], dtype=torch.float32)
      # D65 白点的 XYZ 值（归一化后 Yn=1.0）
      ref_X = 0.95047
      ref_Y = 1.00000
      ref_Z = 1.08883

      M1_ts = M1_ts.to(device)
      M2_ts = M2_ts.to(device)
      srgb_to_xyz_mat = srgb_to_xyz_mat.to(device)

      # 训练
      train()

      # 推理
      # pts = GetPoints(1000)
      # proj_pts,_ = filter(pts)
      # # direct_pts = pts - proj_pts # 无需映射的点，这些点只需进行简单的坐标变换
      # ckpt_path = "models/Q1/20250524_204258.pth" #模型路径
      #
      # pjt,loss,delta_E = project(ckpt_path, proj_pts) # 送入MLP
      # # print("❤️ MLP映射结果:\n", pjt)
      # loss_95 = np.percentile(loss, 95, 0)
      # loss_mean = np.mean(loss, axis=0)
      # delta_E_mean = np.mean(delta_E,axis=0)
      # print(f'MLP的loss值的95分位数为: {loss_95}')
      # print(f'MLP平均loss为: {loss_mean}')
      # print(f'MLP平均delta_E为: {delta_E_mean}')


# ----------------------

from mpl_toolkits.mplot3d import Axes3D  # 只要导入一下即可启用 3D

def plot_srgb_colors(srgb_array):
    """
    把一组 sRGB 颜色（三个通道均在 0–1）在 3D 空间里画出来。
    srgb_array: numpy.ndarray，shape = (N, 3)，每一行是 [R, G, B]，值都在 [0,1]。
    """
    # 检查输入
    srgb_array = np.asarray(srgb_array, dtype=np.float32)
    if srgb_array.ndim != 2 or srgb_array.shape[1] != 3:
        raise ValueError("输入必须是形状 (N, 3) 的数组，且每个元素都在 [0,1]。")
    if srgb_array.min() < 0 or srgb_array.max() > 1:
        raise ValueError("sRGB 通道值必须在 0 到 1 之间。")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 拆出 R、G、B 三个坐标
    xs = srgb_array[:, 0]
    ys = srgb_array[:, 1]
    zs = srgb_array[:, 2]

    # 在三维中画散点，用本身的 RGB 值来做颜色
    ax.scatter(xs, ys, zs,
               c=srgb_array,      # 每个点的颜色就是对应的 [R,G,B]
               marker='o',
               s=50,             # 点的大小，可根据需要调
               depthshade=True)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title('sRGB Colors in 3D (R–G–B)')

    plt.tight_layout()
    plt.show()


# —— 下面是示例用法 ——
    # 举几个典型颜色

# plot_srgb_colors(pjt)








