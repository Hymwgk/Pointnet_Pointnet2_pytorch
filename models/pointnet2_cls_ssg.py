import torch.nn as nn
import torch.nn.functional as F
#导入子模块，主要是
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        #如果使用了表面法向量就channel=6，否则就是3
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        #第一层 最终抽出512个点，半径是0.2, 多层感知器
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        #第二层 输入的channel变化为128+3；
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        #第三层不进行特征提取
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        #num_class就是分类个数
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        #batchsize，channel，
        B, _, _ = xyz.shape
        #如果输入点云数据含有法向量
        if self.normal_channel:
            #
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        #第一层抽取
        l1_xyz, l1_points = self.sa1(xyz, norm)
        #第二层抽取
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #第三层并不是真正的进行提取，只是进行了PointNet的转换
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #
        x = l3_points.view(B, 1024)
        #进行全连接层变换特征，准备分类
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #
        x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        #这里的trans_feat没有用上
        total_loss = F.nll_loss(pred, target)

        return total_loss
