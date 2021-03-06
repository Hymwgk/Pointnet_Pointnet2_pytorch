import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    计算src点集里面的点，与dst中任意一个点的距离
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: 目标采样点数
        radius:  聚类搜索半径
        nsample:  包围球中最多多少个点
        xyz: input points position data, [B, N, 3]   输入点的位置信息，batch size，点数，channel
        points: input points data, [B, N, D]   输入点的对应法向量信息，batch size，点数，维度（不过一般不用点法向量，为None）
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]  
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    #使用FPS算法抽点，抽npoint个点，返回抽出来的点在xyz点集中的索引
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    #从原始点集合中抽出那npoint个点，组成新的点集new_xyz，里面是三维坐标
    new_xyz = index_points(xyz, fps_idx)  #[B, S, C] batchsize,sample number,channel
    #将新的点集中的点作为中心点来建立包围球，并获得每个球中的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz) #[B, S, nsample]
    #把这些球里面的点抽出来聚合到一起(也是三维坐标)
    grouped_xyz = index_points(xyz, idx) # [B, S, nsample,C] batchsize,sample_number,nsample,channel
    #将group的点坐标转换到以采样点为中心的坐标系中
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) #[B, S, nsample,C]

    #如果输入的原始点云使用了法向量，或者，已经是第二层抽取特征了
    if points is not None:
        #抽取采样点的特征
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        #如果刚开始根本就没有使用法向量，就把坐标变换后的点返回去，此时channel=3
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """特征抽取模块
        npoint：进行FPS采样时，采样出来多少个中心点？npoint个点，比如输入的是1024个点，抽出512个中心点
        radius：聚类包围球的半径大小
        nsample：指的是包围球内部的点数最大不允许超过nsample个
        in_channel：输入数据的Channel数量（是否包含法向量）
        mlp：设置感知层每一层的输出channel数量，也就是每一层的神经元数量，例如mlp=[64, 64, 128]，设置了3个感知层，
                    每个感知层的输出分别为64，64，128
        group_all：主要用在分类中，分类的最后一次使用pointnet，需要对整体所有点
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        #输入的
        last_channel = in_channel
        #构造一个多层感知机(MLP)用来提取每个点的特征
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]    这个是点的位置坐标
            points: input points data, [B, D, N]    这个实际上是点的法向量
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        #将点的坐标调换一下顺序
        xyz = xyz.permute(0, 2, 1)
        #print(xyz.shape)
        #如果使用法向量，将法向量的顺序也调整一下
        if points is not None:
            points = points.permute(0, 2, 1)

        #===============Sample  & Grouping layer  采样+聚类===========
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: 被采样后的点的位置坐标信息, [B, npoint, C]
        # new_points: 采样后的点的  变换后的坐标信息（因为在group之后，进行了一个
        # 类似坐标变换的操作，）？？？, [B, npoint, nsample, C+D] 注意如果原始点云没有法向量，此时的C+D=3 (变换后的坐标)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        #================PointNet层===============================
        #对每个group的点进行特征提取，因为要对new_points进行卷积，
        # new_points本身其实已经将特征进行了拼接，因此channel实际上是变换了的
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        #在这！这里就是maxpool了
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        #返回的，第一是降采样后的点坐标，第二是降采样的点对应的group里面的点的融合特征
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """插值过程
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1) #[B,N,C]
        xyz2 = xyz2.permute(0, 2, 1) #[B,S,C]

        points2 = points2.permute(0, 2, 1)   #[B,S,D]
        #获取需要还原的点数
        B, N, C = xyz1.shape 
        #获取采样后的点数
        _, S, _ = xyz2.shape

        if S == 1:
            #如果采样后的点数之后1个，说明已经抽样抽剩到1个点了
            interpolated_points = points2.repeat(1, N, 1)
        else:
            #计算xyz1中每个点与xyz2的每个点的距离，返回dists= [b,n,m]，其中n=xyz1的点数，m=xyz2的点数
            dists = square_distance(xyz1, xyz2)
            #注意idx是排序后的xyz1的索引
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]  只用前3个最近的点来计算
            #倒数当做权重
            dist_recip = 1.0 / (dists + 1e-8)
            #权重和
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            
            weight = dist_recip / norm
            #疑问是，插值计算出的特征  的顺序是怎么和其他点拼接的？
            #interpolated_points的形状是什么？是[B, N, C]么？还是[B,N-S,C] 呢？（待插值的点数= N-S）
            #利用最近的3个点的特征计算出插值点的特征向量（此时只是手工计算的，后面还要mlp变换一下）
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            print(interpolated_points.shape)

        if points1 is not None:
            #变成[B,N,D] 
            points1 = points1.permute(0, 2, 1)
            #这里就是直接拼接了，直接在特征维度上做拼接，说明，interpolated_points形状[B,N,C]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        #为什么要再添加几个卷积层呢？因为使用上面的简单手工融合的特征
        #并不一定能能够很好的还原出插值点的特征，使用卷积层，再次变换一下
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        #只返回特征就行了，因为xyz坐标数据本身就是有的
        #注意，此时new_points数量已经变多了[B,N,C]
        return new_points

