# @Time    : 2020/8/25 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ


# gat.py
# T为时间，N为节点数，D为节点特征,B是batch分批处理？
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """
        # 不是在这里吧
        # self.TAt = Temporal_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)
        h = self.W(inputs)  # [B, N, D]，一个线性层，就是第一步中公式的 W*h

        # 下面这个就是，第i个节点和第j个节点之间的特征做了一个内积，表示它们特征之间的关联强度
        # 再用graph也就是邻接矩阵相乘，因为邻接矩阵用0-1表示，0就表示两个节点之间没有边相连
        # 那么最终结果中的0就表示节点之间没有边相连
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(
            0)  # [B, N, D]*[B, D, N]->[B, N, N],         x(i)^T * x(j)

        # 由于上面计算的结果中0表示节点之间没关系，所以将这些0换成负无穷大，因为softmax的负无穷大=0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)  # [B, N, N]，在第２维做归一化，就是说所有有边相连的节点做一个归一化，得到了注意力系数
        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D]，，这个是第三步的，利用注意力系数对邻域节点进行有区别的信息聚合


class GATSubNet(nn.Module): # 这个是多头注意力机制
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()

        # 用循环来增加多注意力， 用nn.ModuleList变成一个大的并行的网络
        self.attention_module = nn.ModuleList(
            [GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])  # in_c为输入特征维度，hid_c为隐藏层特征维度
        # 要不把时间块的处理放在这里？

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)

        self.act = nn.LeakyReLU()


    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return: [B, N, hid_c * n_heads]
        """
        # 每一个注意力头用循环取出来，放入list里，然后在最后一维串联起来

        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * n_heads]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads,T=5):
        super(GATNet, self).__init__()
        self.subnet =GATSubNet(in_c, hid_c, out_c, n_heads)
        self.timeB=TimeBlock(in_c, hid_c)
        self.timeB2=TimeBlock(hid_c, out_c)
        self.T=T
        # self.subnet = nn.ModuleList([GATSubNet(in_c, hid_c, out_c, n_heads) for _ in range(T)])
    def forward(self, data, device):
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]
        B, N ,T, C= flow.size(0), flow.size(1),flow.size(2),flow.size(3)
        # flow=flow.view(B,N,-1,self.in_c)
        flow = flow.to(device)  # 将流量数据送入设备
        # t1 = self.timeB(flow)
        t1 = self.timeB(flow)
        # t2=self.timeB2(t1)
        # B, N = flow.size(0), flow.size(1)
        flow = t1.contiguous().view(B, N, -1)  # [B, N, T * C]

        """
       上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
       这种做法可行，但是比较粗糙，当然也可以这么做：
       flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
       每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
       然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理

       """

        prediction = self.subnet(flow, graph).unsqueeze(2)
        # x=torch.zeros(B,N,1,C)
        # for m in self.subnet:
        #     prediction = m(flow, graph).unsqueeze(2)
        #     torch.add(prediction,x)
        # res=torch.div(x,self.T)

        # [B, N, 1, C]，这个１加上就表示预测的是未来一个时刻
        # outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)
        return prediction


if __name__ == '__main__':  # 测试模型是否合适
    x = torch.randn(32, 278, 6, 2)  # [B, N, T, C]
    graph = torch.randn(32, 278, 278)  # [N, N]
    data = {"flow_x": x, "graph": graph}

    device = torch.device("cpu")

    net = GATNet(in_c= 2, hid_c=6, out_c=2, n_heads=2)

    y = net(data, device)
    print(y.size())

