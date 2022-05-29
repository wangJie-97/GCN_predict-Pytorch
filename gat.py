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

        # 上面的多头注意力都得到了不一样的结果，使用注意力层给聚合起来
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)

        self.act = nn.LeakyReLU()


    def forward(self, inputs, graph):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        # 每一个注意力头用循环取出来，放入list里，然后在最后一维串联起来
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)

class Temporal_Attention_layer(nn.Module):
    """
    compute temporal attention scores
    时空注意力快，如何参考？
    它是怎么被使用的，被整合到最后输出的？
    参数怎么耦合？
    它返回了什么 ？与原先输入相同吗？
    问题就是参数耦合了，找好位置即可，下次继续，这个有写头，论文可以加很多内容，参考着论文的时空块来拼接了
    """

    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        """
        Temporal Attention Layer
        :param num_of_vertices: int
        :param num_of_features: int
        :param num_of_timesteps: int
        """
        super(Temporal_Attention_layer, self).__init__()

        global device
        self.U_1 = torch.randn(num_of_vertices, requires_grad=True).to(device)
        self.U_2 = torch.randn(num_of_features, num_of_vertices, requires_grad=True).to(device)
        self.U_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
        self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor, x^{(r - 1)}_h
                       shape is (batch_size, V, C_{r-1}, T_{r-1})
                       相当于这里的这个项目中的data

        Returns
        ----------
        E_normalized: torch.tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        """
        # _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # N == batch_size
        # V == num_of_vertices
        # C == num_of_features
        # T == num_of_timesteps

        # compute temporal attention scores
        # shape of lhs is (N, T, V)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1),
                           self.U_2)

        # shape is (batch_size, V, T)
        # rhs = torch.matmul(self.U_3, x.transpose((2, 0, 1, 3)))
        rhs = torch.matmul(x.permute((0, 1, 3, 2)), self.U_3)  # Is it ok to switch the position?

        product = torch.matmul(lhs, rhs)  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return E_normalized

class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, device):
        graph = data["graph"][0].to(device)  # [N, N]
        flow = data["flow_x"]  # [B, N, T, C]
        flow = flow.to(device)  # 将流量数据送入设备

        B, N = flow.size(0), flow.size(1)
        flow = flow.view(B, N, -1)  # [B, N, T * C]
        """
       上面是将这一段的时间的特征数据摊平做为特征，这种做法实际上忽略了时序上的连续性
       这种做法可行，但是比较粗糙，当然也可以这么做：
       flow[:, :, 0] ... flow[:, :, T-1]   则就有T个[B, N, C]这样的张量，也就是 [B, N, C]*T
       每一个张量都用一个SubNet来表示，则一共有T个SubNet，初始化定义　self.subnet = [GATSubNet(...) for _ in range(T)]
       然后用nn.ModuleList将SubNet分别拎出来处理，参考多头注意力的处理，同理

       """
        # self.TAt = Temporal_Attention_layer(num_of_vertices, num_of_features, num_of_timesteps)
        prediction = self.subnet(flow, graph).unsqueeze(2)  # [B, N, 1, C]，这个１加上就表示预测的是未来一个时刻

        return prediction


if __name__ == '__main__':  # 测试模型是否合适
    x = torch.randn(32, 278, 6, 2)  # [B, N, T, C]
    graph = torch.randn(32, 278, 278)  # [N, N]
    data = {"flow_x": x, "graph": graph}

    device = torch.device("cpu")

    net = GATNet(in_c=6 * 2, hid_c=6, out_c=2, n_heads=2)

    y = net(data, device)
    print(y.size())

