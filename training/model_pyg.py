import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price + 1e-6) #是不是这里要加偏移量
    #return_ratio = torch.sub(prediction, base_price)
    return_ratio = torch.where(torch.isinf(return_ratio), torch.full_like(return_ratio, 0), return_ratio)
    # return ratio's mse loss
    #reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    
    mse_loss = torch.nn.MSELoss()
    reg_loss = mse_loss(return_ratio, ground_truth)
    
    # formula (4-6)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio #预测的当天收盘价和真实的前天收盘价的收益率


class GCNLSTM(nn.Module):#三个大小要改！
    def __init__(self, batch_size, hidden_dim, num_layers, edge_index):#inner_prod默认值是False
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.lstm = nn.LSTM(5, hidden_dim, num_layers, batch_first=True)#这里定义了LSTM的参数
        self.graph_layer = GCN(self.hidden_dim, self.hidden_dim)#这里要debug一下，看下hidden_dim是否需要乘一个倍数
        self.fc = nn.Linear(hidden_dim, 1)#这里是对应后面的LSTM输出和图输出的拼接

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        x, (hn, cn) = self.lstm(inputs, (h0.cuda(), c0.cuda()))
        x = x[:, -1, :]
        
        outputs_graph = self.graph_layer(x, self.edge_index)
        #outputs_cat = torch.cat([x, outputs_graph], dim=1)#将LSTM和图的输出拼在一起，输出给激活函数
        prediction = F.leaky_relu(self.fc(outputs_graph), negative_slope=0.2)
        return prediction

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, inputs, rel_encoding):
        x = self.conv1(inputs, rel_encoding)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, rel_encoding)
        return x

    
class GATLSTM_multi(nn.Module):#三个大小要改！
    def __init__(self, batch_size, hidden_dim, num_layers, edge_index, num_heads):#inner_prod默认值是False
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.num_heads = num_heads
        self.lstm = nn.LSTM(5, hidden_dim, num_layers, batch_first=True)#这里定义了LSTM的参数
        self.graph_layer = GAT2(self.hidden_dim, self.hidden_dim, self.num_heads)#这里要debug一下，看下hidden_dim是否需要乘一个倍数
        self.fc = nn.Linear(hidden_dim, 1)#这里是对应后面的LSTM输出和图输出的拼接
        self.fcr = nn.Linear(hidden_dim, 1)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        x, (hn, cn) = self.lstm(inputs, (h0.cuda(), c0.cuda()))
        x = x[:, -1, :]
        
        outputs_graph_1 = self.graph_layer(x, self.edge_index[0])
        outputs_graph_2 = self.graph_layer(x, self.edge_index[1])
        outputs_graph_3 = self.graph_layer(x, self.edge_index[2])

        a1 = torch.exp(self.fcr(outputs_graph_1))
        a2 = torch.exp(self.fcr(outputs_graph_2))
        a3 = torch.exp(self.fcr(outputs_graph_3))
        a_sum = a1 + a2 + a3
        alpha1 = torch.div(a1, a_sum)
        alpha2 = torch.div(a2, a_sum)
        alpha3 = torch.div(a3, a_sum)
        outputs_graph = alpha1*outputs_graph_1 + alpha2*outputs_graph_2 + alpha3*outputs_graph_3
        prediction = F.leaky_relu(self.fc(outputs_graph), negative_slope=0.2)
        return prediction, (alpha1.cpu().detach().numpy(), alpha2.cpu().detach().numpy(), alpha3.cpu().detach().numpy())


class GATLSTM_multi_pro(nn.Module):#三个大小要改！
    def __init__(self, batch_size, hidden_dim, num_layers, edge_index, num_heads):#inner_prod默认值是False
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.num_heads = num_heads
        self.lstm = nn.LSTM(5, hidden_dim, num_layers, batch_first=True)#这里定义了LSTM的参数
        self.graph_layer = GAT2(self.hidden_dim, self.hidden_dim, self.num_heads)#这里要debug一下，看下hidden_dim是否需要乘一个倍数
        self.fc = nn.Linear(hidden_dim, 1)#这里是对应后面的LSTM输出和图输出的拼接
        self.fcr = nn.Linear(hidden_dim, 1, bias=False)
        self.h = torch.nn.Parameter(torch.zeros(self.batch_size, 2*self.batch_size), requires_grad=True)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        x, (hn, cn) = self.lstm(inputs, (h0.cuda(), c0.cuda()))
        x = x[:, -1, :]
        
        outputs_graph_1 = self.graph_layer(x, self.edge_index[0])
        outputs_graph_2 = self.graph_layer(x, self.edge_index[1])
        outputs_graph_3 = self.graph_layer(x, self.edge_index[2])

        a1 = torch.exp(F.leaky_relu(self.h @ self.fcr(torch.cat((x, outputs_graph_1),0)), negative_slope=0.2))
        a2 = torch.exp(F.leaky_relu(self.h @ self.fcr(torch.cat((x, outputs_graph_2),0)), negative_slope=0.2))
        a3 = torch.exp(F.leaky_relu(self.h @ self.fcr(torch.cat((x, outputs_graph_3),0)), negative_slope=0.2))#N*U
        a_sum = a1 + a2 + a3
        alpha1 = torch.div(a1, a_sum)
        alpha2 = torch.div(a2, a_sum)
        alpha3 = torch.div(a3, a_sum)
        outputs_graph = alpha1*outputs_graph_1 + alpha2*outputs_graph_2 + alpha3*outputs_graph_3
        prediction = F.leaky_relu(self.fc(outputs_graph), negative_slope=0.2)
        return prediction, (alpha1.cpu().detach().numpy(), alpha2.cpu().detach().numpy(), alpha3.cpu().detach().numpy())


class GAT2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads):
        super(GAT2, self).__init__()
        self.conv1 = GATConv(num_node_features, 16, heads=num_heads)
        self.conv2 = GATConv(16*num_heads, num_classes)

    def forward(self, inputs, edge_index):
        x = self.conv1(inputs, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

class GATLSTM_multi_temporal(nn.Module):#三个大小要改！
    def __init__(self, batch_size, hidden_dim, num_layers, rel_encoding, rel_mask):#inner_prod默认值是False
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.relation = nn.Parameter(torch.tensor(rel_encoding, dtype=torch.float32), requires_grad=False)#转变类型
        self.rel_mask = nn.Parameter(torch.tensor(rel_mask, dtype=torch.float32), requires_grad=False)
        self.rel_weight = nn.Linear(rel_encoding.shape[-1], 1)#rel_encoding.shape[-1]就是关系种类数量
        self.lstm = nn.LSTM(5, hidden_dim, num_layers, batch_first=True)#这里定义了LSTM的参数
        self.graph_layer = GAT3(self.hidden_dim, self.hidden_dim, 1)#这里要debug一下，看下hidden_dim是否需要乘一个倍数
        self.fc = nn.Linear(hidden_dim, 1)#这里是对应后面的LSTM输出和图输出的拼接
        self.fcr = nn.Linear(hidden_dim, 1)
        #self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        x, (hn, cn) = self.lstm(inputs, (h0.cuda(), c0.cuda()))
        x = x[:, -1, :]
        
        rel_weight = self.rel_weight(self.relation)[:, :, -1]
        
        inner_weight = x @ x.t()
        weight = inner_weight @ rel_weight
        weight_softmax = F.softmax(self.rel_mask + weight, dim=1)#.cpu().detach().numpy()
        
        #weight_softmax = F.softmax(rel_weight, dim=1)#.cpu().detach().numpy()
        #weight_softmax = torch.tanh(rel_weight)
        
        idx = torch.nonzero(weight_softmax).T  
        data = weight_softmax[idx[0],idx[1]]
        
        outputs_graph = self.graph_layer(x, idx, data)

        prediction = F.leaky_relu(self.fc(outputs_graph), negative_slope=0.2)
        return prediction, self.rel_weight.weight.cpu().detach().numpy()[0,:3]#weight_softmax


class GAT3(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads):
        super(GAT3, self).__init__()
        self.conv1 = GATConv(num_node_features, 16, heads=num_heads)
        self.conv2 = GATConv(16*num_heads, num_classes)

    def forward(self, inputs, edge_index, edge_attr):
        x = self.conv1(inputs, edge_index, edge_attr)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x



class GATLSTM(nn.Module):#三个大小要改！
    def __init__(self, batch_size, hidden_dim, num_layers, edge_index, num_heads):#inner_prod默认值是False
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_index = edge_index
        self.num_heads = num_heads
        self.lstm = nn.LSTM(5, hidden_dim, num_layers, batch_first=True)#这里定义了LSTM的参数
        self.graph_layer = GAT(self.hidden_dim, self.hidden_dim, self.num_heads)#这里要debug一下，看下hidden_dim是否需要乘一个倍数
        self.fc = nn.Linear(hidden_dim, 1)#这里是对应后面的LSTM输出和图输出的拼接

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_dim).requires_grad_()
        x, (hn, cn) = self.lstm(inputs, (h0.cuda(), c0.cuda()))
        x = x[:, -1, :]
        
        outputs_graph, edge_index_new, alpha = self.graph_layer(x, self.edge_index)
        #outputs_cat = torch.cat([x, outputs_graph], dim=1)#将LSTM和图的输出拼在一起，输出给激活函数
        prediction = F.leaky_relu(self.fc(outputs_graph), negative_slope=0.2)
        return prediction, (edge_index_new, alpha)

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 16, heads=num_heads)
        self.conv2 = GATConv(16*num_heads, num_classes)

    def forward(self, inputs, edge_index):
        x = self.conv1(inputs, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x, (edge_index_new, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
        return x, edge_index_new, alpha