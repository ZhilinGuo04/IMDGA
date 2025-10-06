import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv,SGConv
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_torch_sparse_tensor
from torch_sparse import SparseTensor
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=False):
        super(GCN, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels,  normalize=True, cached=False,add_self_loops = True))
        # self.lns = torch.nn.ModuleList()
        # self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=True, cached=False,add_self_loops = True))
            # self.lns.append(torch.nn.LayerNorm(hidden_channels))
        # self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels,  normalize=True, cached=False,add_self_loops = True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for ln in self.lns:
        #     ln.reset_parameters()

    def forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # obtain output from the i-th layer
            if layers == i+1:
                return x
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        x = x.log_softmax(dim=-1) # applied based on parameter. Default: not applied
        return x
    
    
class SGC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=False):
        super(SGC, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConv(in_channels, hidden_channels, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels, cached=False))
        self.convs.append(SGConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # obtain output from the i-th layer
            if layers == i+1:
                return x
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def normalization(adjacency):
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        L = d_hat.dot(adjacency).dot(d_hat).tocoo()
        #  torch.sparse.FloatTensor
        indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
        values = torch.from_numpy(L.data.astype(np.float32))
        tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
        return tensor_adjacency


class EGCNGuard(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True, attention_drop=True, threshold=0.1):
        super(EGCNGuard, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=True, cached=False,add_self_loops = True))
        self.convs.append(GCNConv(hidden_channels, out_channels,  normalize=True, cached=False,add_self_loops = True))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # specific designs from GNNGuard
        self.attention_drop = attention_drop
        # the definition of p0 is confusing comparing the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0. #Parameter(torch.rand(1)) 
        self.prune_edge = True
        self.threshold = threshold

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        

    def forward(self, x, edge_index,edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            new_edge_index = self.att_coef(x, edge_index)
            x = conv(x, new_edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        new_edge_index = self.att_coef(x, edge_index)
        x = self.convs[-1](x, new_edge_index)
        return x.log_softmax(dim=-1)


    def att_coef(self, features, edge_index):
        with torch.no_grad():
            row, col = edge_index
            n_total = features.size(0)
            if features.size(1) > 512 or row.size(0)>5e5:
                # an alternative solution to calculate cosine_sim
                # feat_norm = F.normalize(features,p=2)
                batch_size = int(1e8//features.size(1))
                bepoch = row.size(0)//batch_size+(row.size(0)%batch_size>0)
                sims = []
                for i in range(bepoch):
                    st = i*batch_size
                    ed = min((i+1)*batch_size,row.size(0))
                    sims.append(F.cosine_similarity(features[row[st:ed]],features[col[st:ed]]))
                sims = torch.cat(sims,dim=0)
            else:
                sims = F.cosine_similarity(features[row],features[col])
            mask = torch.logical_or(sims>=self.threshold,row==col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            edge_index = torch.stack([row, col])
            # normalize sim
        return edge_index

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=False):
        super(GAT, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.dropout = dropout
        self.use_ln = use_ln
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, 8, heads=8))
        self.convs.append(GATConv(8 * 8, out_channels, heads=1, concat=False))

        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for ln in self.lns:
        #     ln.reset_parameters()

    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        x = x.log_softmax(dim=-1) # applied based on parameter. Default: not applied
        return x
    
class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=False):
        super(SAGE, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.dropout = dropout
        self.use_ln = use_ln
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for ln in self.lns:
        #     ln.reset_parameters()

    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        x = x.log_softmax(dim=-1) # applied based on parameter. Default: not applied
        return x
    


class RobustGCNConv(nn.Module):
    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()
    
    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""
        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        """
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= mean
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= F.dropout(var, p=self.dropout, training=self.training)
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)+1e-6 #avoid abnormal gradient
            attention = torch.exp(-var)

            mean = mean * attention
            var = var * attention * attention
            mean = adj0 @ mean
            var = adj1 @ var

        return mean, var

from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    
    # if not adj_t.has_value():
    #     adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

class RobustGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
    # def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(in_channels, hidden_channels, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(hidden_channels, hidden_channels,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(hidden_channels, out_channels, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.use_ln = True
        self.gaussian = None
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        row,col = edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(x.size(0), x.size(0)))
        adj0, adj1 = gcn_norm(adj), gcn_norm(adj, order=-1.0)
        # adj0, adj1 = normalize_adj(adj), normalize_adj(adj, -1.0)
        mean = x
        var = x
        for layer in self.layers:
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
        # if self.gaussian == None:
        # self.gaussian = MultivariateNormal(torch.zeros(var.shape),
        #         torch.diag_embed(torch.ones(var.shape)))
        sample = torch.randn(var.shape).to(x.device)
        # sample = self.gaussian.sample().to(x.device)
        output = mean + sample * torch.pow(var, 0.5)

        return output.log_softmax(dim=-1)