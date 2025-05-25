
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from model.graph_frames import Graph
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch_geometric.nn import GCNConv
# from model.graph_frames import New_Graph
# from model.graph_frame import Graph
from queue import Queue
from common.utils import get_edge_points_GPU
import numpy as np


class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class encoder(nn.Module):  # 2,256,512
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        dim_0 = 2
        dim_2 = 64
        dim_3 = 128
        dim_4 = 256
        dim_5 = 512
        self.fc1 = nn.Linear(dim_0, dim_2)
        self.fc3 = nn.Linear(dim_2, dim_3)
        self.fc4 = nn.Linear(dim_3, dim_4)
        self.fc5 = nn.Linear(dim_4, dim_5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.linear512_256 = linear_block(512, 256, drop)
        self.linear256_256 = linear_block(256, 256, drop)
        self.linear256_512 = linear_block(256, 512, drop)

    def forward(self, x):
        # down
        x = self.linear512_256(x)
        res_256 = x
        x = self.linear256_256(x)
        x = x + res_256
        x = self.linear256_512(x)
        return x

def rescale_distance_matrix(w):
    constant_value = torch.tensor(1.0, dtype=torch.float32)
    return (constant_value + torch.exp(constant_value)) / (constant_value + torch.exp(constant_value - w))

class Globe_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #
        self.attn_drop = nn.Dropout(attn_drop)  # p=0

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 0

    def forward(self, x, f, relative_dist):
        B, N, C = x.shape  # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        d_g = q.shape[-1]

        similarity = q @ k.transpose(-2, -1)
        relative_dist = rescale_distance_matrix(relative_dist)
        similarity_with_distance = similarity * relative_dist
        relu_similarity = F.relu(similarity_with_distance)

        relu_similarity = relu_similarity / torch.sqrt(torch.tensor(d_g, dtype=torch.float32))
        attn = (relu_similarity) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        f = f.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  # b,j,h,c -> b,h,j,c
        x = (attn @ v)
        x = x + f
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CoAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop):
        super().__init__()
        print('CoAttention')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #
        self.attn_drop = nn.Dropout(attn_drop)  # p=0

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 0

    def forward(self, x):
        B, N, C = x.shape  # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # b,heads,17,4 @ b,heads,4,17 = b,heads,17,17
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Joint_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj  # 4,17,17
        self.kernel_size = adj.size(0)
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)

    def forward(self, x):  # b,j,c
        # conv1d
        x = rearrange(x, "b j c -> b c j")
        x = self.conv1d(x)  # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x, "b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, t, v)  # b,k, kc/k, 1, j
        x = torch.einsum('bkctv, kvw->bctw', (x, self.adj))  # bctw   b,c,1,j
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c')
        return x.contiguous()

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

class FFN(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        middle = dim // 2
        self.linear1 = nn.Linear(dim, middle)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(middle, dim)
        self.norm = nn.LayerNorm(dim)  # 保证输出稳定

    def forward(self, x):
        residual = x  # 保存原始特征
        x = self.linear1(x)
        x = gelu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + residual)  # 跳跃连接并归一化
        return x

class Block(nn.Module):  # drop=0.1
    def __init__(self, length, dim, adj, nodes, Direction_adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()

        # GCN
        self.norm1 = norm_layer(length)
        self.GCN_Block1 = Joint_GCN(dim, dim, adj)

        self.adj = adj
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_Fusion = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.FFN = FFN(dim, drop_path)

        # attention
        self.norm_att1 = norm_layer(dim)
        self.num_heads = 8
        qkv_bias = True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25

        self.co_Attention = CoAttentionLayer(dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             attn_drop=attn_drop,
                                             proj_drop=proj_drop)

        self.GlobeAttention = Globe_Attention(dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop,
                                              proj_drop=proj_drop)
        # 512,1024
        self.norm2 = norm_layer(dim)
        self.transEncoderNorm = norm_layer(dim)
        self.FFN = FFN(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.20)
        self.FFN1 = FFN(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.20)
        self.MGCN_Block1 = Bone_Direction_GCN(dim, dim, Direction_adj, 1, attn_drop)

        drop1 = 0.15
        self.trans_drop = nn.Dropout(p=drop1)
        self.before_trans = nn.Dropout(p=drop1)
        self.proportion = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False)
        self.trans_proportion = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False)
        self.Dynamic_Fusion = Dynamic_Fusion(nodes)
        self.edge_collection = {0: (1, 0), 1: (1, 2), 2: (3, 2), 3: (14, 8), 4: (14, 15),
                                5: (16, 15), 6: (4, 0), 7: (4, 5), 8: (6, 5), 9: (11, 8),
                                10: (11, 12), 11: (13, 12), 12: (7, 0), 13: (7, 8), 14: (9, 8),
                                15: (9, 10)}
        self.point_position = np.array(list(self.edge_collection.values()))
        self.point_position = torch.from_numpy(self.point_position)
        self.beta = 0.99

    def forward(self, x, update_vector, edge_indice, angle, relative_dist, before_attn, num):
        # B,J,dim
        res1 = x  # b,j,c
        x_atten = x.clone()
        x_gcn_1 = x.clone()
        update_vector0 = update_vector.clone()
        # GCN
        x_gcn_1 = rearrange(x_gcn_1, "b j c -> b c j").contiguous()
        x_gcn_1 = self.norm1(x_gcn_1)  # b,c,j
        x_gcn_1 = rearrange(x_gcn_1, "b j c -> b c j").contiguous()
        x_gcn_1 = self.GCN_Block1(x_gcn_1)  # b,j,c
        update_vector1 = self.MGCN_Block1(update_vector, edge_indice, angle)

        '''Attention_module'''
        bp, np, fp = x_atten.shape
        bd, nd, fd = update_vector.shape
        x_atten = torch.cat((x_atten, update_vector0), dim=1)
        x_atten = self.norm_att1(x_atten)

        x_atten, attn = self.co_Attention(x_atten)

        x_atten, update_vector = torch.split_with_sizes(x_atten, [np, nd], dim=1)
        attn, _ = torch.split_with_sizes(attn, [np, nd], dim=2)
        if num != 0:
            attn = attn * self.beta + before_attn * (1 - self.beta)
        '''Attention_module'''

        '''Fusion'''
        update_vector1 = update_vector+update_vector1
        x_fusion = self.Dynamic_Fusion(x_atten, update_vector1, attn)
        x_fusion = self.drop_path_Fusion(x_fusion)
        '''Fusion'''

        '''Attention_fusion'''
        x_final = self.GlobeAttention(x_fusion, self.trans_drop(x_atten * self.proportion), relative_dist)
        '''Attention_fusion'''

        x_final = res1 + self.before_trans(x_final * self.trans_proportion) + x_gcn_1

        res3 = x_final  # b,j,c
        x_final = self.norm2(x_final)
        x_final = res3 + self.drop_path(self.FFN(x_final))

        return x_final, attn

class PoseGRAF(nn.Module):
    def __init__(self, depth, embed_dim, adj, nodes, Direction_adj, drop_rate=0.10, length=27):
        super().__init__()

        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                length, embed_dim, adj, nodes, Direction_adj,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x, update_vector, edge_indice, angle, relative_dist):
        num = 0
        attn = torch.zeros(1, 1, 1, 1).cuda()
        for blk in self.blocks:
            x, attn = blk(x, update_vector, edge_indice, angle, relative_dist, attn, num)
            num += 1

        x = self.norm(x)
        return x

class Bone_Direction_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, adj, adjp, drop):
        super(Bone_Direction_GCN, self).__init__()
        middle = out_channels // 2
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv3 = GCNConv(middle, out_channels)
        self.adj = adj  # 4,17,17
        self.kernel_size = adj.size(0)
        #
        self.conv2 = nn.Conv1d(in_channels, middle, kernel_size=1)
        self.conv4 = nn.Conv1d(middle, out_channels, kernel_size=1)
        norm_layer = nn.LayerNorm
        self.norm2 = norm_layer(17)
        self.in_channelsp = in_channels
        self.out_channelsp = out_channels
        self.dropout = nn.Dropout(drop)
        self.proportion = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, vector, edge_index, edge_weight):
        res1 = vector.clone()

        directionCopy = vector.clone()
        vector = self.conv1(vector, edge_index, edge_weight=edge_weight)

        vector = rearrange(vector, "b j c -> b c j").contiguous()

        directionCopy = rearrange(directionCopy, "b j c -> b c j").contiguous()
        directionCopy = self.conv2(directionCopy)
        directionCopy = self.leaky_relu(directionCopy)
        directionCopy = self.conv4(directionCopy)
        directionCopy = rearrange(directionCopy, "b ck j -> b ck 1 j")

        directionCopy = torch.einsum('bctv,vw->bctw', (directionCopy, self.adj))

        vector = rearrange(vector, "b ck j -> b ck 1 j")

        vector = self.dropout(vector + self.proportion * directionCopy)

        vector = rearrange(vector, 'b c 1 j -> b j c')
        vector = res1 + vector
        return vector.contiguous()

# ——————————— 1. Straight-Through Round ———————————
def round_st(x: torch.Tensor) -> torch.Tensor:
    """
    """
    return (x.round() - x).detach() + x


class Dynamic_Fusion(nn.Module):
    def __init__(self, nodes):
        super(Dynamic_Fusion, self).__init__()
        self.nodes = nodes
        self.every = nn.Parameter(torch.empty(17, 1), requires_grad=True).cuda()
        self.Fa_Final = nn.Parameter(torch.empty(17, 1), requires_grad=True).cuda()
        self.super  = nn.Parameter(torch.tensor(0.5))
        self.Fa = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=True)
        # simple initial
        nn.init.uniform_(self.every, 0, 1)
        nn.init.uniform_(self.Fa_Final, 0, 1)

    @property
    def k_cont(self):
        return torch.sigmoid(self.super) * 15. + 1.      # ∈(1,16)

    def forward(self, points, vectors, attntion_scors):
        attntion_scors = attntion_scors.mean(dim=1)
        attntion_scors = attntion_scors.sum(dim=2)
        _, atten_list = attntion_scors.sort(dim=1, descending=True)
        # ——★ 1. 计算连续 k，并用 STE 得到可反传梯度的“整数” k_ste
        super = round_st(self.super)  # 前向≈整数，反向≈连续
        supre_int = int(super.clamp(1, 16).item())  # 真正用于索引的 python int
        top_indices = atten_list[:, :supre_int]
        nodes = self.nodes
        b, n, z = points.size()

        update = points[:, 0, :]
        update = update.unsqueeze(1)
        for index in range(1, n):
            node = nodes[index]
            while node.parent is not None:
                if node.parent.weights[node.value]['weight2'] == 1:
                    root = points[:, node.value, :] + vectors[:, node.parent.weights[node.value]['weight1'], :]
                elif node.parent.weights[node.value]['weight2'] == -1:
                    root = points[:, node.value, :] - vectors[:, node.parent.weights[node.value]['weight1'], :]
                node = node.parent
            root = root.unsqueeze(1)

            update = torch.cat((update, root), dim=1)
        # update = update[torch.arange(update.size(1))[:,None,:],top_indices]
        _, num_top_indices = top_indices.shape
        batch_indices = torch.arange(b).unsqueeze(-1).expand(-1, num_top_indices)
        update = update[batch_indices, top_indices]
        # update = torch.sum(update * self.every.unsqueeze(0).unsqueeze(-1), dim=1)
        if update.size(1) > 1:  # 确保至少有一个非零节点贡献
            update = update.mean(dim=1)
        update = update.unsqueeze(1)
        z = update.size(2)
        q = Queue(maxsize=16)
        index = Queue(maxsize=17)
        q.put(nodes[0].children)
        index.put(0)
        big_point = torch.zeros(b, n, z).cuda()
        big_point[:, 0, :] = update[:, 0, :]
        while q.empty() == False:
            get = index.get()
            if len(nodes[get].children) == 0:
                continue
            child = q.get()
            for ch in child:
                if len(ch.children) != 0:
                    q.put(ch.children)
                index.put(ch.value)
                val = ch.value
                val2 = nodes[get].weights[ch.value]['weight1']
                if nodes[get].weights[ch.value]['weight2'] == 1:
                    big_point[:, val, :] = big_point[:, get, :] - vectors[:, val2, :]
                elif nodes[get].weights[ch.value]['weight2'] == -1:
                    big_point[:, val, :] = big_point[:, get, :] + vectors[:, val2, :]
        big_point = big_point * self.Fa + points
        return big_point

class Model(nn.Module):
    def __init__(self, nodes, args):
        super().__init__()

        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False).cuda(0)
        self.relative_dist = torch.tensor(self.graph.compute_relative_distance_matrix(), dtype=torch.float32).cuda(0)
        self.Direction = nn.Parameter(torch.tensor(self.graph.revert(), dtype=torch.float32), requires_grad=False).cuda(0)

        self.encoderp = encoder(2, args.channel // 2, args.channel)
        self.encoderv = encoder(2, args.channel // 2, args.channel)
        #
        self.PoseGRAF = PoseGRAF(args.layers, args.channel, self.A, nodes, self.Direction, length=args.n_joints)  # 256

        self.edge_collection = {0: (1, 0), 1: (1, 2), 2: (3, 2), 3: (14, 8), 4: (14, 15),
                                5: (16, 15), 6: (4, 0), 7: (4, 5), 8: (6, 5), 9: (11, 8),
                                10: (11, 12), 11: (13, 12), 12: (7, 0), 13: (7, 8), 14: (9, 8),
                                15: (9, 10)}
        self.point_position = np.array(list(self.edge_collection.values()))
        self.point_position = torch.from_numpy(self.point_position)

        self.fcn = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()  # B 17 2

        point = x.clone()
        # encoder
        x = self.encoderp(x)  # B 17 512

        update_matrix, update_vector = get_edge_points_GPU(self.edge_collection, point, self.point_position)
        update_matrix = update_matrix
        batch_indices = torch.nonzero(update_matrix, as_tuple=False)
        batch_graph_idx = batch_indices[:, 0]  # first latitude index
        source_nodes = batch_indices[:, 1]  # second latitude represents original node
        target_nodes = batch_indices[:, 2]  # third latitude represents target node
        edge_indice = torch.stack([source_nodes, target_nodes], dim=0)  # [2, num_edges]
        angle = update_matrix[batch_graph_idx, source_nodes, target_nodes]  # [num_edges]
        update_vector = self.encoderv(update_vector)

        x = self.PoseGRAF(x, update_vector, edge_indice, angle, self.relative_dist)  # B 17 512

        # regression
        x = self.fcn(x)  # B 17 3

        x = rearrange(x, 'b j c -> b 1 j c').contiguous()  # B, 1, 17, 3
        return x



