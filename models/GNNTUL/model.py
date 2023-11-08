import torch

from config import *

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_geometric.nn as pyg_nn
#from torch_sparse import SparseTensor


class BiLSTM_GNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_classes, n_poi, gnn_tag, dropout=0.5):
        super(BiLSTM_GNN, self).__init__()
        self.num_layers = 1
        self.gnn_tag = gnn_tag
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(n_poi, n_input)
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=self.num_layers, batch_first=False, dropout=dropout,
                            bidirectional=True)

        self.gnn = pyg_nn.GCNConv(n_input, n_hidden)

        self.fc = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=n_classes))

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, x, edge_index):
        # x: tensor of shape (batch_size, seq_len, n_input)
        # edge_index: tensor of shape (2, num_edges)

        # Learn node embeddings using GNNs
        x = x.permute(0, 1)  # (batch_size, seq_len)
        x = self.emb(x) # (batch_size, seq_len, n_input)
        if self.gnn_tag:
          x = self.gnn(x, edge_index)
          x = F.relu(x)


        # Concatenate node embeddings with word embeddings
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, n_hidden)
        #x = x.view(seq_len * batch_size, n_input)
        #lstm_input = torch.cat([x, x + x.new_zeros(x.size()).normal_(0, 0.1)],
        #                       dim=-1)  # (seq_len, batch_size, 2 * n_hidden)

        # Pass the concatenated embeddings through the LSTM model
        outputs, _ = self.lstm(x)
        outputs = self.dropout(outputs)
        val = self.fc(outputs[-1]).view(1, n_classes)

        return val


class BiLSTM(nn.Module):
    """
    BiLSTM + FC
    """
    def __init__(self, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.num_layers = 1
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=self.num_layers, batch_first=False, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=n_classes))

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        outputs, _ = self.lstm(x)
        # outputs = outputs.view(1, -1)
        outputs = self.dropout(outputs)
        # print(x.shape)
        # print(outputs.shape)
        val = self.fc(outputs[-1]).view(1, n_classes)
        return val


# class SageLayer(nn.Module):
#     """
#     Encodes a node's using 'convolutional' GraphSage approach
#     """
#
#     def __init__(self, input_size, out_size, gcn=False):
#         super(SageLayer, self).__init__()
#
#         self.input_size = input_size
#         self.out_size = out_size
#
#         self.gcn = gcn
#         self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))
#
#         self.init_params()
#
#     def init_params(self):
#         for param in self.parameters():
#             nn.init.xavier_uniform_(param)
#
#     def forward(self, self_feats, aggregate_feats, neighs=None):
#         """
#         Generates embeddings for a batch of nodes.
#
#         nodes	 -- list of nodes
#         """
#         if not self.gcn:
#             combined = torch.cat([self_feats, aggregate_feats], dim=1)
#         else:
#             combined = aggregate_feats
#         combined = F.relu(self.weight.mm(combined.t())).t()
#         return combined


# class GraphSage(nn.Module):
#     """docstring for GraphSage"""
#
#     def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
#         super(GraphSage, self).__init__()
#         self.input_size = input_size
#
#         # num_node = 300
#         # self.input_size = num_node
#         self.out_size = out_size
#         self.num_layers = num_layers
#         self.gcn = gcn
#         self.device = device
#         self.agg_func = agg_func
#
#         self.raw_features = raw_features
#         # self.embedding = nn.Embedding(vocab_size, input_size)
#         # self.raw_features = self.embedding.weight
#
#         # self.features_W = nn.Parameter(torch.ones(size=(raw_features.size(1), num_node))).to(self.device)
#
#         # 字典：记录了每个节点的邻居节点列表
#         self.adj_lists = adj_lists
#
#         for index in range(1, num_layers + 1):
#             layer_size = out_size if index != 1 else self.input_size
#             setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))
#
#     def forward(self, nodes_batch):
#         """
#         Generates embeddings for a batch of nodes.
#         nodes_batch	-- batch of nodes to learn the embeddings
#         """
#         lower_layer_nodes = list(nodes_batch)
#         nodes_batch_layers = [(lower_layer_nodes,)]
#         # self.dc.logger.info('get_unique_neighs.')
#         for i in range(self.num_layers):
#             lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
#                 lower_layer_nodes)
#             nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
#
#         assert len(nodes_batch_layers) == self.num_layers + 1
#
#         # pre_hidden_embs = torch.matmul(self.raw_features, self.features_W).to(self.device)
#         pre_hidden_embs = self.raw_features
#         for index in range(1, self.num_layers + 1):
#             nb = nodes_batch_layers[index][0]
#             pre_neighs = nodes_batch_layers[index - 1]
#             # self.dc.logger.info('aggregate_feats.')
#             aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
#             sage_layer = getattr(self, 'sage_layer' + str(index))
#             if index > 1:
#                 nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
#             # self.dc.logger.info('sage_layer.')
#             cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
#                                          aggregate_feats=aggregate_feats)
#             pre_hidden_embs = cur_hidden_embs
#         # 输出为节点的embedding
#         return pre_hidden_embs
#
#     def _nodes_map(self, nodes, hidden_embs, neighs):
#         layer_nodes, samp_neighs, layer_nodes_dict = neighs
#         assert len(samp_neighs) == len(nodes)
#         index = [layer_nodes_dict[x] for x in nodes]
#         return index
#
#     # num_sample = 10
#     def _get_unique_neighs_list(self, nodes, num_sample=20):
#         _set = set
#         to_neighs = [self.adj_lists[int(node)] for node in nodes]
#         if not num_sample is None:
#             _sample = random.sample
#             samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
#                            in to_neighs]
#         else:
#             samp_neighs = to_neighs
#         samp_neighs = [set(samp_neigh) | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
#         _unique_nodes_list = list(set.union(*samp_neighs))
#         i = list(range(len(_unique_nodes_list)))
#         unique_nodes = dict(list(zip(_unique_nodes_list, i)))
#         return samp_neighs, unique_nodes, _unique_nodes_list
#
#     def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
#         # [0], [embedding.weights], ([0], [{0}], {0: 0})
#         # [0], [{0}], {0: 0}
#         unique_nodes_list, samp_neighs, unique_nodes = pre_neighs
#
#         assert len(nodes) == len(samp_neighs)
#         indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
#         assert (False not in indicator)
#         if not self.gcn:
#             samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]
#         # self.dc.logger.info('2')
#         if len(pre_hidden_embs) == len(unique_nodes):
#             embed_matrix = pre_hidden_embs
#         else:
#             embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
#         # self.dc.logger.info('3')
#         mask = torch.zeros(len(samp_neighs), len(unique_nodes))
#         column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
#         row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
#         mask[row_indices, column_indices] = 1
#         # self.dc.logger.info('4')
#
#         if self.agg_func == 'MEAN':
#             num_neigh = mask.sum(1, keepdim=True)
#             # Add a small number for preventing nan value
#             mask = mask.div(num_neigh+1e-10).to(embed_matrix.device)
#             aggregate_feats = mask.mm(embed_matrix)
#
#         elif self.agg_func == 'MAX':
#             # print(mask)
#             indexs = [x.nonzero() for x in mask == 1]
#             aggregate_feats = []
#             # self.dc.logger.info('5')
#             for feat in [embed_matrix[x.squeeze()] for x in indexs]:
#                 if len(feat.size()) == 1:
#                     aggregate_feats.append(feat.view(1, -1))
#                 else:
#                     aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
#             aggregate_feats = torch.cat(aggregate_feats, 0)
#
#         # self.dc.logger.info('6')
#
#         return aggregate_feats
