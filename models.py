
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GINConv, GATConv
from torch_geometric.nn import *
import torch
import torch.nn as nn

class GTNN(torch.nn.Module):

    def get_decoder_dim(self, ir_dim):
        decoder_dim = None
        if not self.add_additional_feature:
            decoder_dim = 2 * self.gs_dim
        elif self.fusion_type == "concat":
            decoder_dim = 2 * self.gs_dim + ir_dim
        elif self.fusion_type == "inner":
            decoder_dim = 1
        elif self.fusion_type == "outer":
            decoder_dim = 2 * self.gs_dim * ir_dim
        elif self.fusion_type == "conv":
            decoder_dim = ((2 * self.gs_dim + ir_dim) - 1) * 20
        return decoder_dim

    def __init__(self, node_feat_dim, add_additional_feature, device, gs_dim, additional_feature_dim, fusion_type, model_type):
        super(GTNN, self).__init__()
        print('Layers are on device = {}'.format(device))
        #self.dim = node_feat_dim
        self.add_additional_feature = add_additional_feature
        self.device = device
        self.gs_dim = gs_dim

        self.num_layers = 1
        self.convs = torch.nn.ModuleList()
        if model_type == "sage":
            self.convs.append(SAGEConv(node_feat_dim, gs_dim).to(device))
        elif model_type == "gcn":
            self.convs.append(GCNConv(node_feat_dim, gs_dim).to(device))
        elif model_type == "gat":
            self.convs.append(GATConv(node_feat_dim, gs_dim).to(device))
        elif model_type == "gin":
            self.convs.append(GINConv(nn.Linear(node_feat_dim, gs_dim)))

        else:
            pass


        self.fusion_type = "concat" if fusion_type is None else fusion_type

        new_add_feat_dim = 2 * self.gs_dim if self.fusion_type == "inner" else 2 * self.gs_dim
        self.hidden_layer_add_feat = nn.Linear(additional_feature_dim, new_add_feat_dim).to(device)

        self.conv1d = nn.Conv1d(1, 20, 2).to(device) if self.fusion_type == "conv" else None

        decoder_input_dim = self.get_decoder_dim(new_add_feat_dim)
        print("decoder_dim = ", decoder_input_dim)
        self.hidden_layer_1 = nn.Linear(decoder_input_dim, 200).to(device)
        self.hidden_layer_2 = nn.Linear(200, 1).to(device)
        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)


    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            x = self.relu(x)
        x = self.get_minibatch_embeddings(x, batch)
        x = self.decode(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch_, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch_].to(device).sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        assert (index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        x = x[set_indices_batch]  # shape [B, set_size, F]

        x = self.fusion(x, batch.ir_score) if self.add_additional_feature else x.reshape(-1, 2 * self.gs_dim)

        return x


    def fusion(self, x, feature_vecs):

        # x.shape =  (256, 2, 100)
        # this method combains IR scores with concantenated node representations
        # (u CONCAT v) FUSION_OPERATOR (IR scores)
        # here, fusion operator is fusion_type

        x = x.reshape(-1, 2 * self.gs_dim)

        if self.fusion_type == "concat":

            S = self.hidden_layer_add_feat(feature_vecs)  # S dim = 4
            S = self.relu(S)
            x = torch.cat([x, S], dim=1)
            return x

        elif self.fusion_type == "inner":
            S = self.hidden_layer_add_feat(feature_vecs)  # S dim = 200
            S = self.relu(S)
            x = torch.sum(S * x, dim=1)
            x = x.unsqueeze(1)
            return x

        elif self.fusion_type == "outer":
            S = self.hidden_layer_add_feat(feature_vecs)  # S dim = 4
            outer_product = torch.bmm(x.unsqueeze(2), S.unsqueeze(1))
            x = outer_product.reshape(outer_product.shape[0], -1)
            return x

        elif self.fusion_type == "conv":
            S = self.hidden_layer_add_feat(feature_vecs)  # S dim = 4
            Z = torch.cat([x, S], dim=1)
            Z = Z.unsqueeze(1)
            conv_output = self.conv1d(Z)
            x = conv_output.reshape(conv_output.shape[0], -1)
            return x

        return x

    def decode(self, x):

        logits = self.hidden_layer_1(x)
        logits = self.relu(logits)
        logits = self.hidden_layer_2(logits)
        logits = logits.squeeze(dim=1)
        logits = self.sigmoid(logits)
        return logits



