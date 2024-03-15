import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)
        
    def forward(self, node_feats, adj_matrix):
        
        """
        node_feats = Tensor with node features, shape (batch_size, num_nodes, c_in)
        adj_matrix = Batch adjacency matrix of the graph, shape (batch_size, num_nodes, num_nodes)
        """

        #num_neighbors = number of incoming edges
        num_neighbors = adj_matrix.sum(dim=-1,keepdims=True)
        node_feats = self.projection(node_feats)
        #bmm = batch matrix-matrix product
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbors
        return node_feats
    
class GCN(nn.Module):
    def __init__(self, c_in, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.gcn_1 = GCNLayer(c_in, hidden_dim)
        self.gcn_2 = GCNLayer(hidden_dim,num_classes)
        
    def forward(self, x, adj):
        x = F.relu(self.gcn_1(x, adj))
        x = self.gcn_2(x,adj)
        #apply softmax for classification task
        return F.log_softmax(x, dim=1)