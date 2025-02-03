"""This module contains the class definition for all graph neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter

class GraphAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim):
        super(GraphAttentionPooling, self).__init__()

        self.n_features = n_features

        self.query_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.key_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.value_weight = nn.Parameter(torch.Tensor(n_features, n_features))

        nn.init.xavier_uniform_(self.query_weight)
        nn.init.xavier_uniform_(self.key_weight)
        nn.init.xavier_uniform_(self.value_weight)

    def get_attention_scores(self, node_out, batch):
        _ , n_features = node_out.size()
        device = node_out.device

        Q = torch.matmul(node_out, self.query_weight) 
        K = torch.matmul(node_out, self.key_weight)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (n_features ** 0.5)

        # First, create a mask that identifies all pairs of nodes within the same graph
        # This will be used to filter out interactions between nodes of different graphs
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).to(device)

        # Set attention scores for nodes in different graphs to -inf
        attention_scores[~mask] = float('-inf')

        # Apply softmax to the attention scores
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores

    def forward(self, node_out, batch):
        _, n_features = node_out.size()
        n_graphs = batch.max().item() + 1
        device = node_out.device

        V = torch.matmul(node_out, self.value_weight)

        attention_scores = self.get_attention_scores(node_out, batch)

        # Apply attention scores to the value matrix
        context_matrix = torch.matmul(attention_scores, V)

        # Sum the context matrix for each graph
        return scatter(context_matrix.view(-1, n_features), batch, dim=0, dim_size=n_graphs, reduce='sum')
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim, num_pooling_heads):
        super(MultiHeadAttentionPooling, self).__init__()

        self.n_features = n_features
        self.num_heads = num_pooling_heads

        self.heads = nn.ModuleList([GraphAttentionPooling(n_features, key_dim) for _ in range(self.num_heads)])

    def forward(self, node_out, batch):
        # Apply each head to the node embeddings
        head_outputs = [head(node_out, batch) for head in self.heads]

        # Get the mean of the head outputs
        return torch.mean(torch.stack(head_outputs), dim=0)

class GNN_GAT(nn.Module):
    """Implementation of GAT"""

    def __init__(self, node_dim, edge_dim, conv_dim, heads=5, dropout=0.1, num_layers=3):
        """Initializes GAT model. Takes in node and edge dimensions conv_dim is the hidden dimension
        of the GAT convolutional layers. Heads is the number of attention heads to use in the GAT"""
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(conv_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

    def forward(self, x, edge_index, edge_attr):

        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        for conv in self.convs:
            conv.to(device)
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)

        return x

class ScaleOutput(nn.Module):
    """Custom activation function that uses the sigmoid activation function and then scales the ouputs into a given range.
    This is useful for scaling Antoine parameters to ensure they are within a certain range.
    Takes in the ranges for all parameters and scales the output accordingly."""
    def __init__(self, ranges):
        super(ScaleOutput, self).__init__()
        self.ranges = torch.tensor(ranges, device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
    def forward(self, x):
        # Perform operations without in-place modifications
        scaled_output = torch.sigmoid(x)  # Apply sigmoid to the output
        scaled_output = scaled_output * (self.ranges[:, 1] - self.ranges[:, 0]) + self.ranges[:, 0]  # Scale the output to the desired range
        return scaled_output

class AntoineLayer(nn.Module):
    def __init__(self):
        super(AntoineLayer, self).__init__()
    def forward(selfTdep, antoine_parameters, temperature):
        """Returns the vapor pressure given the Antoine parameters and temperature"""

        A = antoine_parameters[:, 0]
        B = antoine_parameters[:, 1]
        C = antoine_parameters[:, 2]
        result = A - B / (C + temperature + 1e-8)  # Basic Antoine equation (1e-8 to avoid division by zero)
        
        return result # Antoine equation 
    
class Head(nn.Module):
    """Prediction Head that is added to the end of the GNN. Takes in the pooled node embeddings
    and yields the Antoine parameters"""

    def __init__(self, input_dim, hidden_dim, num_hidden_layers, out_dim):
        super().__init__()

        # Input Layer
        layers = []
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        layers.append(nn.ELU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())

        # Output layer
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))

        # Combine all layers
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class GRAPPA(nn.Module):
    def __init__(self, node_dim, edge_dim, arch, 
                 conv_dim, hidden_dim, num_hidden_layers, 
                 dropout, pool, 
                 num_gnn_layers=3, num_antoine_params = 3, gnn_heads = 5, pooling_heads = 1):
        super().__init__()

        # GNN that graphs are passed to
        self.gnn = GNN_GAT(node_dim, edge_dim, conv_dim, num_layers=num_gnn_layers, dropout=dropout, heads=gnn_heads)

        self.out_dim = conv_dim

        self.pooling_function = MultiHeadAttentionPooling(self.out_dim, key_dim=32, num_pooling_heads=pooling_heads) 

        # Head takes the final node embeddings and temperature and gives the Antoine parameters
        # Two dimensions added for number of H-Donors and H-Acceptors
        self.head = Head(self.out_dim +2, hidden_dim, num_hidden_layers, num_antoine_params)

        # Scales parameters to viable ranges to prevent extrem values
        self.parameter_scaler = ScaleOutput([[5.0,20.0], [1500.0, 6000.0], [-300.0, 0.0]])

        # Antoine layer takes the Antoine parameters and temperature and gives the pressure
        self.antoine_layer = AntoineLayer()

    def get_embedding(self, x, edge_index, edge_attr, batch):
        '''Returns the graph embedding for the given graph. This is the output of the GNN and pooling function.'''

        gnn_out = self.gnn(x, edge_index, edge_attr)
        graph_out = self.pooling_function(gnn_out, batch)
        return graph_out

    def get_antoine_parameters(self, x, temperature, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        '''Returns the Antoine parameters for the given graph. This is the output of the GNN, pooling function, and head.
        Antoine Parameters are scaled to obtain output in log(p/kPa) = A - B/(C+T) where T is in Kelvin.'''
        
        # Produces graph with generated embeddings
        gnn_out = self.gnn(x, edge_index, edge_attr)

        # Aggregates node embeddings to single node/vector.
        graph_out = self.pooling_function(gnn_out, batch)

        # Append number of H-Donors and H-Acceptors to the final graph embedding
        graph_out = torch.cat((graph_out, numHDonors.unsqueeze(1), numHAcceptors.unsqueeze(1)), dim=1)

        antoine_parameters = self.head(graph_out)
        
        antoine_parameters = self.parameter_scaler(antoine_parameters)

        return antoine_parameters

    def forward(self, x, temperature, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        
        antoine_parameters = self.get_antoine_parameters(x, temperature, edge_index, edge_attr, numHDonors, numHAcceptors, batch)

        logp = self.antoine_layer(antoine_parameters, temperature)

        return logp