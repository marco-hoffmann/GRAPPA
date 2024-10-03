"""This module contains the class definition for all graph neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.nn import BatchNorm1d
from torch_geometric.nn import (GINEConv, GATv2Conv, NNConv, FiLMConv,TransformerConv,
                                global_add_pool, global_max_pool, global_mean_pool, Set2Set)
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
        

def get_pooling_function(pool, out_dim, pooling_heads = 1):
    """Returns the pooling function specified by the string pool"""
    if pool == 'sum':
        return out_dim, global_add_pool
    elif pool == 'mean':
        return out_dim, global_mean_pool
    elif pool == 'max':
        return out_dim, global_max_pool
    elif pool == 'set2set':
        return 2*out_dim, Set2Set(out_dim, processing_steps=4, num_layers=2)
    elif pool == 'attention':
        return out_dim, MultiHeadAttentionPooling(out_dim, key_dim=32, num_pooling_heads=pooling_heads)
    else:
        print('Unknown pooling function!')
        exit()

def get_jk_out_dim(jk_mode, conv_dim, num_layers):
    """Returns the output dimension of the JK layer"""
    if jk_mode == 'last':
        return conv_dim
    elif jk_mode == "mean":
        return conv_dim
    elif jk_mode == "concat":
        return num_layers * conv_dim
    else:
        raise ValueError("Unknown JK mode")

def forward_jk(x, edge_index, edge_attr, convs, jk_mode):
    """Forward pass for the GNN with JK pooling"""
    x_layers = []
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    for conv in convs:
        conv.to(device)
        x = conv(x, edge_index, edge_attr)
        x = F.elu(x)
        x_layers.append(x)
    
    if jk_mode == 'last':
        pass
    elif jk_mode == "mean":
        x = torch.mean(torch.stack(x_layers), dim=0)
    elif jk_mode == "concat":
        x = torch.cat(x_layers, dim=1)
    else:
        raise ValueError("Unknown JK mode")

    return x

class GNN_GAT(nn.Module):
    """Implementation of GAT"""

    def __init__(self, node_dim, edge_dim, conv_dim, heads=5, dropout=0.1, num_layers=3, jk_mode = "last"):
        """Initializes GAT model. Takes in node and edge dimensions conv_dim is the hidden dimension
        of the GAT convolutional layers. Heads is the number of attention heads to use in the GAT"""
        super().__init__()

        self.num_layers = num_layers
        self.jk_mode = jk_mode

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(conv_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

    def forward(self, x, edge_index, edge_attr):

        x = forward_jk(x, edge_index, edge_attr, self.convs, self.jk_mode)

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

        num_params = antoine_parameters.size(1)
        
        # Ensure there are at least 3 parameters
        if num_params < 3:
            raise ValueError("At least three Antoine parameters (A, B, C) are required.")
        
        A = antoine_parameters[:, 0]
        B = antoine_parameters[:, 1]
        C = antoine_parameters[:, 2]
        result = A - B / (C + temperature * 1000 + 1e-8)  # Basic Antoine equation

        # Extend Antoine equation based on the number of parameters
        if num_params > 3:
            D = antoine_parameters[:, 3]
            result += D * temperature * 1000  # Add D term if present
        
        if num_params > 4:
            E = antoine_parameters[:, 4]
            result += E * (temperature * 1000) ** 2  # Add E term if present
        
        if num_params > 5:
            F = antoine_parameters[:, 5]
            result += F * torch.log(temperature * 1000)  # Add F term if present
        
        # Add more terms if needed, following a similar pattern.
        
        return result # Antoine equation (1e-8 to avoid division by zero)
    
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
                 num_gnn_layers=3, num_antoine_params = 3, gnn_heads = 5, pooling_heads = 1, Tdep=False, jk_mode='last'):
        super().__init__()
    
        self.Tdep = Tdep

        gnn_choice = {
            'GAT': GNN_GAT,
        }

        # GNN that graphs are passed to
        self.gnn = gnn_choice[arch](node_dim, edge_dim, conv_dim, num_layers=num_gnn_layers, dropout=dropout, heads=gnn_heads, jk_mode=jk_mode)

        self.out_dim, self.pooling_function = get_pooling_function(pool, conv_dim, pooling_heads) 
        # change out_dim due to Jumping Knowledge layer
        self.out_dim = get_jk_out_dim(jk_mode, self.out_dim, num_gnn_layers)            

        # Head takes the final node embeddings and temperature and gives the Antoine parameters
        # Two dimensions added for number of H-Donors and H-Acceptors
        if Tdep:
            self.head = Head(self.out_dim + 3, hidden_dim, num_hidden_layers, num_antoine_params)
        else:
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

        # Append temperature to the final graph embedding
        if self.Tdep:
            graph_out = torch.cat((graph_out, temperature.unsqueeze(1)), dim=1)

        antoine_parameters = self.head(graph_out)
        
        antoine_parameters = self.parameter_scaler(antoine_parameters)

        return antoine_parameters

    def forward(self, x, temperature, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        
        # Produces graph with generated embeddings
        gnn_out = self.gnn(x, edge_index, edge_attr)
        # Aggregates node embeddings to single node/vector.
        graph_out = self.pooling_function(gnn_out, batch)
        
        # Append number of H-Donors and H-Acceptors to the final graph embedding
        graph_out = torch.cat((graph_out, numHDonors.unsqueeze(1), numHAcceptors.unsqueeze(1)), dim=1)

        # Append temperature to the final graph embedding
        if self.Tdep:
            graph_out = torch.cat((graph_out, temperature.unsqueeze(1)), dim=1)

        antoine_parameters = self.head(graph_out)

        antoine_parameters = self.parameter_scaler(antoine_parameters)

        logp = self.antoine_layer(antoine_parameters, temperature)

        return logp