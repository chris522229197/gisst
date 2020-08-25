import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from sig.nn.models.base_model import BaseClassifier
from torch_geometric.nn import GCNConv, global_mean_pool
from sig.nn.modules.attention import AttentionProb
from sig.nn.modules.mask import ProbMask


class SIGCN(BaseClassifier):
    """
    Sparse Interpretable Graph Convolution Network.

    Args:
        input_size (int): Number of input node features.
        output_size (int): Number of output node features.
        hidden_conv_sizes (tuple of int): Output sizes for hidden convolution layers.
        hidden_dropout_probs (tuple of float): Dropout probabilities after the hidden
            convolution layers.
        activation (torch.nn.functional): Non-linear activation function after the hidden
            convolution layers.  
        classify_graph (boolean): Whether the model is a graph classifier. Default False
            for node classifier.
        lin_dropout_prob (None or float): Dropout probability after the hidden linear 
            layer for graph classifier. None for node classifier.
        **kwargs: Additoinal kwargs for GCNConv.
    """
    def __init__(
        self, 
        input_size,
        output_size,
        hidden_conv_sizes, 
        hidden_dropout_probs, 
        activation=F.relu,
        classify_graph=False,
        lin_dropout_prob=None,
        **kwargs
    ):
        super().__init__(
            input_size,
            output_size,
            hidden_conv_sizes, 
            hidden_dropout_probs, 
            activation,
            classify_graph,
            lin_dropout_prob
        )
        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.convs.append(
                GCNConv(
                    self.conv_input_sizes[i], 
                    self.conv_output_sizes[i],
                    **kwargs
                )
            )
            self.dropouts.append(
                Dropout(self.dropout_probs[i])
            )
        self.prob_mask = ProbMask(self.input_size)
        self.attention_prob = AttentionProb(self.input_size)
    
    def forward(
        self, 
        x, 
        edge_index,
        edge_weight=None,
        return_probs=False,
        batch=None
    ):
        """
        Forward pass.

        Args:
            x (torch.float): Node feature tensor with shape [num_nodes, num_node_feat].
            edge_index (torch.long): Edges in COO format with shape [2, num_edges].
            edge_weight (torch.float): Weight for each edge with shape [num_edges].
            return_probs (boolean): Whether to return x_prob and edge_prob.
            batch (None or torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification. None for node classification.

        Return:
            x (torch.float): Final output of the network with shape 
                [num_nodes, output_size].
            x_prob (torch.float): Node feature probability with shape [input_size].
            edge_prob (torch.float): Edge probability with shape [num_edges].
        """
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()
        x = x * x_prob

        edge_prob = self.attention_prob(x, edge_index)
        edge_prob.requires_grad_()
        edge_prob.retain_grad()
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(edge_prob.device)
        edge_weight = edge_weight * edge_prob
        
        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if self.classify_graph or i < self.num_conv_layers - 1:
                # no activation for output conv layer in node classifier
                x = self.activation(x)
            x = self.dropouts[i](x)
        if self.classify_graph:
            x = global_mean_pool(x, batch)
            x = self.activation(self.lin1(x))
            x = self.lin_dropout(x)
            x = self.lin2(x)
        if return_probs:
            return x, x_prob, edge_prob
        else:
            return x