import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from sig.nn.models.base_model import BaseClassifier
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(BaseClassifier):
    """
    Graph Attention Network.

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
        **kwargs: addtional keyword arguments for GATConv
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
                GATConv(
                    self.conv_input_sizes[i], 
                    self.conv_output_sizes[i], 
                    **kwargs
                )
            )
            self.dropouts.append(
                Dropout(self.dropout_probs[i])
            )
    
    def forward(
        self, 
        x, 
        edge_index,
        return_all_attentions=False,
        return_no_selfloop_attentions=True,
        batch=None
    ):
        """
        Forward pass.

        Args:
            x (torch.float): Node feature tensor with shape [num_nodes, num_node_feat].
            edge_index (torch.long): Edges in COO format with shape [2, num_edges].
            return_all_attentions (boolean): Whether to return attention weights across
                all the layers.
            return_no_selfloop_attentions (boolean): Whether to remove the self-loop 
                attention weights for returning all_atts.
            batch (None or torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification. None for node classification.

        Return:
            x (torch.float): Final output of the network with shape 
                [num_nodes, output_size].
            all_atts (torch.float): Edge attention weights for all the layers with shape
                [num_edges, num_layers] if return_no_selfloop_attentions is True, else with 
                shape [num_edges + num_nodes, num_layers].
        """
        all_atts = []
        for i in range(self.num_conv_layers):
            x, att = self.convs[i](
                x, 
                edge_index, 
                return_attention_weights=True
            )
            if self.classify_graph or i < self.num_conv_layers - 1:
                # no activation for output conv layer in node classifier
                x = self.activation(x)
            x = self.dropouts[i](x)
            if return_no_selfloop_attentions:
                att = att[:-x.shape[0]]
            all_atts.append(att)
        all_atts = torch.cat(all_atts, dim=1)
        if self.classify_graph:
            x = global_mean_pool(x, batch)
            x = self.activation(self.lin1(x))
            x = self.lin_dropout(x)
            x = self.lin2(x)
        if return_all_attentions:
            return x, all_atts
        else:
            return x