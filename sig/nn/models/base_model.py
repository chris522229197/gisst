import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import global_mean_pool

class BaseClassifier(torch.nn.Module):
    """
    Base class for all the classifier models.

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
    """
    def __init__(
        self, 
        input_size,
        output_size,
        hidden_conv_sizes, 
        hidden_dropout_probs, 
        activation=F.relu,
        classify_graph=False,
        lin_dropout_prob=None
    ):
        super(BaseClassifier, self).__init__()
        assert len(hidden_conv_sizes) == len(hidden_dropout_probs), \
            'lengths of hidden_conv_sizes and hidden_dropout_probs are not equal'
        
        conv_input_sizes = (input_size, ) + hidden_conv_sizes
        conv_output_sizes = hidden_conv_sizes + (output_size, )
        dropout_probs = hidden_dropout_probs + (0, )
        if classify_graph: # remove output layers
            conv_input_sizes = conv_input_sizes[:-1]
            conv_output_sizes = conv_output_sizes[:-1]
            dropout_probs = dropout_probs[:-1]
        num_conv_layers = len(conv_input_sizes)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_conv_sizes = hidden_conv_sizes
        self.hidden_dropout_probs = hidden_dropout_probs
        self.activation = activation
        self.classify_graph = classify_graph
        self.lin_dropout_prob = lin_dropout_prob
        self.lin1 = None
        self.lin2 = None
        self.lin_dropout = None

        self.conv_input_sizes = conv_input_sizes
        self.conv_output_sizes = conv_output_sizes
        self.dropout_probs = dropout_probs
        self.num_conv_layers = num_conv_layers

        if classify_graph: # replace output layers
            final_conv_output_size = conv_output_sizes[-1]
            self.lin1 = Linear(final_conv_output_size, final_conv_output_size)
            self.lin2 = Linear(final_conv_output_size, output_size)
            self.lin_dropout = Dropout(lin_dropout_prob)