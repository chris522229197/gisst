import torch


class AttentionProb(torch.nn.Module):
    """
    Edge attention mechanism for generating sigmoid probability using concatentation of 
    source and target node features.

    Args:
        input_size (int): Number of input node features.
        clamp_min (float): Clamping minimum for the output probability, for numerical
            stability.
        clamp_max (float): Clamping maximum for the output probability, for numerical 
            stability.
    """
    def __init__(
        self, 
        input_size, 
        clamp_min=0.00001,
        clamp_max=0.99999
    ):
        super(AttentionProb, self).__init__()
        self.input_size = input_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.att_weight = torch.nn.Parameter(
            torch.Tensor(input_size * 2)
        )
    
    def forward(
        self,
        x,
        edge_index
    ):
        """
        Forward pass.

        Args:
            x (tensor): Node feature tensor with shape [num_nodes, input_size].
            edge_index (torch.long): edges in COO format with shape [2, num_edges].

        Return:
            att (tensor): Edge attention probability with shape [num_edges].
        """
        att = torch.matmul(
            torch.cat(
                (
                    x[edge_index[0, :], :], # source node features
                    x[edge_index[1, :], :]  # target node features
                ), 
                dim=1
            ), 
            self.att_weight
        )
        att = torch.sigmoid(att)
        att = torch.clamp(att, self.clamp_min, self.clamp_max)
        return att


class RelationalAttentionProb(torch.nn.Module):
    """
    Edge attention mechanism for multi-relational graph.

    Args:
        input_size (int): Number of input node features.
        num_relations (int): Number of edge types in the graph.
        clamp_min (float): Clamping minimum for the output probability, for numerical
            stability.
        clamp_max (float): Clamping maximum for the output probability, for numerical 
            stability.
    
    """
    def __init__(
        self, 
        input_size, 
        num_relations, 
        clamp_min=0.00001,
        clamp_max=0.99999
    ):
        super(RelationalAttentionProb, self).__init__()
        self.input_size = input_size
        self.num_relations = num_relations
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.att_weight = torch.nn.Parameter(
            torch.Tensor(input_size * 2, num_relations)
        )

    def forward(
        self, 
        x, 
        edge_index, 
        edge_type
    ):
        """
        Forward pass.
        
        Args:
            x (tensor): Node feature tensor shape shape [num_nodes, input_size].
            edge_index (torch.long): edges in COO format with shape [2, num_edges].
            edge_type (torch.long): edge type index with shape [num_edges].
        
        Return:
            att (tensor): Edge attention probability with shape [num_edges].
        """
        att = torch.matmul(
            torch.cat(
                (
                    x[edge_index[0, :], :], # source node features
                    x[edge_index[1, :], :]  # target node features
                ), 
                dim=1
            ), 
            self.att_weight
        )
        att = torch.sigmoid(
            att.gather(1, edge_type.view(-1, 1)).squeeze()
        )
        att = torch.clamp(att, self.clamp_min, self.clamp_max)
        return att