import torch


class EmbeddingLookup(torch.nn.Module):
    """
    Generator for node embedding lookup tensor.
    
    Args:
        num_embedded_nodes (int): Number of nodes requiring embedding.
        embedding_size (int): Embedding dimension.
    """
    def __init__(
        self, 
        num_embedded_nodes, 
        embedding_size
    ):
        super(EmbeddingLookup, self).__init__()
        self.num_embedded_nodes = num_embedded_nodes
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Parameter(
            torch.Tensor(num_embedded_nodes, embedding_size)
        )

    def forward(
        self, 
        num_nodes,
        embedded_node_index
    ):
        """
        Generate the embedding lookup tensor for all the nodes.

        Args:
            num_nodes (int): Total number of nodes in the graph.
            embedded_node_index (torch.long): Indicies for the embedded nodes.

        Return:
            lookup (torch.float): Embedding lookup tensor with shape 
                [num_nodes, embedding_size]. Nodes without embedding have all-zero 
                entries.
        """
        lookup = torch.zeros(num_nodes, self.embedding_size)
        lookup[embedded_node_index, :] = self.embedding
        return lookup