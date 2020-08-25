import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

class BaseExplainer(torch.nn.Module):
    """
    Base class for all the Explainers.
    
    Args:
        model (torch.nn.Module): Trained model for explanation.
        flow (str): Message passing flow {'source_to_target', 'target_to_source'}.
    """
    def __init__(
        self,
        model, 
        flow='source_to_target'
    ):
        super(BaseExplainer, self).__init__()
        self.model = model
        self.flow = flow
    
    def __num_hops__(self):
        """
        Find the number of hops in the model computation.
        
        Return:
            num_hops (int): Number of hops.
        """
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops
    
    def __subgraph__(
        self, 
        node_index, 
        x, 
        edge_index
    ):
        """
        Find the computation subgraph for a node.
        
        Args:
            node_index (int): Index for the central node.
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].

        Return:
            x (torch.float): Node feature matrix with shape 
                [subgraph_num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, subgraph_num_edges].
            node_mask (torch.bool): Subgraph node mask with shape [num_nodes].
            edge_mask (torch.bool): Subgraph edge mask with shape [num_edges].
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        node_mask, edge_index, edge_mask = k_hop_subgraph(
            node_index, 
            self.__num_hops__(), 
            edge_index, 
            relabel_nodes=True,
            num_nodes=num_nodes, 
            flow=self.flow
        )
        x = x[node_mask]
        return x, edge_index, node_mask, edge_mask
    
    def __edge_score__(
        self,
        num_all_edges,
        subgraph_edge_mask,
        subgraph_edge_score
    ):
        """
        Get the edge explanation score for a subgraph.
        
        Args:
            num_all_edges (int): Number of edges in the entire graph.
            subgraph_edge_mask (torch.bool): Whether each edge in the entire graph is in 
                the subgraph.
            subgraph_edge_score (torch.float): Explanation score for edges in the 
                subgraph.
        
        Return:
            all_edge_score (torch.float): Explanation score for all edges in the 
                entire graph with shape [num_all_edges]. Edges not in the subgraph 
                have zero scores.
        """
        all_edge_score = torch.zeros(num_all_edges).to(subgraph_edge_mask.device)
        all_edge_score[subgraph_edge_mask] = subgraph_edge_score
        return all_edge_score