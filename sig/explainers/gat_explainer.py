import torch
from sig.explainers.base_explainer import BaseExplainer


class GATExplainer(BaseExplainer):
    """
    Graph Attention Network (GAT) explainer.
    
    Args:
        model (torch.nn.Module): Trained GAT model for explanation.
        flow (str; optional): Message passing flow 
            {'source_to_target', 'target_to_source'}.
    """
    def __init__(
        self,
        model, 
        flow='source_to_target'
    ):
        super().__init__(model, flow)
    
    def explain_node(
        self, 
        node_index,
        x,
        edge_index,
        **kwargs
    ):
        """
        Explain the edges based on the computation subgraph of a node.
        
        Args:
            node_index (int): Index of the node to explain.
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            
        Return:
            node_feat_score (None): None since not applicable for GAT.
            edge_score (torch.float): Edge explanation score with shape [num_edges].
        """
        self.model.eval()
        num_all_edges = edge_index.shape[1]
        x, edge_index, hard_node_mask, hard_edge_mask = self.__subgraph__(
            node_index, 
            x, 
            edge_index
        )
        _, all_atts = self.model(
            x, 
            edge_index, 
            return_all_attentions=True,
            return_no_selfloop_attentions=True,
            **kwargs
        )
        node_feat_score = None
        edge_score = self.__edge_score__(
            num_all_edges, 
            hard_edge_mask,
            torch.mean(all_atts, dim=1)
        )
        return node_feat_score, edge_score
    
    def explain_graph(
        self,
        x,
        edge_index,
        batch,
        **kwargs
    ):
        """
        Explain the edges for a graph.

        Args:
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            batch (torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification.
        
        Return:
            node_feat_score (None): None since not applicable for GAT.
            edge_score (torch.float): Edge explanation score with shape [num_edges].
        """
        self.model.eval()
        _, all_atts = self.model(
            x, 
            edge_index, 
            return_all_attentions=True,
            return_no_selfloop_attentions=True,
            batch=batch,
            **kwargs
        )
        node_feat_score = None
        edge_score = torch.mean(all_atts, dim=1)
        return node_feat_score, edge_score