import torch
from torch.autograd import Variable
from sig.explainers.base_explainer import BaseExplainer
from sig.nn.normalization.minmax import minmax


class SIGExplainer(BaseExplainer):
    """
    Sparse Interpretable Graph Neural Network (SIG) explainer.
    
    Args:
        model (torch.nn.Module): Trained SIG model for explanation.
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
        use_grad=False,
        y=None,
        loss_fn=None,
        take_abs=False,
        pred_for_grad=False,
        **kwargs
    ):
        """
        Explain the edges based on the computation subgraph of a node.
        
        Args:
            node_index (int): Index of the node to explain.
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            use_grad (boolean): Whether to adjust for node-specific gradient.
            y (torch.long): Ground truth label with shape [num_nodes].
            loss_fn (function): Loss function for the gradient computation.
            take_abs (boolean): Whether to take absolute value for the gradient output.
            pred_for_grad (boolean): Whether to use the predicted label for the loss
                gradient.

        Return:
            node_feat_score (torch.float): Node feature explanation score with shape 
                [num_node_feats].
            edge_score (torch.float): Edge explanation score with shape [num_edges].
        """
        self.model.eval()
        num_all_edges = edge_index.shape[1]
        x, edge_index, hard_node_mask, hard_edge_mask = self.__subgraph__(
            node_index, 
            x, 
            edge_index
        )
        if y is not None:
            y = y[hard_node_mask]
        out, x_prob, edge_prob = self.model(
            x, 
            edge_index,
            return_probs=True,
            **kwargs
        )
        node_feat_score = x_prob
        edge_score = edge_prob
        if use_grad:
            out_dim = out.shape[1]
            if pred_for_grad:
                label = out.max(1)[1][0].view(1)
            else:
                label = y[0].view(1)
            loss = loss_fn(
                out[0, :].view(1, out_dim), # node_index relabelled to zero
                label
            )
            loss.backward()
            if take_abs:
                node_feat_score = torch.abs(node_feat_score)
                edge_score = torch.abs(edge_score)
            else:
                node_feat_score = -node_feat_score.grad
                edge_score = -edge_score.grad
        edge_score = self.__edge_score__(
            num_all_edges, 
            hard_edge_mask,
            edge_score
        )
        return node_feat_score, edge_score

    def explain_graph(
        self,
        x,
        edge_index,
        batch,
        use_grad=False,
        y=None,
        loss_fn=None,
        take_abs=False,
        pred_for_grad=False,
        **kwargs
    ):
        """
        Explain the node features and edges for a graph.
        
        Args:
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            batch (torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification.
            use_grad (boolean): Whether to adjust for node-specific gradient.
            y (torch.long): Ground truth label with shape [num_nodes].
            loss_fn (function): Loss function for the gradient computation.
            take_abs (boolean): Whether to take absolute value for the gradient output.
            pred_for_grad (boolean): Whether to use the predicted label for the loss
                gradient.

        Return:
            node_feat_score (torch.float): Node feature explanation score with shape 
                [num_node_feats].
            edge_score (torch.float): Edge explanation score with shape [num_edges].
        """
        self.model.eval()
        out, x_prob, edge_prob = self.model(
            x, 
            edge_index,
            return_probs=True,
            batch=batch,
            **kwargs
        )
        node_feat_score = x_prob
        edge_score = edge_prob
        if use_grad:
            out_dim = out.shape[1]
            if pred_for_grad:
                label = out.max(1)[1][0].view(1)
            else:
                label = y[0].view(1)
            loss = loss_fn(
                out.view(1, out_dim),
                label
            )
            loss.backward()
            if take_abs:
                node_feat_score = torch.abs(node_feat_score)
                edge_score = torch.abs(edge_score)
            else:
                node_feat_score = -node_feat_score.grad
                edge_score = -edge_score.grad
        return node_feat_score, edge_score