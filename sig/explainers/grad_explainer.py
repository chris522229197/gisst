import torch
from torch.autograd import Variable
from sig.explainers.base_explainer import BaseExplainer


class GradExplainer(BaseExplainer):
    """
    Gradient saliency explainer.
    
    Args:
        model (torch.nn.Module): Trained model for explanation.
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
        y,
        loss_fn,
        edge_weight=None,
        take_abs=False,
        take_mag=False,
        pred_for_grad=False,
        **kwargs
    ):
        """
        Explain the edges based on the computation subgraph of a node.
        
        Args:
            node_index (int): Index of the node to explain.
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            y (torch.long): Ground truth label with shape [num_nodes].
            loss_fn (function): Loss function for the gradient computation. 
            edge_weight (torch.float or None): Edge weights with shape [num_edges].
            take_abs (boolean): Whether to take the absolute value of the gradient.
            take_mag (boolean): Whether to take the magnitude of the input node features 
                for the gradient.
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
        y = y[hard_node_mask]
        if take_mag:
            x = torch.abs(x)
        x = Variable(x, requires_grad=True)
        edge_score = Variable(
            torch.ones(edge_index.shape[1]).to(x.device),
            requires_grad=True
        )
        if edge_weight is not None:
            edge_weight = edge_weight[hard_edge_mask]
        else:
            edge_weight = torch.ones(edge_index.shape[1]).to(edge_score.device)
        edge_weight = edge_weight * edge_score
        out = self.model(
            x,
            edge_index,
            edge_weight=edge_weight,
            **kwargs
        )
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
        node_feat_score = torch.mean(x.grad, dim=0)
        edge_score = self.__edge_score__(
            num_all_edges, 
            hard_edge_mask,
            edge_score.grad
        )
        if take_abs:
            node_feat_score = torch.abs(node_feat_score)
            edge_score = torch.abs(edge_score)
        else:
            node_feat_score = -node_feat_score
            edge_score = -edge_score
        return node_feat_score, edge_score

    def explain_graph(
        self,
        x,
        edge_index,
        y,
        loss_fn,
        batch,
        edge_weight=None,
        take_abs=False,
        take_mag=False,
        pred_for_grad=False,
        **kwargs
    ):
        """
        Explain the node features and edges for a graph.

        Args:
            x (torch.float): Node feature matrix with shape [num_nodes, num_node_feats].
            edge_index (torch.long): Edge COO with shape [2, num_edges].
            y (torch.long): Ground truth label for the graph.
            loss_fn (function): Loss function for the gradient computation. 
            batch (torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification.
            edge_weight (torch.float or None): Edge weights with shape [num_edges].
            take_abs (boolean): Whether to take the absolute value of the gradient.
            take_mag (boolean): Whether to take the magnitude of the input node features 
                for the gradient.
            pred_for_grad (boolean): Whether to use the predicted label for the loss
                gradient.
        
        Return:
            node_feat_score (torch.float): Node feature explanation score with shape 
                [num_node_feats].
            edge_score (torch.float): Edge explanation score with shape [num_edges].
        """
        self.model.eval()
        if take_mag:
            x = torch.abs(x)
        x = Variable(x, requires_grad=True)
        edge_score = Variable(
            torch.ones(edge_index.shape[1]).to(x.device),
            requires_grad=True
        )
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(edge_score.device)
        edge_weight = edge_weight * edge_score
        out = self.model(
            x,
            edge_index,
            edge_weight=edge_weight,
            batch=batch,
            **kwargs
        )
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
        node_feat_score = torch.mean(x.grad, dim=0)
        edge_score = edge_score.grad
        if take_abs:
            node_feat_score = torch.abs(node_feat_score)
            edge_score = torch.abs(edge_score)
        else:
            node_feat_score = -node_feat_score
            edge_score = -edge_score
        return node_feat_score, edge_score