# Loss functions for parameter regularization

import torch


def reg_binary_entropy_loss(out):
    """
    Mean binary entropy loss to drive values toward 0 and 1.

    Args:
        out (torch.float): Values for the binary entropy loss. The values have to be 
        within (0, 1).

    Return:
        torch.float for the loss value.
    """
    return torch.mean(-out * torch.log(out) - (1 - out) * torch.log(1 - out))

def reg_l1_loss(out):
    """
    Mean L1 loss to drive values toward 0.

    Args:
        out (torch.float): Values for the L1 loss.

    Return:
        torch.float for the loss value.
    """
    return torch.mean(torch.abs(out))

def reg_sig_loss(
    x_prob, 
    edge_prob,
    coeffs={
        'x_l1': 1.0,
        'x_ent': 1.0,
        'edge_l1': 1.0,
        'edge_ent': 1.0
    }
):
    """
    Regularization losses for Sparse Interpretable GNN, multiplied by the regularization
        coefficients.

    Args:
        x_prob (torch.float): Node feature probability values between (0, 1).
        edge_prob (torch.float): Edge probability values between (0, 1).
        coeffs (dict): Regualization coefficients.
    
    Return:
        loss_x_l1, loss_x_ent, loss_edge_l1, loss_edge_ent (torch.float): Values for 
            node feature probability L1, node feature probabilitiy entropy, edge
            probability L1, and edge probability entropy losses, respectively.
    """
    loss_x_l1 = coeffs['x_l1'] * reg_l1_loss(x_prob)
    loss_x_ent = coeffs['x_ent'] * reg_binary_entropy_loss(x_prob)
    loss_edge_l1 = coeffs['edge_l1'] * reg_l1_loss(edge_prob)
    loss_edge_ent = coeffs['edge_ent'] * reg_binary_entropy_loss(edge_prob)
    return loss_x_l1, loss_x_ent, loss_edge_l1, loss_edge_ent