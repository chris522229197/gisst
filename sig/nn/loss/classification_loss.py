# Classification loss functions

import torch.nn.functional as F


def cross_entropy_loss(out, y):
    """
    Mean cross entropy loss.

    Args:
        out (torch.float): Output from a graph neural network with shape 
            [num_nodes, num_classes].
        y (torch.long): Ground truth label with shape [num_nodes].
    
    Return:
        torch.float for the loss value.
    """
    return F.nll_loss(F.log_softmax(out, dim=1), y)