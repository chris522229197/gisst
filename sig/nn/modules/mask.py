import torch


class ProbMask(torch.nn.Module):
    """
    Probability mask generator.

    Args:
        shape (tuple of int): Shape of the probability mask.
        clamp_min (float): Clamping minimum for probability, for numerical stability.
        clamp_max (float): Clamping maximum for probability, for numerical stability. 
    """
    def __init__(
        self, 
        shape, 
        clamp_min=0.00001,
        clamp_max=0.99999
    ):
        super(ProbMask, self).__init__()
        self.shape = shape
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mask_weight = torch.nn.Parameter(torch.Tensor(shape))

    def forward(self):
        return torch.clamp(
            torch.sigmoid(self.mask_weight), 
            self.clamp_min, 
            self.clamp_max
        )