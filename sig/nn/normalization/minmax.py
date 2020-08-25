import torch


def minmax(input):
    """
    Minmax normalization of input.

    Args:
        input (torch.Tensor): The input tensor.

    Return:
        The minmax normalized output tensor (torch.Tensor).
    """
    input_min = torch.min(input)
    input_max = torch.max(input)
    return (input - input_min) / (input_max - input_min)