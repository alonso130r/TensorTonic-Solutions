import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    a = weight_matrix @ inputs + biases
    if nonlinear:
        return torch.nn.functional.tanh(a)
    return a
