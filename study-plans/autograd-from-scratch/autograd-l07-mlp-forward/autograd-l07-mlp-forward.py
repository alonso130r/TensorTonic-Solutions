import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    h_i = inputs 
    h = []
    for i in range(len(weights)):
        h_i = torch.nn.functional.tanh(weights[i] @ h_i + biases[i])
        h.append(h_i)
        
    return (h_i, h)
