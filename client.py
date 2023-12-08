import torch

from cifar10.cifar10_models import return_model
from cifar10.sgd import SGD


def __init__(self, client_idx, is_mal, model_type, fed_lr):
    self.client_idx = client_idx
    self.is_mal = is_mal
    self.fed_model, _ = return_model(model_type, 0.1, 0.9, parallel=False)
    self.optimizer_fed = SGD(self.fed_model.parameters(), lr=fed_lr)


def train(self, inputs, labels):
    # Convert inputs and labels to PyTorch Variables
    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

    # Forward pass
    outputs = self.fed_model(inputs)

    # Compute loss
    loss = self.criterion(outputs, labels)

    # Zero out gradients, perform backward pass, and collect gradients
    self.fed_model.zero_grad()
    loss.backward(retain_graph=True)
    param_grad = []
    for param in self.fed_model.parameters():
        param_grad = param.grad.data.view(-1) if not len(param_grad) else torch.cat(
            (param_grad, param.grad.view(-1)))

    return param_grad


def update_model(self, agg_grads):
    # Initialize the starting index for aggregating gradients
    start_idx = 0

    # Zero out the gradients in the optimizer to avoid accumulation
    self.optimizer_fed.zero_grad()

    # List to store model gradients
    model_grads = []

    # Iterate over model parameters and get new gradients for parameters which exist in this model
    for i, param in enumerate(self.fed_model.parameters()):
        # Extract a slice of aggregated gradients corresponding to the current parameter
        param_ = agg_grads[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
        start_idx = start_idx + len(param.data.view(-1))
        param_ = param_  # .cuda()
        model_grads.append(param_)

    # Perform a step in the optimizer using the aggregated gradients
    self.optimizer_fed.step(model_grads)