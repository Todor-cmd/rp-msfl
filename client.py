import torch

from cifar10.cifar10_models import return_model
from cifar10.sgd import SGD


class Client:
    def __init__(self, client_idx, is_mal, args,  data_loader, criterion):
        self.client_idx = client_idx
        self.args = args
        self.data_loader = data_loader
        self.is_mal = is_mal
        self.model_type = args.arch
        self.fed_lr = args.fed_lr
        self.criterion = criterion

        self.fed_model, self.optimizer_fed = return_model(self.model_type,\
                                                          lr=args.fed_lr,\
                                                          momentum=0.9,\
                                                          parallel=args.parallel,\
                                                          cuda=args.cuda)
        
        self.optimizer_fed = SGD(self.fed_model.parameters(), lr=args.fed_lr)
        self.data_loader_iter = iter(self.data_loader)
        self.data_loader_i = 0
       


    def train(self):
        # If all batches of data used, reset the data loader
        if (self.data_loader_i == len(self.data_loader)):
            self.data_loader_i = 0
            self.data_loader_iter = iter(self.data_loader)

        # Get next batch of data
        inputs, targets = next(self.data_loader_iter)
        self.data_loader_i = self.data_loader_i + 1


        # Convert inputs and labels to PyTorch Variables
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # Forward pass
        outputs = self.fed_model(inputs.cuda())

        # Compute loss
        loss = self.criterion(outputs, targets.cuda())

        # Zero out gradients, perform backward pass, and collect gradients
        self.fed_model.zero_grad()
        loss.backward(retain_graph=True)

        # Move the gradients to CPU for further processing if needed
        param_grad = []
        for param in self.fed_model.parameters():
            grad_data = param.grad.data.cpu().view(-1)
            param_grad = grad_data if not len(param_grad) else torch.cat((param_grad, grad_data))

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
            param_ = param_.cuda()
            model_grads.append(param_)

        # Perform a step in the optimizer using the aggregated gradients
        self.optimizer_fed.step(model_grads)

    def update_learning_rate(self, epoch_num, schedule, gamma):
        if epoch_num in schedule:
            # Iterate over parameter groups in the optimizer
            for param_group in self.optimizer_fed.param_groups:
                # Update the learning rate of each parameter group using the specified decay factor (gamma)
                param_group['lr'] *= gamma

                # Print the updated learning rate
                print('New learning rate:', param_group['lr'])