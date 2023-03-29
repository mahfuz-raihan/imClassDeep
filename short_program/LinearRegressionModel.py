# This program is a demo code in pytorch
# if there is any confuciotion let me know. 

# improt the basic libraries
import torch

class LinearRegressionModel(nn.Module):
    def __init__(self): # initial the fucntion for
        super().__init__()
        # initialize weight
        self.weight = nn.Parameter(torch.randn(1, # it may generate random weight (this would be adjested later during training)
                                               dtype=torch.float, # data type
                                               required_grad=True)) # we can update this value with gradient descent
        # initialize bias
        self.bias = nn.Parameter(torch.randn(1, # it may generate random bias (this would be adjested later during training) 
                                             dtype=torch.float, # data type
                                             required_grad=True)) # we can update this value with gradient descent
    
    # Forward defines the computations in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input data (training/testing features)
        return self.weight*x + self.bias # linear regression formula (y = mx + b)
