import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self, ndim):
        super().__init__()
        
        self.l1 = torch.nn.Linear(ndim, ndim)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(ndim, ndim)
        self.layers = torch.nn.Sequential(
            self.l1, self.relu, self.l
        )
    
    def forward(self, x):
        return self.layers(x)
    
