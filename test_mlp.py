import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.neurons = neurons
        self.layers = torch.nn.ModuleList()
        for i in range(len(neurons) - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            if i != len(neurons) - 2:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


mlp = MLP([784,100,100,10])
for i in mlp.parameters():
    print(i.shape)