import torch
import torch.nn as nn
import numpy as np

class FCNN(nn.Module):
    def __init__(self, num_hidden=1, dropout_rate=0.3, input_length = 1, num_input_channels = 64):

        super().__init__()

        self.input_length = input_length
        self.num_input_channels = num_input_channels

        self.num_hidden = num_hidden
        units = np.round(np.linspace(1, self.input_length*self.num_input_channels, self.num_hidden+2)[::-1]).astype(int)
        self.fully_connected = torch.nn.ModuleList([torch.nn.Linear(units[i], units[i+1]) for i in range(len(units)-1)])
        self.activations = torch.nn.ModuleList([torch.nn.Tanh() for i in range(len(units)-2)])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_rate) for i in range(len(units)-2)])

    def __str__(self):
        return 'fcnn'

    def forward(self, x):
        for i, layer in enumerate(self.fully_connected[:-1]):
            x = layer(x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        
        x = self.fully_connected[-1](x)
        return x