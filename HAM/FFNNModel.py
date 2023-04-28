import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

class Net(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert len(layers) >= 2
        self.layers = layers

        linear_layers = []

        for i in range(len(self.layers)-1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = torch.nn.Linear(n_in, n_out)

            #He initialization
            a = 1 if i == 0 else 2
            layer.weight.data = torch.randn((n_out, n_in))*np.sqrt(a/n_in)
            layer.bias.data = torch.zeros(n_out)

            linear_layers.append(layer)


        self.linear_layers = torch.nn.ModuleList(linear_layers)

        self.act = torch.nn.ReLU()

    def forward(self, T_rooms, T_wall, T_out, door, timing):
        x = torch.cat((T_rooms, T_wall, T_out, door, timing), dim=1)
        for i in range(len(self.linear_layers)-1):
            x = self.linear_layers[i](x.float())
            x = self.act(x)
        output_layer = self.linear_layers[-1]
        output = output_layer(x)
        return output.float()

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

                    
