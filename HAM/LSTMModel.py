import copy
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_shape, num_lstm_layers,
                 proj_size = 0,linear_layers=[], bidir=False):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape)
        self.num_lstm_layers = num_lstm_layers
        self.proj_size = proj_size
        self.bidir = bidir

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            batch_first=True, proj_size=proj_size, bidirectional=bidir)
        if linear_layers == []:
            self.linear_layers = [nn.Linear(hidden_size + 26, self.output_size)]
        else:
            #He initialization
            first_layer = nn.Linear(hidden_size + 26, linear_layers[0])
            first_layer.weight.data = torch.randn((hidden_size+26, linear_layers[0]))*np.sqrt(1/hidden_size)
            first_layer.bias.data = torch.zeros(linear_layers[0])
            self.linear_layers = [first_layer]

            for i in range(len(linear_layers)-1):
                n_in = linear_layers[i]
                n_out = linear_layers[i+1]
                layer = torch.nn.Linear(n_in, n_out)

                layer.weight.data = torch.randn((n_out, n_in))*np.sqrt(2/n_in)
                layer.bias.data = torch.zeros(n_out)
                self.linear_layers.append(layer)
            
            last_layer = nn.Linear(linear_layers[-1], self.output_size)
            last_layer.weight.data = torch.randn((self.output_size, linear_layers[-1]))*np.sqrt(2/linear_layers[-1])
            last_layer.bias.data = torch.zeros(self.output_size)
            self.linear_layers.append(last_layer)

        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.act = nn.ReLU()
        

    def forward(self, input_seq, T_room_hat, T_wall_hat):
        d = 2 if self.bidir else 1
        h_0 = Variable(torch.zeros(d*self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #hidden state
        c_0 = Variable(torch.zeros(d*self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #internal state
        output, (hn, cn) = self.lstm(input_seq, (h_0, c_0))

        x = torch.cat((hn[-1], T_room_hat, T_wall_hat), dim=1)
        for layer in self.linear_layers:
            x = self.act(x)
            x = layer(x)
        return x