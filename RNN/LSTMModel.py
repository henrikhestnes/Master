import copy
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_shape, num_lstm_layers,
                 proj_size = 0,linear_layers=[]):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape)
        self.num_lstm_layers = num_lstm_layers
        self.proj_size = proj_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers,
                            batch_first=True, proj_size=proj_size)
        if linear_layers == []:
            self.linear_layers = [nn.Linear(hidden_size, self.output_size)]
        else:
            #He initialization
            first_layer = nn.Linear(hidden_size, linear_layers[0])
            first_layer.weight.data = torch.randn((hidden_size, linear_layers[0]))*np.sqrt(1/hidden_size)
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
        

    def forward(self, input_seq):
        torch.autograd.set_detect_anomaly(True)
        h_0 = Variable(torch.zeros(self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #hidden state
        c_0 = Variable(torch.zeros(self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #internal state
        output, (hn, cn) = self.lstm(input_seq, (h_0, c_0))

        x = hn[-1]
        for layer in self.linear_layers:
            x = self.act(x)
            x = layer(x)
        return x.reshape((input_seq.size(0), *self.output_shape))

    def train(
            self,
            train_data: DataLoader,
            val_data: DataLoader,
            n_epochs: int,
            lr: float,
            l2_reg: float,
    ) -> torch.nn.Module:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        best_mse = np.inf
        patience = 5
        i_since_last_update = 0

        for epoch in range(n_epochs):
            for inputs, targets in train_data:
                optimizer.zero_grad()
                outputs = self(inputs)

                batch_mse = criterion(outputs, targets)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l2_reg * reg_loss
                cost.backward()
                optimizer.step()
            print(f'Epoch: {epoch+1}, Test loss: {cost.item()}')

            mse_val = 0
            for inputs, label in val_data:
                pred = self(inputs)
                mse_val += torch.sum(torch.pow(label-pred, 2))
            mse_val /= (len(val_data.dataset)*label.shape[2])
            print(f'Epoch: {epoch+1}: Val MSE: {mse_val}')

            #Early stopping
            if mse_val < best_mse:
                best_weights = copy.deepcopy(self.state_dict())
                i_since_last_update = 0
                best_mse = mse_val
            else:
                i_since_last_update += 1

            if i_since_last_update > patience:
                print(f"Stopping early with mse={best_mse}")
                break
        self.load_state_dict(best_weights)