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
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        x = input
        for i in range(len(self.linear_layers)-1):
            x = self.linear_layers[i](x)
            x = self.act(x)
        output_layer = self.linear_layers[-1]
        output = self.sigmoid(output_layer(x))
        return output

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())
    
    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int,
            lr: float,
            l2_reg: float,
    ) -> torch.nn.Module:
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        best_acc = 0
        patience = 50
        i_since_last_update = 0
        for epoch in range(n_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                batch_mse = criterion(outputs, labels)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l2_reg * reg_loss
                cost.backward()
                optimizer.step()
            acc = 0

            for inputs, labels in val_loader:
                pred = self(inputs)
                row_len = len(labels[0])
                acc += (pred.round() == labels).float().mean()
            print(f'Epoch: {epoch + 1}: Val acc: {acc}')
            
            #Early stopping
            if acc > best_acc:
                best_weights = copy.deepcopy(self.state_dict())
                i_since_last_update = 0
                best_acc = acc
            else:
                i_since_last_update += 1

            if i_since_last_update > patience:
                print(f"Stopping early with acc={best_acc}")
                break
        print("updating self")
        self.load_state_dict(best_weights)
                    
