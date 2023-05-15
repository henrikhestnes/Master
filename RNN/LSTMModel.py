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
        d = 2 if self.bidir else 1
        h_0 = Variable(torch.zeros(d*self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #hidden state
        c_0 = Variable(torch.zeros(d*self.num_lstm_layers, input_seq.size(0), self.hidden_size)).requires_grad_() #internal state
        output, (hn, cn) = self.lstm(input_seq, (h_0, c_0))

        x = hn[-1]
        for layer in self.linear_layers:
            x = self.act(x)
            x = layer(x)
        return x.reshape((input_seq.size(0), *self.output_shape))

    def train(
            self,
            door_model,
            train_data: DataLoader,
            val_windows,
            n_epochs: int,
            lr: float,
            l1_reg: float,
            final_training: bool = False
    ) -> torch.nn.Module:
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        best_mse = np.inf
        patience = 8
        i_since_last_update = 0

        for epoch in range(n_epochs):
            train_loss = 0
            for i, (data) in enumerate(train_data):
                if len(data) == 4:
                    inputs, targets, true_input, true_target = data
                else:
                    inputs, targets = data
                optimizer.zero_grad()
                outputs = self(inputs)

                batch_mse = criterion(outputs, targets)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l1_reg * reg_loss
                cost.backward()
                optimizer.step()
                train_loss += cost.item()
            print(f'Epoch: {epoch+1}, Train loss: {train_loss/i}')

            if not final_training:
                rf_mae_val = 0

                steps_per_pred = 8
                preds_per_day = 12
                num_preds = 10

                np.random.seed(420)
                eval_windows_index = np.random.choice(range(0, len(val_windows)-preds_per_day*steps_per_pred-1), num_preds)
                for i, start_index in enumerate(eval_windows_index):
                    start_window = val_windows[start_index][0]
                    future_window = val_windows[start_index+1:start_index+steps_per_pred*preds_per_day+1]
                    prediction, ground_truth = _rolling_forecast(self, door_model, start_window, future_window, preds_per_day, steps_per_pred)
                    rf_mae_val += np.mean(np.abs(ground_truth-prediction))
                rf_mae_val /= num_preds

                print(f'Epoch: {epoch+1}: Val RF MAE: {rf_mae_val}')

                #Early stopping
                if rf_mae_val < best_mse:
                    best_weights = copy.deepcopy(self.state_dict())
                    i_since_last_update = 0
                    best_mse = rf_mae_val
                else:
                    i_since_last_update += 1

                if i_since_last_update > patience:
                    print(f"Stopping early with mse={best_mse}")
                    break
        
        if not final_training:
            self.load_state_dict(best_weights)

def _rolling_forecast(model, door_model, start_window, future_windows, steps, timesteps_per_step):
    temp_indices = [1, 2, 4, 5, 6, 9, 11, 12, 13, 15]
    cols_to_update_indices = [8, 16, 17, 18, 19, 20, 21]
    door_indices = [0, 3, 7, 10, 14]
    not_door_indices = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]

    current_window = start_window.clone()

    predictions = []
    ground_truth = []
    for i in range(steps):
        next_pred = model(current_window.reshape(1, current_window.shape[0], current_window.shape[1]))
        next_state = current_window[-timesteps_per_step:].clone()
        
        n_temp = current_window[-1][temp_indices].clone()
        # for label_index, window_index in enumerate(temp_indices):
        for n, row in enumerate(next_pred[0]):
            next_state[n, temp_indices] = row
        
        for window_index in cols_to_update_indices:
            for n in range(timesteps_per_step):
                if window_index == cols_to_update_indices[0]:
                    bias = float(np.random.randint(75, 125))/100
                    next_state[n][window_index] = bias*future_windows[timesteps_per_step*i][0][-(timesteps_per_step-n)][window_index]
                else:
                    next_state[n][window_index] = future_windows[timesteps_per_step*i][0][-(timesteps_per_step-n)][window_index]
        
        doors_removed = next_state[:, not_door_indices]
        door_pred = door_model(doors_removed).round()
        for door_index, window_index in enumerate(door_indices):
            for n in range(timesteps_per_step):
                next_state[n][window_index] = door_pred[n][door_index]
        current_window = torch.roll(current_window, -timesteps_per_step, dims=0)
        current_window[-timesteps_per_step:] = next_state

        predictions.append(next_state[:, temp_indices].detach().numpy())
        ground_truth.append(future_windows[timesteps_per_step*i][0][-timesteps_per_step:, temp_indices].detach().numpy())
    
    return np.array(predictions), np.array(ground_truth)