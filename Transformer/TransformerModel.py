# Inspiration: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

# Other good sources: https://github.com/nklingen/Transformer-Time-Series-Forecasting
#                     https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
#                     https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630
#                     https://arxiv.org/pdf/1706.03762.pdf
#                     https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import copy


class Transformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_labels,
        d_model,
        input_seq_length,
        output_seq_length,
        num_heads = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dropout_p = 0.1,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_features = num_features
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

        n_pos_encoding = 6
        d_model = d_model
        d_model_input_features = d_model - n_pos_encoding
        # Layers
        self.src_linear_in = nn.Linear(num_features, d_model_input_features)
        self.tgt_linear_in = nn.Linear(num_labels, d_model_input_features)

        self.pos_encoding = PositionalEncoder()

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )

        self.output_linear = nn.Linear(d_model, num_labels)

    def run_inference(self, src, pos_enc, forecast_window):
        label_indices = [1, 2, 4, 5, 6, 9, 11, 12, 13, 15]
        
        with torch.no_grad():
            self.eval()
            tgt = src[:, -1, label_indices].unsqueeze(1)
            for _ in range(forecast_window-1):
                pred = self(src, tgt, pos_enc)
                tgt = torch.cat((tgt, pred[:, -1, :].unsqueeze(1).detach()), dim=1)
            final_pred = self(src, tgt, pos_enc)
            self.train()
        return final_pred

    def forward(self, src, tgt, pos_enc):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(torch.bool)
        src = self.src_linear_in(src)
        tgt = self.tgt_linear_in(tgt)

        src = self.pos_encoding(src, pos_enc[:, :src.shape[1], :])
        tgt = self.pos_encoding(tgt, pos_enc[:, src.shape[1]-1:src.shape[1]+tgt.shape[1]-1, :])

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_linear(out)

    def train_model(            
            self,
            door_model,
            train_loader: DataLoader,
            val_windows,
            n_epochs: int,
            lr: float,
            l1_reg: float,
    ) -> nn.Module:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        best_mae = float('inf')
        patience = 10
        i_since_last_update = 0

        for epoch in range(n_epochs):
            # train_mse = 0
            # for src, tgt, pos_enc, label in train_loader:
            #     optimizer.zero_grad()

            #     prediction = self(src, tgt, pos_enc)

            #     batch_mse = criterion(prediction, label)
            #     reg_loss = 0
            #     for param in self.parameters():
            #             reg_loss += param.abs().sum()
            #     cost = batch_mse + l1_reg * reg_loss
            #     # print(f'Batch: {i}, Batch Train MSE: {cost.item()}')
            #     train_mse += cost.item()
            #     cost.backward()
            #     optimizer.step()

            # print(f'Epoch: {epoch+1}, Epoch Train MSE: {train_mse/len(train_loader)}')

            rf_val_mae = 0 

            steps_per_pred = 8
            preds_per_day = 12
            num_preds = 40
            np.random.seed(420)
            eval_windows_index = np.random.choice(range(0, len(val_windows)-preds_per_day*steps_per_pred-1), num_preds)
            for start_index in eval_windows_index:
                start_window = val_windows[start_index]
                future_windows = val_windows[start_index+1:start_index+preds_per_day*steps_per_pred+1]
                pred, true = _rolling_forecast(self, door_model, start_window, future_windows, preds_per_day, steps_per_pred)
                rf_val_mae += np.mean(np.abs(true-pred))
            rf_val_mae /= num_preds

            print(f'Epoch: {epoch + 1}: Val MSE: {rf_val_mae}')

            #Early stopping
            if rf_val_mae < best_mae:
                best_weights = copy.deepcopy(self.state_dict())
                i_since_last_update = 0
                best_mae = rf_val_mae
            else:
                i_since_last_update += 1

            if i_since_last_update > patience:
                print(f"Stopping early with mse={best_mae}")
                break

        self.load_state_dict(best_weights)


def _rolling_forecast(model, door_model, start_window, future_windows, steps, timesteps_per_step):
    temp_indices = [1, 2, 4, 5, 6, 9, 11, 12, 13, 15]
    cols_to_update_indices = [8]
    door_indices = [0, 3, 7, 10, 14]
    not_door_indices = [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15]

    current_window = start_window[0].clone()
    current_pos_enc = start_window[2].clone()

    predictions = []
    ground_truth = []
    for i in range(steps):
        next_pred = model.run_inference(current_window.unsqueeze(0), current_pos_enc.unsqueeze(0), timesteps_per_step)
        next_state = current_window[-timesteps_per_step:].clone()
        
        for label_index, window_index in enumerate(temp_indices):
            for n, row in enumerate(next_pred[0]):
                next_state[n][window_index] = row[label_index]
        
        for window_index in cols_to_update_indices:
            for n in range(timesteps_per_step):
                if window_index == cols_to_update_indices[0]:
                    bias = float(np.random.randint(75, 125))/100
                    next_state[n][window_index] = bias*future_windows[timesteps_per_step*i][0][-(timesteps_per_step-n)][window_index]
                else:
                    next_state[n][window_index] = future_windows[timesteps_per_step*i][0][-(timesteps_per_step-n)][window_index]
        
        doors_removed = next_state[:, not_door_indices]
        door_pred = door_model(torch.cat((doors_removed, future_windows[timesteps_per_step*i][2][-8:]), dim=1)).round()
        for door_index, window_index in enumerate(door_indices):
            for n in range(timesteps_per_step):
                next_state[n][window_index] = door_pred[n][door_index]
        current_window = torch.roll(current_window, -timesteps_per_step, dims=0)
        current_window[-timesteps_per_step:] = next_state

        predictions.append(next_pred[0].detach().numpy())
        ground_truth.append(future_windows[timesteps_per_step*i][0][-timesteps_per_step:, temp_indices].detach().numpy())
    
    return np.array(predictions), np.array(ground_truth)


class PositionalEncoder(nn.Module):
    def __init__(
               self,
               dropout=0.1,
               d_model=512,
               batch_first=True
    ):
            super().__init__()
            
            self.d_model = d_model
            self.dropout = nn.Dropout(p=dropout)
            self.batch_first = batch_first
    
    def forward(self, seq, positional_encoding):
         enc = torch.cat((seq, positional_encoding), dim=2)
         return self.dropout(enc)