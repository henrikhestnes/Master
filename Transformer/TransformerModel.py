# Inspiration: https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_basic.py
# Other good sources: https://github.com/nklingen/Transformer-Time-Series-Forecasting
#                     https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
#                     https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630
#                     https://arxiv.org/pdf/1706.03762.pdf
#                     https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#                     https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# import torch
# import torch.optim as optim
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import copy

def run_inference(model, src, pos_enc, forecast_window):
    label_indices = [1, 2, 4, 5, 6, 9, 11, 12, 13, 15]

    tgt = src[:, -1, label_indices].reshape(src.shape[0], 1, len(label_indices))
    for _ in range(forecast_window-1):
        pred = model(src, tgt, pos_enc)
        tgt = torch.cat((tgt, pred[:, -1, :].unsqueeze(1).detach()), dim=1)
    final_pred = model(src, tgt, pos_enc)
    return final_pred

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


    def forward(self, src, tgt, pos_enc):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
        src = self.src_linear_in(src)
        tgt = self.tgt_linear_in(tgt)

        src = self.pos_encoding(src, pos_enc[:, :src.shape[1], :])
        tgt = self.pos_encoding(tgt, pos_enc[:, src.shape[1]-1:src.shape[1]+tgt.shape[1]-1, :])

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_linear(out)

    def train_model(            
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int,
            lr: float,
            l1_reg: float,
    ) -> nn.Module:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        best_mse = float('inf')
        patience = 10
        i_since_last_update = 0

        for epoch in range(n_epochs):
            train_mse = 0
            for i, (src, tgt, pos_enc, label) in enumerate(train_loader):
                optimizer.zero_grad()

                prediction = self(src, tgt, pos_enc)

                batch_mse = criterion(prediction, label)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l1_reg * reg_loss
                # print(f'Batch: {i}, Batch Train MSE: {cost.item()}')
                train_mse += cost.item()
                cost.backward()
                optimizer.step()

            print(f'Epoch: {epoch+1}, Epoch Train MSE: {train_mse/len(train_loader)}')

            val_mse = 0 
            with torch.no_grad():
                self.eval()
                for src, tgt, pos, label in val_loader:
                    pred = run_inference(self, src, pos, label.shape[1])
                    val_mse += torch.mean(torch.pow(label-pred, 2))
                val_mse /= len(val_loader)
                self.train()

            print(f'Epoch: {epoch + 1}: Val MSE: {val_mse}')

            #Early stopping
            if val_mse < best_mse:
                best_weights = copy.deepcopy(self.state_dict())
                i_since_last_update = 0
                best_mse = val_mse
            else:
                i_since_last_update += 1

            if i_since_last_update > patience:
                print(f"Stopping early with mse={best_mse}")
                break

        self.load_state_dict(best_weights)

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