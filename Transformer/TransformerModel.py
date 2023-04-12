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


class Transformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_labels,
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
        d_model = 512
        d_model_input_features = d_model - n_pos_encoding
        # Layers
        self.src_linear_in = nn.Linear(num_features, d_model_input_features)
        self.tgt_linear_in = nn.Linear(num_labels, d_model_input_features)

        self.pos_encoding = PositionalEncoder()

        self.transformer = nn.Transformer(
            # d_model=num_features,
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )

        self.output_linear = nn.Linear(d_model, num_labels)

        self.tgt_mask = self.transformer.generate_square_subsequent_mask(output_seq_length)
        #TODO: Do we need src_mask??
        # self.src_mask = self.transformer.generate_square_subsequent_mask(output_seq_length)


    def forward(self, src, tgt, pos_enc):
        src = self.src_linear_in(src)
        tgt = self.tgt_linear_in(tgt)

        src = self.pos_encoding(src, pos_enc[:, :src.shape[1], :])
        tgt = self.pos_encoding(tgt, pos_enc[:, -tgt.shape[1]:, :])

        out = self.transformer(src, tgt, tgt_mask=self.tgt_mask)
        return self.output_linear(out)

    def train(            
            self,
            train_data: DataLoader,
            val_data: DataLoader,
            n_epochs: int,
            lr: float,
            l1_reg: float,
    ) -> nn.Module:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        for epoch in range(n_epochs):
            for i, (src, tgt, pos_enc, label) in enumerate(train_data):
                optimizer.zero_grad()

                prediction = self(src, tgt, pos_enc)

                batch_mse = criterion(prediction, label)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l1_reg * reg_loss
                print(f'Batch: {i}, Test Loss: {cost.item()}')
                cost.backward()
                optimizer.step()
            print(f'Epoch: {epoch+1}, Test loss: {cost.item()}')

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