# Inspiration: https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_basic.py
# Other good sources: https://github.com/nklingen/Transformer-Time-Series-Forecasting
#                     https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
#                     https://towardsdatascience.com/how-to-use-transformer-networks-to-build-a-forecasting-model-297f9270e630
#                     https://arxiv.org/pdf/1706.03762.pdf
#                     https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#                     https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# import torch
# import torch.optim as optim

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

        d_model = 512

        # Layers
        self.src_linear_in = nn.Linear(num_features, d_model)
        self.tgt_linear_in = nn.Linear(num_labels, d_model)

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


    def forward(self, src, tgt):
        #TODO: Add positional encoding logic
        src = self.src_linear_in(src)
        tgt = self.tgt_linear_in(tgt)
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
            i=0
            for src, tgt, label in train_data:
                print(i)
                i+=1
                optimizer.zero_grad()

                prediction = self(src, tgt)

                batch_mse = criterion(prediction, label)
                reg_loss = 0
                for param in self.parameters():
                        reg_loss += param.abs().sum()
                cost = batch_mse + l1_reg * reg_loss
                cost.backward()
                optimizer.step()
            print(f'Epoch: {epoch+1}, Test loss: {cost.item()}')