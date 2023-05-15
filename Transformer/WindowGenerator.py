import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowGenerator(Dataset):
    def __init__(self, sequence_width, label_width, shift):
        self.sequence_width = int(sequence_width)
        self.label_width = int(label_width)
        self.shift = int(shift)
        self.data = []

    def add_data(self, data, positional_encoding, label_columns):
        positional_encoding = positional_encoding.loc[data.index]
        labels = data.loc[:, label_columns]
        for i in range(len(data) - (self.sequence_width + self.label_width + self.shift)):
            src = data.iloc[i:i+self.sequence_width]
            pos = positional_encoding[i:i+self.shift+self.sequence_width+self.label_width-1]
        
            tgt = labels.iloc[i+self.shift+self.sequence_width-1:i+self.shift+self.sequence_width+self.label_width-1]
            label = labels.iloc[i+self.shift+self.sequence_width:i+self.shift+self.sequence_width+self.label_width]
            # assert all(src.loc[:, label_columns].values[-1] == tgt.values[0])
            self.data.append((torch.tensor(src.values, dtype=torch.float), torch.tensor(tgt.values, dtype=torch.float),
                              torch.tensor(pos.values, dtype=torch.float), torch.tensor(label.values, dtype=torch.float)))


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '\n'.join([
            f'Num train windows: {len(self.data)}'])
