import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowGenerator(Dataset):
    def __init__(self, sequence_width, label_width, shift):
        self.sequence_width = int(sequence_width)
        self.label_width = int(label_width)
        self.shift = int(shift)
        self.data = []

    def add_data(self, df, label_columns):
        features = df.iloc[:, :-6]
        positions = df.iloc[:, -6:]
        print(features.shape)
        print(positions.shape)
        labels = df.loc[:, label_columns]
        for i in range(len(df) - (self.sequence_width + self.label_width + self.shift)):
            src = features.iloc[i:i+self.sequence_width]
            tgt = labels.iloc[i+self.shift+self.sequence_width-1:i+self.shift+self.sequence_width+self.label_width-1]
            pos = positions[i:i+self.shift+self.sequence_width+self.label_width-1]
            label = labels.iloc[i+self.shift+self.sequence_width:i+self.shift+self.sequence_width+self.label_width]
            self.data.append((torch.tensor(src.values, dtype=torch.float), torch.tensor(tgt.values, dtype=torch.float),
                              torch.tensor(pos.values, dtype=torch.float), torch.tensor(label.values, dtype=torch.float)))

        # Work out the label column indices.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '\n'.join([
            f'Num train windows: {len(self.data)}'])
