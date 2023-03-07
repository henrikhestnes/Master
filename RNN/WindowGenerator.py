import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowGenerator(Dataset):
    def __init__(self, sequence_width, label_width, shift, df, label_columns):
        labels = df.loc[:, label_columns]#.sort_index(axis=1)

        self.sequence_width = sequence_width
        self.label_width = label_width
        self.shift = shift

        self.data = []
        for i in range(len(df) - (sequence_width + label_width + shift)):
            sequence = df.iloc[i:i+sequence_width]
            label = labels.iloc[i+shift+sequence_width:i+shift+sequence_width+label_width]
            self.data.append((torch.tensor(sequence.values, dtype=torch.float), torch.tensor(label.values, dtype=torch.float)))

        # Work out the label column indices.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        # Work out the window parameters.
        self.total_window_size = sequence_width + shift + label_width

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '\n'.join([
            f'Window sizes: {self.total_window_size}',
            f'Num train windows: {len(self.data)}'])
