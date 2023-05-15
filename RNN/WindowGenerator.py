import torch
from torch.utils.data import Dataset
import pandas as pd

class WindowGenerator(Dataset):
    def __init__(self, sequence_width, label_width, shift):
        self.sequence_width = int(sequence_width)
        self.label_width = int(label_width)
        self.shift = int(shift)
        self.data = []

    def add_data(self, df, label_columns, df_not_scaled = None):
        labels = df.loc[:, label_columns]
        not_scaled_labels = df_not_scaled.loc[:, label_columns]
        for i in range(len(df) - (self.sequence_width + self.label_width + self.shift)):
            sequence = df.iloc[i:i+self.sequence_width]
            label = labels.iloc[i+self.shift+self.sequence_width:i+self.shift+self.sequence_width+self.label_width]
            if df_not_scaled is not None:
                not_scaled_sequence = df_not_scaled.iloc[i:i+self.sequence_width]
                not_scaled_label = not_scaled_labels.iloc[i+self.shift+self.sequence_width:i+self.shift+self.sequence_width+self.label_width]
                self.data.append((torch.tensor(sequence.values, dtype=torch.float),
                              torch.tensor(label.values, dtype=torch.float),
                              torch.tensor(not_scaled_sequence.values, dtype=torch.float),
                              torch.tensor(not_scaled_label.values, dtype=torch.float)))
            else:
                self.data.append((torch.tensor(sequence.values, dtype=torch.float),
                              torch.tensor(label.values, dtype=torch.float)))

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
            f'Window sizes: {self.total_window_size}',
            f'Num train windows: {len(self.data)}'])
