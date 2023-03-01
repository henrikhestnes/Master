import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WindowGenerator():
    def __init__(self, sequence_width, label_width, shift, train_df, val_df, test_df, label_columns):
        train_labels = train_df.loc[:, label_columns].sort_index(axis=1)
        val_labels = val_df.loc[:, label_columns].sort_index(axis=1)
        test_labels = test_df.loc[:, label_columns].sort_index(axis=1)

        self.sequence_width = sequence_width
        self.label_width = label_width
        self.shift = shift

        self.train_data = []
        for i in range(len(train_df) - (sequence_width + label_width)):
            train_sequence = train_df.iloc[i:i+sequence_width]
            train_label = train_labels.iloc[i+shift+sequence_width:i+shift+sequence_width+label_width]
            self.train_data.append((train_sequence, train_label))

        self.val_data = []
        for i in range(len(val_df) - (sequence_width + label_width)):
            val_sequence = val_df.iloc[i:i+sequence_width]
            val_label = val_labels.iloc[i+shift+sequence_width:i+shift+sequence_width+label_width]
            self.val_data.append((val_sequence, val_label))
        
        self.test_data = []
        for i in range(len(test_df) - (sequence_width + label_width)):
            test_sequence = test_df.iloc[i:i+sequence_width]
            test_label = test_labels.iloc[i+shift+sequence_width:i+shift+sequence_width+label_width]
            self.test_data.append((test_sequence, test_label))


        # Work out the label column indices.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.total_window_size = sequence_width + shift + label_width


    def __repr__(self):
        return '\n'.join([
            f'Window sizes: {self.total_window_size}',
            f'Num train windows: {len(self.train_data)}',
            f'Num val windows: {len(self.val_data)}',
            f'Num test windows: {len(self.test_data)}',])
