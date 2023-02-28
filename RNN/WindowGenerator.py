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

        self.input_slice = slice(0, sequence_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        return '\n'.join([
            f'Window sizes: {self.total_window_size}',
            f'Num train windows: {len(self.train_data)}',
            f'Num val windows: {len(self.val_data)}',
            f'Num test windows: {len(self.test_data)}',])
    
    def get_train_data(self):
        return self.train_data

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = np.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

