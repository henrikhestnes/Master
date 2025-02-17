import torch
from torch.utils.data import Dataset
import pandas as pd

class Dataset(Dataset):
    def __init__(self, label_width, warmup_width, indoor_temp_columns, outdoor_temp_columns, 
                 radiation_column, door_columns, timing_columns, is_lstm=False, x_scaler=None, label_scaler=None):
        self.label_width = int(label_width)
        self.warmup_width = int(warmup_width)
        self.indoor_temp_columns = indoor_temp_columns
        self.outdoor_temp_columns = outdoor_temp_columns
        self.radiation_column = radiation_column
        self.door_columns = door_columns
        self.timing_columns = timing_columns
        self.is_lstm = is_lstm
        self.x_scaler = x_scaler
        self.label_scaler = label_scaler
        self.data = []

    def add_data(self, df):
        indoor_temp_data = df.loc[:, self.indoor_temp_columns]
        outdoor_temp_data = df.loc[:, self.outdoor_temp_columns]
        radiation_data = df.loc[:, self.radiation_column]
        door_data = df.loc[:, self.door_columns]
        timing_data = df.loc[:, self.timing_columns]

        for i in range(len(df) - (self.label_width + self.warmup_width + 1)):
            warmup_indoor_temp = indoor_temp_data.iloc[i:i+self.warmup_width]
            warmup_outdoor_temp = outdoor_temp_data.iloc[i:i+self.warmup_width]
            warmup_radiation = radiation_data.iloc[i:i+self.warmup_width]
            indoor_temp = indoor_temp_data.iloc[i+self.warmup_width]
            outdoor_temp = outdoor_temp_data.iloc[i+self.warmup_width:i+self.warmup_width+self.label_width]
            radiation = radiation_data.iloc[i+self.warmup_width:i+self.warmup_width+self.label_width]
            door = door_data.iloc[i+self.warmup_width:i+self.warmup_width+self.label_width]
            timing = timing_data.iloc[i+self.warmup_width:i+self.warmup_width+self.label_width]
            label = indoor_temp_data.iloc[i+self.warmup_width+1:i+self.warmup_width+self.label_width+1]
            if self.is_lstm:
                lstm_input = df.iloc[i:i+self.warmup_width+1].copy()
                lstm_input.loc[:, self.indoor_temp_columns | self.outdoor_temp_columns] = lstm_input.loc[:, self.indoor_temp_columns | self.outdoor_temp_columns].diff().shift(-1).dropna()
                lstm_input = lstm_input.iloc[:-1]
                self.data.append((torch.tensor(warmup_indoor_temp.values, dtype=torch.float32),
                              torch.tensor(warmup_outdoor_temp.values, dtype=torch.float32),
                              torch.tensor(warmup_radiation.values, dtype=torch.float32),
                              torch.tensor(indoor_temp.values, dtype=torch.float32),
                              torch.tensor(outdoor_temp.values.flatten(), dtype=torch.float32),
                              torch.tensor(radiation.values.flatten(), dtype=torch.float32),
                              torch.tensor(door.values, dtype=torch.float32),
                              torch.tensor(timing.values, dtype=torch.float32),
                              torch.tensor(label.values, dtype=torch.float32),
                              torch.tensor(self.x_scaler.transform(lstm_input), dtype=torch.float32)))
            else:
                self.data.append((torch.tensor(warmup_indoor_temp.values, dtype=torch.float32),
                              torch.tensor(warmup_outdoor_temp.values, dtype=torch.float32), 
                              torch.tensor(indoor_temp.values, dtype=torch.float32),
                              torch.tensor(outdoor_temp.values.flatten(), dtype=torch.float32),
                              torch.tensor(door.values, dtype=torch.float32),
                              torch.tensor(timing.values, dtype=torch.float32),
                              torch.tensor(label.values, dtype=torch.float32)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
