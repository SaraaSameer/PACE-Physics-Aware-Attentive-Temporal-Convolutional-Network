import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class BatteryDataset(Dataset):
    def __init__(self, csv_path, input_window=100, output_window=30, feature_cols=None, label_col='SoH'):
        data = pd.read_csv(csv_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        if feature_cols is None:
            self.feature_cols = ['Uocv', 'R0', 'R1', 'C1', 'I_avg', 'V_avg', 'Time_Stamp', 'Tavg', 'cycle']  # with ecm
            # self.feature_cols = [ 'I_avg', 'V_avg', 'Time_Stamp', 'Tavg', 'cycle']
        else:
            self.feature_cols = feature_cols   
        
        self.input_window = input_window
        self.output_window = output_window
        self.label_col = label_col
        self.X = data[self.feature_cols].values.astype('float32')
        self.y = data[self.label_col].values.astype('float32')
        self.scaler = None

    def __len__(self):
        return len(self.X) - self.input_window - self.output_window + 1

    # def __getitem__(self, idx):
    #     end_idx = idx + self.input_window
    #     target_idx = end_idx + self.output_window
    #     features = self.X[idx:end_idx]
    #     label = self.y[end_idx:target_idx]
    #     return torch.from_numpy(features), torch.from_numpy(label)
    
    def __getitem__(self, idx):
        end_idx = idx + self.input_window
        target_idx = end_idx + self.output_window
        features = self.X[idx:end_idx]
        if self.scaler is not None:
            features_scaled = np.zeros_like(features)
            for i in range(features.shape[0]):
                features_scaled[i] = self.scaler.transform(features[i].reshape(1, -1)).flatten()
            features = features_scaled
            
        label = self.y[end_idx:target_idx]
        return torch.from_numpy(features), torch.from_numpy(label)
    
    def apply_scaler(self, scaler):
        self.scaler = scaler
        print(f"StandardScaler applied to {self.__class__.__name__} (mean: {scaler.mean_}, var: {scaler.var_})")
