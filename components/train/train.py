import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset

# 1. Dataset Class: Converts flat CSV rows into "Windows" of time
class SepsisDataset(Dataset):
    def __init__(self, csv_path, window_size=6):
        df = pd.read_csv(csv_path)
        self.y = df['SepsisLabel'].values
        # Drop label to get features
        self.X = df.drop(columns=['SepsisLabel']).values
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # Returns a window of (window_size, num_features)
        return (
            torch.tensor(self.X[idx : idx + self.window_size], dtype=torch.float32),
            torch.tensor(self.y[idx + self.window_size], dtype=torch.float32)
        )

# 2. Simple LSTM Model
class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
        # return self.sigmoid(out)

def train(input_path, epochs=5):
    dataset = SepsisDataset(input_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Auto-detect input dimension from the data
    input_dim = dataset.X.shape[1]
    model = SepsisLSTM(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Complete. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    train(args.input)