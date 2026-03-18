import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class LogBERLoss(nn.Module):
    """Loss function operating in log-scale to handle BER magnitudes."""
    def __init__(self):
        super(LogBERLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        eps = 1e-9
        return self.mse(torch.log10(pred + eps), torch.log10(target + eps))

class MolecularCNN(nn.Module):
    """Hybrid architecture combining 1D-CNN for taps and MLP for scalars."""
    def __init__(self, tap_size=14, scalar_size=4):
        super(MolecularCNN, self).__init__()
        
        # Extract temporal features from hitting probability sequence
        self.tap_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(4) 
        )
        
        # Process global parameters (p0, N, threshold, mem_len)
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Feature fusion and final regression
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 + 32, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, taps, scalars):
        x_taps = taps.unsqueeze(1) # Add channel dimension for Conv1d
        x_taps = self.tap_conv(x_taps)
        x_taps = x_taps.view(x_taps.size(0), -1) 
        
        x_scal = self.scalar_fc(scalars)
        
        combined = torch.cat((x_taps, x_scal), dim=1)
        return self.fusion(combined)

def prepare_data(csv_path, batch_size=64):
    df = pd.read_csv(csv_path)
    
    # Ensure all 14 tap columns are present and handle missing values
    tap_cols = [f'tap_{i}' for i in range(1, 15)]
    for col in tap_cols:
        if col not in df.columns:
            df[col] = 0.0
    df[tap_cols] = df[tap_cols].fillna(0.0)
    
    X_taps = df[tap_cols].values.astype(np.float32)
    
    # Log-transform highly skewed physical variables (N and threshold)
    X_scal = np.column_stack([
        df['p0'].values,
        np.log10(df['N'].values + 1e-9),
        np.log10(df['threshold'].values + 1e-9),
        df['mem_len'].values
    ]).astype(np.float32)
    
    y = df['BER'].values.reshape(-1, 1).astype(np.float32)

    # 85/15 Train/Validation split
    t_taps, v_taps, t_scal, v_scal, t_y, v_y = train_test_split(
        X_taps, X_scal, y, test_size=0.15, random_state=42
    )

    # Standardize metadata features
    scaler = StandardScaler()
    t_scal = scaler.fit_transform(t_scal)
    v_scal = scaler.transform(v_scal)

    train_ds = TensorDataset(torch.from_numpy(t_taps), torch.from_numpy(t_scal), torch.from_numpy(t_y))
    val_ds = TensorDataset(torch.from_numpy(v_taps), torch.from_numpy(v_scal), torch.from_numpy(v_y))
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size), \
           scaler

def train_engine(csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = prepare_data(csv_file)
    
    model = MolecularCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = LogBERLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(50):
        model.train()
        for b_taps, b_scal, b_y in train_loader:
            b_taps, b_scal, b_y = b_taps.to(device), b_scal.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(b_taps, b_scal), b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_taps, b_scal, b_y in val_loader:
                b_taps, b_scal, b_y = b_taps.to(device), b_scal.to(device), b_y.to(device)
                val_loss += criterion(model(b_taps, b_scal), b_y).item()
        
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:02d} | Val Log-Loss: {avg_val:.4f}")

    return model

if __name__ == "__main__":
    DATA_PATH = "data_physics_total.csv" 
    if os.path.exists(DATA_PATH):
        print(f"Loading {DATA_PATH} and beginning training...")
        trained_model = train_engine(DATA_PATH)
        torch.save(trained_model.state_dict(), "molecular_cnn_v1.pth")
        print("Model saved to disk.")
    else:
        print(f"Error: {DATA_PATH} not found.")