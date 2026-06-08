import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
BASE_DIR = "/mnt/erencem-ozbey/ber_estimation"
DATA_PATH = os.path.join(BASE_DIR, "data_physics_with_variances_total.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "molecular_cnn_v3.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "threshold_scaler.pkl")

EPS = 1e-12


class MolecularCNN(nn.Module):
    """
    taps branch: CNN over tap_1...tap_14
    vars branch: CNN over var_1...var_14
    threshold branch: MLP over threshold
    output: predicted log10(BER)
    """
    def __init__(self, scalar_size=1):
        super().__init__()

        self.tap_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(4)
        )

        self.var_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(4)
        )

        self.threshold_fc = nn.Sequential(
            nn.Linear(scalar_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear((64 * 4) + (64 * 4) + 16, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # predicts log10(BER)
        )

    def forward(self, taps, vars_, threshold):
        x_taps = taps.unsqueeze(1)            # [B, 1, 14]
        x_taps = self.tap_conv(x_taps)
        x_taps = x_taps.view(x_taps.size(0), -1)

        x_vars = vars_.unsqueeze(1)           # [B, 1, 14]
        x_vars = self.var_conv(x_vars)
        x_vars = x_vars.view(x_vars.size(0), -1)

        x_thr = self.threshold_fc(threshold)

        combined = torch.cat((x_taps, x_vars, x_thr), dim=1)
        return self.fusion(combined)


def prepare_data(csv_path, batch_size=128, nrows=500000):
    df = pd.read_csv(csv_path, nrows=nrows)

    tap_cols = [f"tap_{i}" for i in range(1, 15)]
    var_cols = [f"var_{i}" for i in range(1, 15)]

    required_cols = tap_cols + var_cols + ["threshold", "BER"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean
    df[tap_cols] = df[tap_cols].fillna(0.0)
    df[var_cols] = df[var_cols].fillna(0.0)
    df["threshold"] = df["threshold"].fillna(0.0)
    df["BER"] = df["BER"].fillna(0.0)

    # Remove invalid rows for log transform
    df = df[(df["threshold"] > 0) & (df["BER"] > 0)].copy()

    X_taps = df[tap_cols].to_numpy(dtype=np.float32)
    X_vars = df[var_cols].to_numpy(dtype=np.float32)
    X_thr = np.log10(df["threshold"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS).astype(np.float32)

    # Target is log10(BER)
    y_log = np.log10(df["BER"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS).astype(np.float32)

    # 70 / 15 / 15 split
    t_taps, temp_taps, t_vars, temp_vars, t_thr, temp_thr, t_y, temp_y = train_test_split(
        X_taps, X_vars, X_thr, y_log, test_size=0.30, random_state=42
    )

    v_taps, te_taps, v_vars, te_vars, v_thr, te_thr, v_y, te_y = train_test_split(
        temp_taps, temp_vars, temp_thr, temp_y, test_size=0.50, random_state=42
    )

    scaler = StandardScaler()
    t_thr = scaler.fit_transform(t_thr).astype(np.float32)
    v_thr = scaler.transform(v_thr).astype(np.float32)
    te_thr = scaler.transform(te_thr).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(t_taps),
        torch.from_numpy(t_vars),
        torch.from_numpy(t_thr),
        torch.from_numpy(t_y)
    )
    val_ds = TensorDataset(
        torch.from_numpy(v_taps),
        torch.from_numpy(v_vars),
        torch.from_numpy(v_thr),
        torch.from_numpy(v_y)
    )
    test_ds = TensorDataset(
        torch.from_numpy(te_taps),
        torch.from_numpy(te_vars),
        torch.from_numpy(te_thr),
        torch.from_numpy(te_y)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, scaler


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_taps, b_vars, b_thr, b_y in loader:
            b_taps = b_taps.to(device, non_blocking=True)
            b_vars = b_vars.to(device, non_blocking=True)
            b_thr = b_thr.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            pred = model(b_taps, b_vars, b_thr)
            loss = criterion(pred, b_y)

            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(b_y.cpu().numpy())

    avg_loss = total_loss / len(loader)

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    rmse_log = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae_log = float(np.mean(np.abs(preds - targets)))
    factor_error = float(10 ** rmse_log)

    return avg_loss, rmse_log, mae_log, factor_error


def train_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    train_loader, val_loader, test_loader, scaler = prepare_data(
        DATA_PATH,
        batch_size=128,
        nrows=500000
    )
    joblib.dump(scaler, SCALER_SAVE_PATH)

    model = MolecularCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    best_val = float("inf")
    best_state = None
    patience = 10
    wait = 0

    for epoch in range(50):
        model.train()
        for b_taps, b_vars, b_thr, b_y in train_loader:
            b_taps = b_taps.to(device, non_blocking=True)
            b_vars = b_vars.to(device, non_blocking=True)
            b_thr = b_thr.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(b_taps, b_vars, b_thr)
            loss = criterion(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        train_loss, train_rmse, _, train_factor = evaluate(model, train_loader, criterion, device)
        val_loss, val_rmse, _, val_factor = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train RMSE(log10): {train_rmse:.4f} (~{train_factor:.2f}x) | "
            f"Val RMSE(log10): {val_rmse:.4f} (~{val_factor:.2f}x)"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_rmse, test_mae, test_factor = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test RMSE(log10): {test_rmse:.4f} | "
        f"Test MAE(log10): {test_mae:.4f} | "
        f"Typical multiplicative error: ~{test_factor:.2f}x"
    )

    return model


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"Reading data from: {DATA_PATH}")
        trained_model = train_engine()
        torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print(f"Scaler saved to: {SCALER_SAVE_PATH}")
    else:
        print(f"Critical Error: Data file not found at {DATA_PATH}")
