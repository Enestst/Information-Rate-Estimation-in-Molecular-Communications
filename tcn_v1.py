import os
import re
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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "ber_tcn_v1.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "ber_tcn_scalers.pkl")

EPS = 1e-12


def get_sorted_seq_cols(columns, prefix):
    """
    Returns columns like tap_1, tap_2, ... tap_L sorted by numeric suffix.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matched = []
    for col in columns:
        m = pattern.match(col)
        if m:
            matched.append((int(m.group(1)), col))
    matched.sort(key=lambda x: x[0])
    return [col for _, col in matched]


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()

        # "same" padding for odd kernel size
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + residual
        out = self.activation(out)
        return out


class ChannelHistoryTCN(nn.Module):
    """
    Input sequence: [B, 2, L]
      channel 0 -> taps
      channel 1 -> variances

    Separate threshold branch, then fusion head.
    Output: predicted log10(BER)
    """
    def __init__(self, seq_channels=2, threshold_dim=1, dropout=0.1):
        super().__init__()

        self.tcn = nn.Sequential(
            ResidualTCNBlock(seq_channels, 32, kernel_size=3, dilation=1, dropout=dropout),
            ResidualTCNBlock(32, 64, kernel_size=3, dilation=2, dropout=dropout),
            ResidualTCNBlock(64, 64, kernel_size=3, dilation=4, dropout=dropout),
            ResidualTCNBlock(64, 128, kernel_size=3, dilation=8, dropout=dropout),
        )

        self.threshold_branch = nn.Sequential(
            nn.Linear(threshold_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU()
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)   # predicts log10(BER)
        )

    def forward(self, seq, threshold):
        # seq: [B, 2, L]
        x = self.tcn(seq)

        # Global average pooling over sequence length
        x = x.mean(dim=-1)  # [B, 128]

        t = self.threshold_branch(threshold)
        x = torch.cat([x, t], dim=1)

        return self.head(x)


def prepare_data(csv_path, batch_size=256, nrows=500000, num_workers=0):
    df = pd.read_csv(csv_path, nrows=nrows)

    tap_cols = get_sorted_seq_cols(df.columns, "tap")
    var_cols = get_sorted_seq_cols(df.columns, "var")

    if not tap_cols:
        raise ValueError("No tap_* columns found.")
    if not var_cols:
        raise ValueError("No var_* columns found.")
    if len(tap_cols) != len(var_cols):
        raise ValueError(
            f"tap/var length mismatch: {len(tap_cols)} tap cols vs {len(var_cols)} var cols"
        )

    required_cols = tap_cols + var_cols + ["threshold", "BER"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean
    df[tap_cols] = df[tap_cols].fillna(0.0)
    df[var_cols] = df[var_cols].fillna(0.0)
    df["threshold"] = df["threshold"].fillna(0.0)
    df["BER"] = df["BER"].fillna(0.0)

    # Keep rows valid for logs
    df = df[(df["threshold"] > 0) & (df["BER"] > 0)].copy()

    # Raw arrays
    X_taps = df[tap_cols].to_numpy(dtype=np.float32)      # [N, L]
    X_vars = df[var_cols].to_numpy(dtype=np.float32)      # [N, L]
    X_thr = np.log10(df["threshold"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS).astype(np.float32)
    y_log = np.log10(df["BER"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS).astype(np.float32)

    # Optional: log-transform variances if nonnegative and heavy-tailed
    if np.all(X_vars >= 0):
        X_vars = np.log10(X_vars + EPS).astype(np.float32)

    # Split 70 / 15 / 15
    t_taps, temp_taps, t_vars, temp_vars, t_thr, temp_thr, t_y, temp_y = train_test_split(
        X_taps, X_vars, X_thr, y_log, test_size=0.30, random_state=42
    )

    v_taps, te_taps, v_vars, te_vars, v_thr, te_thr, v_y, te_y = train_test_split(
        temp_taps, temp_vars, temp_thr, temp_y, test_size=0.50, random_state=42
    )

    # Separate scalers
    tap_scaler = StandardScaler()
    var_scaler = StandardScaler()
    thr_scaler = StandardScaler()

    t_taps = tap_scaler.fit_transform(t_taps).astype(np.float32)
    v_taps = tap_scaler.transform(v_taps).astype(np.float32)
    te_taps = tap_scaler.transform(te_taps).astype(np.float32)

    t_vars = var_scaler.fit_transform(t_vars).astype(np.float32)
    v_vars = var_scaler.transform(v_vars).astype(np.float32)
    te_vars = var_scaler.transform(te_vars).astype(np.float32)

    t_thr = thr_scaler.fit_transform(t_thr).astype(np.float32)
    v_thr = thr_scaler.transform(v_thr).astype(np.float32)
    te_thr = thr_scaler.transform(te_thr).astype(np.float32)

    # Stack taps + vars as channels: [N, 2, L]
    t_seq = np.stack([t_taps, t_vars], axis=1).astype(np.float32)
    v_seq = np.stack([v_taps, v_vars], axis=1).astype(np.float32)
    te_seq = np.stack([te_taps, te_vars], axis=1).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(t_seq),
        torch.from_numpy(t_thr),
        torch.from_numpy(t_y)
    )
    val_ds = TensorDataset(
        torch.from_numpy(v_seq),
        torch.from_numpy(v_thr),
        torch.from_numpy(v_y)
    )
    test_ds = TensorDataset(
        torch.from_numpy(te_seq),
        torch.from_numpy(te_thr),
        torch.from_numpy(te_y)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )

    scalers = {
        "tap_scaler": tap_scaler,
        "var_scaler": var_scaler,
        "thr_scaler": thr_scaler,
        "tap_cols": tap_cols,
        "var_cols": var_cols,
    }

    return train_loader, val_loader, test_loader, scalers, len(tap_cols)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_seq, b_thr, b_y in loader:
            b_seq = b_seq.to(device, non_blocking=True)
            b_thr = b_thr.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            pred = model(b_seq, b_thr)
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

    train_loader, val_loader, test_loader, scalers, seq_len = prepare_data(
        DATA_PATH,
        batch_size=256,
        nrows=500000,
        num_workers=0
    )

    joblib.dump(scalers, SCALER_SAVE_PATH)

    model = ChannelHistoryTCN(seq_channels=2, threshold_dim=1, dropout=0.1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    best_val = float("inf")
    best_state = None
    patience = 10
    wait = 0

    print(f"Detected sequence length L = {seq_len}")

    for epoch in range(50):
        model.train()

        for b_seq, b_thr, b_y in train_loader:
            b_seq = b_seq.to(device, non_blocking=True)
            b_thr = b_thr.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(b_seq, b_thr)
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
        print(f"Scalers saved to: {SCALER_SAVE_PATH}")
    else:
        print(f"Critical Error: Data file not found at {DATA_PATH}")
