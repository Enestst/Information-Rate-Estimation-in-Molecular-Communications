import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mp.set_sharing_strategy("file_system")

# --- Configuration ---
BASE_DIR = "/mnt/erencem-ozbey/ber_estimation"
DATA_PATH = os.path.join(BASE_DIR, "data_physics_with_variances_total.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "ber_multiscale_rescnn_attnpool_large.pth")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "ber_multiscale_rescnn_attnpool_scalers_large.pkl")

EPS = 1e-12


def get_sorted_seq_cols(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matched = []
    for col in columns:
        m = pattern.match(col)
        if m:
            matched.append((int(m.group(1)), col))
    matched.sort(key=lambda x: x[0])
    return [col for _, col in matched]


def make_strat_bins(y_log, n_bins=10):
    """
    Build stratification bins from log10(BER).
    Falls back safely if quantile edges collapse.
    """
    y_flat = y_log.reshape(-1)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_flat, quantiles)
    edges = np.unique(edges)

    if len(edges) < 3:
        return None

    bins = np.digitize(y_flat, edges[1:-1], right=True)
    counts = np.bincount(bins)
    if np.any(counts < 2):
        return None
    return bins


class WeightedHuberLoss(nn.Module):
    """
    Mildly weighted Huber in log10(BER) space.
    Lower BER => more negative target => slightly larger weight.
    """
    def __init__(self, delta=0.5, low_ber_weight=0.05):
        super().__init__()
        self.delta = delta
        self.low_ber_weight = low_ber_weight

    def forward(self, pred, target):
        err = pred - target
        abs_err = err.abs()

        huber = torch.where(
            abs_err < self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
        )

        weights = 1.0 + self.low_ber_weight * (-target).clamp(min=0.0)
        return (weights * huber).mean()


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleResidualBlock(nn.Module):
    """
    Multi-branch residual block with 3 receptive fields.
    """
    def __init__(self, in_ch, out_ch, dropout=0.1, se=True):
        super().__init__()
        branch_ch = out_ch // 3
        rem = out_ch - 2 * branch_ch

        self.b1 = ConvBNAct(in_ch, branch_ch, kernel_size=3, dilation=1)
        self.b2 = ConvBNAct(in_ch, branch_ch, kernel_size=5, dilation=1)
        self.b3 = ConvBNAct(in_ch, rem, kernel_size=3, dilation=2)

        self.mix = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.se = SqueezeExcite1D(out_ch) if se else nn.Identity()
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.out_act = nn.GELU()

    def forward(self, x):
        residual = self.skip(x)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.mix(out)
        out = self.se(out)
        out = out + residual
        out = self.out_act(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1, se=True):
        super().__init__()
        self.block = nn.Sequential(
            MultiScaleResidualBlock(in_ch, out_ch, dropout=dropout, se=se),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class ThresholdFiLM(nn.Module):
    """
    Feature-wise linear modulation using threshold embedding.
    """
    def __init__(self, threshold_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(threshold_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        self.feature_dim = feature_dim

    def forward(self, x, thr_emb):
        """
        x: [B, C, L]
        thr_emb: [B, D]
        """
        gamma_beta = self.net(thr_emb)  # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + 0.1 * gamma) + 0.1 * beta


class MultiHeadAttentionPooling(nn.Module):
    """
    Learned attention pooling over sequence length.
    Returns concatenated pooled vectors from multiple heads.
    """
    def __init__(self, input_dim, attn_dim=128, num_heads=4):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(input_dim, attn_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(attn_dim, num_heads, kernel_size=1)
        )
        self.num_heads = num_heads
        self.input_dim = input_dim

    def forward(self, x):
        """
        x: [B, C, L]
        returns: [B, num_heads * C]
        """
        logits = self.score(x)                  # [B, H, L]
        weights = torch.softmax(logits, dim=-1) # [B, H, L]

        pooled = torch.einsum("bhl,bcl->bhc", weights, x)  # [B, H, C]
        pooled = pooled.reshape(x.size(0), self.num_heads * self.input_dim)
        return pooled


class SequenceStem(nn.Module):
    """
    Separate stem for each modality stream.
    """
    def __init__(self, in_ch=1, stem_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, stem_ch, kernel_size=5),
            MultiScaleResidualBlock(stem_ch, stem_ch, dropout=0.05, se=True)
        )

    def forward(self, x):
        return self.net(x)


class BERMultiScaleResCNN(nn.Module):
    """
    Inputs:
      seq: [B, C, L]
        channels are:
          0 -> scaled tap means
          1 -> scaled variances
          2 -> normalized position
          3 -> abs(mean)
          4 -> local snr proxy = mean^2 / (var + eps)

      threshold: [B, 1]

    Output:
      predicted log10(BER)
    """
    def __init__(
        self,
        threshold_dim=1,
        stem_ch=32,
        channels=(64, 96, 128),
        dropout=0.1,
        num_attn_heads=4
    ):
        super().__init__()

        # Separate stems
        self.mean_stem = SequenceStem(in_ch=1, stem_ch=stem_ch)
        self.var_stem = SequenceStem(in_ch=1, stem_ch=stem_ch)
        self.extra_stem = SequenceStem(in_ch=3, stem_ch=stem_ch)

        fusion_in = stem_ch * 3

        self.thr_embed = nn.Sequential(
            nn.Linear(threshold_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        self.stage1 = MultiScaleResidualBlock(fusion_in, channels[0], dropout=dropout, se=True)
        self.film1 = ThresholdFiLM(32, channels[0])

        self.stage2 = DownsampleBlock(channels[0], channels[1], dropout=dropout, se=True)
        self.film2 = ThresholdFiLM(32, channels[1])

        self.stage3 = DownsampleBlock(channels[1], channels[2], dropout=dropout, se=True)
        self.film3 = ThresholdFiLM(32, channels[2])

        self.bottleneck = nn.Sequential(
            MultiScaleResidualBlock(channels[2], channels[2], dropout=dropout, se=True),
            MultiScaleResidualBlock(channels[2], channels[2], dropout=dropout, se=True),
        )

        self.attn_pool = MultiHeadAttentionPooling(
            input_dim=channels[2],
            attn_dim=128,
            num_heads=num_attn_heads
        )

        pooled_dim = num_attn_heads * channels[2]
        stats_dim = channels[2] * 3  # mean + std + max
        head_in = pooled_dim + stats_dim + 32

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, seq, threshold):
        mean_x = seq[:, 0:1, :]   # [B,1,L]
        var_x = seq[:, 1:2, :]    # [B,1,L]
        extra_x = seq[:, 2:5, :]  # [B,3,L]

        mean_f = self.mean_stem(mean_x)
        var_f = self.var_stem(var_x)
        extra_f = self.extra_stem(extra_x)

        x = torch.cat([mean_f, var_f, extra_f], dim=1)

        thr = self.thr_embed(threshold)

        x = self.stage1(x)
        x = self.film1(x, thr)

        x = self.stage2(x)
        x = self.film2(x, thr)

        x = self.stage3(x)
        x = self.film3(x, thr)

        x = self.bottleneck(x)

        attn_pooled = self.attn_pool(x)
        mean_pool = x.mean(dim=-1)
        std_pool = x.std(dim=-1, unbiased=False)
        max_pool = x.amax(dim=-1)

        stats_pool = torch.cat([mean_pool, std_pool, max_pool], dim=1)
        fused = torch.cat([attn_pooled, stats_pool, thr], dim=1)

        return self.head(fused)


def prepare_data(csv_path, batch_size=256, nrows=2000000, num_workers=0):
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

    # Fill missing values
    df[tap_cols] = df[tap_cols].fillna(0.0)
    df[var_cols] = df[var_cols].fillna(0.0)
    df["threshold"] = df["threshold"].fillna(0.0)
    df["BER"] = df["BER"].fillna(0.0)

    # Keep rows valid for logs
    df = df[(df["threshold"] > 0) & (df["BER"] > 0)].copy()

    X_taps = df[tap_cols].to_numpy(dtype=np.float32)
    X_vars = df[var_cols].to_numpy(dtype=np.float32)

    X_thr = np.log10(
        df["threshold"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS
    ).astype(np.float32)

    y_log = np.log10(
        df["BER"].to_numpy(dtype=np.float32).reshape(-1, 1) + EPS
    ).astype(np.float32)

    # Use log-variance if nonnegative
    if np.all(X_vars >= 0):
        X_vars = np.log10(X_vars + EPS).astype(np.float32)

    strat_labels = make_strat_bins(y_log, n_bins=10)

    if strat_labels is not None:
        t_taps, temp_taps, t_vars, temp_vars, t_thr, temp_thr, t_y, temp_y = train_test_split(
            X_taps,
            X_vars,
            X_thr,
            y_log,
            test_size=0.30,
            random_state=42,
            stratify=strat_labels
        )

        temp_strat = make_strat_bins(temp_y, n_bins=6)

        v_taps, te_taps, v_vars, te_vars, v_thr, te_thr, v_y, te_y = train_test_split(
            temp_taps,
            temp_vars,
            temp_thr,
            temp_y,
            test_size=0.50,
            random_state=42,
            stratify=temp_strat if temp_strat is not None else None
        )
    else:
        t_taps, temp_taps, t_vars, temp_vars, t_thr, temp_thr, t_y, temp_y = train_test_split(
            X_taps,
            X_vars,
            X_thr,
            y_log,
            test_size=0.30,
            random_state=42
        )

        v_taps, te_taps, v_vars, te_vars, v_thr, te_thr, v_y, te_y = train_test_split(
            temp_taps,
            temp_vars,
            temp_thr,
            temp_y,
            test_size=0.50,
            random_state=42
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

    # Engineered channels
    L = t_taps.shape[1]
    pos = np.linspace(0.0, 1.0, L, dtype=np.float32)

    def build_features(taps, vars_):
        pos_ch = np.tile(pos, (taps.shape[0], 1)).astype(np.float32)
        abs_taps = np.abs(taps).astype(np.float32)
        snr_proxy = (taps ** 2 / (np.abs(vars_) + 1e-6)).astype(np.float32)

        # [N, C, L] with C=5
        seq = np.stack([taps, vars_, pos_ch, abs_taps, snr_proxy], axis=1).astype(np.float32)
        return seq

    t_seq = build_features(t_taps, t_vars)
    v_seq = build_features(v_taps, v_vars)
    te_seq = build_features(te_taps, te_vars)

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

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_mem,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=num_workers
    )

    scalers = {
        "tap_scaler": tap_scaler,
        "var_scaler": var_scaler,
        "thr_scaler": thr_scaler,
        "tap_cols": tap_cols,
        "var_cols": var_cols,
        "seq_channels": 5,
        "feature_order": ["tap", "var", "pos", "abs_tap", "snr_proxy"],
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

    avg_loss = total_loss / max(len(loader), 1)

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    rmse_log = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae_log = float(np.mean(np.abs(preds - targets)))
    factor_error = float(10 ** rmse_log)

    return avg_loss, rmse_log, mae_log, factor_error


def evaluate_by_target_range(model, loader, device):
    """
    Extra diagnostic metrics by BER regime.
    Uses only RMSE in log space.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_seq, b_thr, b_y in loader:
            b_seq = b_seq.to(device, non_blocking=True)
            b_thr = b_thr.to(device, non_blocking=True)

            pred = model(b_seq, b_thr)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(b_y.numpy())

    preds = np.vstack(all_preds).reshape(-1)
    targets = np.vstack(all_targets).reshape(-1)

    ranges = {
        "low_BER(y<-6)": targets < -6,
        "mid_BER(-6<=y<-3)": (targets >= -6) & (targets < -3),
        "high_BER(y>=-3)": targets >= -3,
    }

    metrics = {}
    for name, mask in ranges.items():
        if np.any(mask):
            rmse = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
            mae = float(np.mean(np.abs(preds[mask] - targets[mask])))
            metrics[name] = {
                "count": int(mask.sum()),
                "rmse_log": rmse,
                "mae_log": mae,
                "factor_error": float(10 ** rmse),
            }
        else:
            metrics[name] = None

    return metrics


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

    model = BERMultiScaleResCNN(
        threshold_dim=1,
        stem_ch=32,
        channels=(64, 96, 128),
        dropout=0.10,
        num_attn_heads=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = WeightedHuberLoss(delta=0.5, low_ber_weight=0.05)

    # Schedule by RMSE, not weighted loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=4,
        factor=0.5
    )

    best_val_rmse = float("inf")
    best_state = None
    patience = 12
    wait = 0

    print(f"Detected sequence length L = {seq_len}")

    for epoch in range(60):
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

        train_loss, train_rmse, train_mae, train_factor = evaluate(model, train_loader, criterion, device)
        val_loss, val_rmse, val_mae, val_factor = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train RMSE(log10): {train_rmse:.4f} (~{train_factor:.2f}x) | "
            f"Val RMSE(log10): {val_rmse:.4f} (~{val_factor:.2f}x) | "
            f"Train MAE(log10): {train_mae:.4f} | Val MAE(log10): {val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f"  -> New best model saved at epoch {epoch+1} with Val RMSE(log10): {val_rmse:.4f}")
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

    range_metrics = evaluate_by_target_range(model, test_loader, device)
    print("\nPer-range test diagnostics:")
    for name, stats in range_metrics.items():
        if stats is None:
            print(f"{name}: no samples")
        else:
            print(
                f"{name} | count={stats['count']} | "
                f"RMSE(log10)={stats['rmse_log']:.4f} | "
                f"MAE(log10)={stats['mae_log']:.4f} | "
                f"factor~{stats['factor_error']:.2f}x"
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
