import json
import os
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from utils import plot_ship_trajectory

WINDOW = 20
HORIZON = 2
STEP = 10
PATIENCE = 5
MIN_DELTA = 1e-4
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-4
DEVICE = "cpu"  # set to "cuda" if GPU

CSV_PATH = "data/ais_data_final.csv"

# Physics-aware features:
FEATURES = [
    "Latitude",
    "Longtitude",
    "SOG",
    "COG",
    "vx",
    "vy",
    "dSOG",
    "dCOG"
]

TARGET_DIM = 2  # Δlat, Δlon


def compute_mae(pred, true):
    if not isinstance(pred, np.ndarray):
        # FIX 1: Ensure input is not double/float64 if it comes from NumPy arrays
        pred = pred.detach().cpu().numpy().astype(np.float32)
    if not isinstance(true, np.ndarray):
        # FIX 1: Ensure input is not double/float64 if it comes from NumPy arrays
        true = true.detach().cpu().numpy().astype(np.float32)

    err = pred - true  # works for both (B,H,2) and (T,2)

    mae_lat = np.mean(np.abs(err[..., 0]))
    mae_lon = np.mean(np.abs(err[..., 1]))

    return mae_lat, mae_lon


def compute_rmse(pred, true):
    if not isinstance(pred, np.ndarray):
        # FIX 1: Ensure input is not double/float64 if it comes from NumPy arrays
        pred = pred.detach().cpu().numpy().astype(np.float32)
    if not isinstance(true, np.ndarray):
        # FIX 1: Ensure input is not double/float64 if it comes from NumPy arrays
        true = true.detach().cpu().numpy().astype(np.float32)

    err = pred - true

    rmse_lat = np.sqrt(np.mean(err[..., 0] ** 2))
    rmse_lon = np.sqrt(np.mean(err[..., 1] ** 2))

    return rmse_lat, rmse_lon


def compute_haversine_metrics(pred_coords, true_coords):
    distances = []
    n = min(len(pred_coords), len(true_coords))

    for i in range(n):
        p = pred_coords[i]
        t = true_coords[i]
        d = haversine(p[0], p[1], t[0], t[1])
        distances.append(d)

    distances = np.array(distances)
    mean_haversine = distances.mean()
    rmse_haversine = np.sqrt((distances ** 2).mean())
    return mean_haversine, rmse_haversine


def compute_speed_error(pred_coords, true_coords):
    speeds_pred = []
    speeds_true = []

    n = min(len(pred_coords) - 1, len(true_coords) - 1)

    for i in range(n):
        dp = haversine(pred_coords[i][0], pred_coords[i][1],
                       pred_coords[i + 1][0], pred_coords[i + 1][1])
        dt = haversine(true_coords[i][0], true_coords[i][1],
                       true_coords[i + 1][0], true_coords[i + 1][1])

        speeds_pred.append(dp / 60.0)  # 1-minute AIS interval
        speeds_true.append(dt / 60.0)

    return np.mean(np.abs(np.array(speeds_pred) - np.array(speeds_true)))


def compute_turn_rate_error(pred_coords, true_coords):
    def compute_cog(lat, lon):
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        # Note: np.degrees(np.arctan2(dlon, dlat)) computes COG from lat/lon differences.
        # This formula is correct for the COG calculation logic used elsewhere.
        return np.degrees(np.arctan2(dlon, dlat))

    n = min(len(pred_coords), len(true_coords))
    pred = pred_coords[:n]
    true = true_coords[:n]

    pred_cog = compute_cog(pred[:, 0], pred[:, 1])
    true_cog = compute_cog(true[:, 0], true[:, 1])

    # Ensure cogs have at least two points to compute turn rate
    if len(pred_cog) < 2 or len(true_cog) < 2:
        return 0.0  # Return 0 if not enough data for turn rate

    pred_turn = np.diff(pred_cog)
    true_turn = np.diff(true_cog)

    n2 = min(len(pred_turn), len(true_turn))
    return np.mean(np.abs(pred_turn[:n2] - true_turn[:n2]))


def haversine(lat1, lon1, lat2, lon2):
    # This uses the standard 'math' library imports (radians, sin, cos, etc.)
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def evaluate_metrics(pred_coords, actual_coords):
    n = min(len(pred_coords), len(actual_coords))
    pred = pred_coords[:n]
    true = actual_coords[:n]
    mae_lat, mae_lon = compute_mae(pred, true)
    mean_hav, rmse_hav = compute_haversine_metrics(pred, true)
    speed_err = compute_speed_error(pred, true)
    turn_err = compute_turn_rate_error(pred, true)

    print("\n===== METRICS =====")
    print(f"MAE Latitude:     {mae_lat:.6f}")
    print(f"MAE Longitude:    {mae_lon:.6f}")
    print(f"Mean Haversine:   {mean_hav:.3f} m")
    print(f"Haversine RMSE:   {rmse_hav:.3f} m")
    print(f"Speed Error:      {speed_err:.4f} m/s")
    print(f"Turn-Rate Error:  {turn_err:.4f} deg")
    print("===================\n")

    return {
        "mae_lat": mae_lat,
        "mae_lon": mae_lon,
        "mean_haversine": mean_hav,
        "rmse_haversine": rmse_hav,
        "speed_error": speed_err,
        "turn_rate_error": turn_err
    }


def compute_physics_features(df):
    """Add vx, vy, dSOG, dCOG features per segment."""
    dfs = []

    for (mmsi, seg), g in df.groupby(["MMSI", "Segment"], observed=True):
        g = g.sort_values("Timestamp").copy()

        # Convert COG to radians
        rad = np.radians(g["COG"].values)

        # Velocity components
        g["vx"] = g["SOG"] * np.cos(rad)
        g["vy"] = g["SOG"] * np.sin(rad)

        # Deltas
        g["dSOG"] = g["SOG"].diff().fillna(0)
        # Handle 360-degree wrap-around for COG difference
        dCOG_raw = g["COG"].diff()
        dCOG_wrapped = (dCOG_raw + 540) % 360 - 180
        g["dCOG"] = dCOG_wrapped.fillna(0)

        dfs.append(g)

    return pd.concat(dfs, ignore_index=True)


class AISTransformer(nn.Module):
    def __init__(self, input_size, embed_size=64, num_heads=2,
                 num_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_size, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.gradient_checkpointing = True

        # predict 1-step delta at a time (autoregressive decoder)
        self.decoder = nn.Linear(embed_size, TARGET_DIM)

    def forward(self, x):
        """Forward only encodes input window; decoding happens outside."""
        # The input x is (B,W,F) and should be float32, which is ensured by the fix.
        z = self.embedding(x)
        z = self.encoder(z)
        return z[:, -1, :]  # last timestep representation


class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        # FIX 2: Ensure all tensors are created with torch.float32 (single precision)
        # NumPy loads data as float64 by default, causing the dtype mismatch.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def autoregressive_rollout(model, window, scaler_X, scaler_y, horizon=HORIZON, training=False):
    device = window.device
    B = window.size(0)

    # scaler stats
    X_mean = torch.tensor(scaler_X.mean_, device=device, dtype=torch.float32)
    X_std  = torch.tensor(scaler_X.scale_, device=device, dtype=torch.float32)
    y_mean = torch.tensor(scaler_y.mean_, device=device, dtype=torch.float32)
    y_std  = torch.tensor(scaler_y.scale_, device=device, dtype=torch.float32)

    pred_list = []

    # Inverse-scale last raw features
    last_raw = window[:, -1, :] * X_std + X_mean

    prev_lat = last_raw[:, 0]
    prev_lon = last_raw[:, 1]

    # During training we don't update physics features
    prev_raw = last_raw.clone()

    for _ in range(horizon):
        # Encode window
        z = model(window)
        delta_scaled = model.decoder(z)
        pred_list.append(delta_scaled)

        # Inverse-scale Δlat/Δlon
        delta_real = delta_scaled * y_std + y_mean
        dlat = delta_real[:, 0]
        dlon = delta_real[:, 1]

        # Update pos
        cur_lat = prev_lat + dlat
        cur_lon = prev_lon + dlon

        if training:
            # 🔥 FAST MODE - NO PHYSICS
            # Only lat/lon change; other features remain constant
            new_raw = prev_raw.clone()
            new_raw[:, 0] = cur_lat
            new_raw[:, 1] = cur_lon

        else:
            # FULL PHYSICS for inference
            dist_m = haversine_torch(prev_lat, prev_lon, cur_lat, cur_lon)
            SOG = dist_m / 60.0
            dY = cur_lat - prev_lat
            dX = cur_lon - prev_lon
            COG = (torch.rad2deg(torch.atan2(dX, dY)) + 360) % 360
            vx = SOG * torch.cos(torch.deg2rad(COG))
            vy = SOG * torch.sin(torch.deg2rad(COG))
            dSOG = SOG - prev_raw[:, 2]
            dCOG = ((COG - prev_raw[:, 3] + 540) % 360) - 180

            new_raw = torch.stack(
                [cur_lat, cur_lon, SOG, COG, vx, vy, dSOG, dCOG],
                dim=1
            )

        # Scale new row
        new_scaled = (new_raw - X_mean) / X_std

        # Slide window
        window = torch.cat([window[:, 1:], new_scaled.unsqueeze(1)], dim=1)

        # Update prev
        prev_lat = cur_lat
        prev_lon = cur_lon
        prev_raw = new_raw

    return torch.stack(pred_list, dim=1), window



def build_dataset():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = compute_physics_features(df)

    raw_windows = []
    raw_targets = []
    meta_rows = []

    print("Building raw (unscaled) windows...")
    for (mmsi, seg), g in df.groupby(["MMSI", "Segment"], observed=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)

        if len(g) < WINDOW + HORIZON + 1:
            continue

        lat = g["Latitude"].values
        lon = g["Longtitude"].values
        dlat = lat[1:] - lat[:-1]
        dlon = lon[1:] - lon[:-1]
        deltas = np.stack([dlat, dlon], axis=1)

        feat = g[FEATURES].values

        for i in range(0, len(g) - WINDOW - HORIZON, STEP):

            X_window_raw = feat[i:i + WINDOW]
            future_raw = deltas[i + WINDOW - 1:i + WINDOW - 1 + HORIZON]

            if future_raw.shape[0] != HORIZON:
                continue

            raw_windows.append(X_window_raw)
            raw_targets.append(future_raw)

            meta_rows.append({
                "mmsi": mmsi,
                "segment": seg,
                "last_lat": lat[i + WINDOW - 1],
                "last_lon": lon[i + WINDOW - 1]
            })

    # FIX 3: Explicitly set NumPy array dtype to float32
    raw_X = np.array(raw_windows, dtype=np.float32)
    raw_y = np.array(raw_targets, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    print(f"Built RAW dataset: X={raw_X.shape}, y={raw_y.shape}, meta={meta.shape}")

    # Ship-based split (done ONCE here)
    train_idx, val_idx, test_idx, train_ships, val_ships, test_ships = ship_based_split(meta)

    # Fit scalers ONLY on train
    train_feat = raw_X[train_idx].reshape(-1, raw_X.shape[-1])
    train_deltas = raw_y[train_idx].reshape(-1, 2)

    global_scaler_X = StandardScaler().fit(train_feat)
    global_scaler_y = StandardScaler().fit(train_deltas)

    # Transform entire dataset
    X_scaled = global_scaler_X.transform(raw_X.reshape(-1, raw_X.shape[-1])).reshape(raw_X.shape).astype(np.float32)
    y_scaled = global_scaler_y.transform(raw_y.reshape(-1, 2)).reshape(raw_y.shape).astype(np.float32)

    print("Dataset scaling done. No leakage.")

    segment_scalers = {
        (meta.loc[i, "mmsi"], meta.loc[i, "segment"]): (global_scaler_X, global_scaler_y)
        for i in range(len(meta))
    }

    return (
        X_scaled, y_scaled, meta,
        segment_scalers,
        train_idx, val_idx, test_idx,
        global_scaler_X, global_scaler_y
    )


def ship_based_split(meta, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, seed=42):
    """
    Split ships into:
      - Train: 64%
      - Validation: 16%
      - Test: 20%

    Using a two-stage split:
      1) 80% train+val, 20% test
      2) 80% / 20% split inside train+val → train / val
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    np.random.seed(seed)

    ships = np.array(meta["mmsi"].unique())
    np.random.shuffle(ships)

    total = len(ships)

    # Compute boundaries
    cutoff_train = int(total * train_ratio)
    cutoff_val = int(total * (train_ratio + val_ratio))

    # Splits
    train_ships = ships[:cutoff_train]
    val_ships = ships[cutoff_train:cutoff_val]
    test_ships = ships[cutoff_val:]

    split_dict = {
        "train_ships": train_ships.tolist(),
        "val_ships": val_ships.tolist(),
        "test_ships": test_ships.tolist(),
        "ratios": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio
        },
        "total_unique_ships": total
    }

    with open("ship_splits.json", "w") as f:
        json.dump(split_dict, f, indent=4)

    print("\n============================")
    print(" Ship-based Split Summary ")
    print("============================")
    print(f"Total unique ships: {total}")
    print(f"Train ships ({len(train_ships)}): {train_ships.tolist()}")
    print(f"Val ships   ({len(val_ships)}): {val_ships.tolist()}")
    print(f"Test ships  ({len(test_ships)}): {test_ships.tolist()}")
    print("============================\n")

    train_idx = meta.index[meta["mmsi"].isin(train_ships)].to_numpy()
    val_idx = meta.index[meta["mmsi"].isin(val_ships)].to_numpy()
    test_idx = meta.index[meta["mmsi"].isin(test_ships)].to_numpy()

    return train_idx, val_idx, test_idx, train_ships, val_ships, test_ships


def train_model():
    (
        X, y, meta,
        segment_scalers,
        train_idx, val_idx, test_idx,
        global_scaler_X, global_scaler_y
    ) = build_dataset()

    train_ds = TrajectoryDataset(X[train_idx], y[train_idx])
    val_ds = TrajectoryDataset(X[val_idx], y[val_idx])
    test_ds = TrajectoryDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4,
    pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4,
    pin_memory=False)

    # Ensure model is on the correct device
    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        # ------------------- TRAIN -------------------
        model.train()
        total_loss = 0
        total_mae_lat = 0
        total_mae_lon = 0
        total_rmse_lat = 0
        total_rmse_lon = 0

        for xb, yb in train_loader:
            # xb, yb are float32 now due to TrajectoryDataset fix
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            pred_scaled, _ = autoregressive_rollout(
                model, xb,
                global_scaler_X, global_scaler_y,
                horizon=HORIZON, training=True
            )

            # FIX 4: pred_scaled is already a float32 Tensor from autoregressive_rollout
            # The line 'pred_t = torch.tensor(pred_scaled, ...)' is problematic
            # as it creates a new Tensor, sometimes inferring dtype incorrectly.
            # pred_scaled is already a Tensor, so we can just use it.
            pred_t = pred_scaled.float()  # Ensure float32

            loss = criterion(pred_t, yb)
            mae_lat, mae_lon = compute_mae(pred_t, yb)
            rmse_lat, rmse_lon = compute_rmse(pred_t, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mae_lat += mae_lat
            total_mae_lon += mae_lon
            total_rmse_lat += rmse_lat
            total_rmse_lon += rmse_lon

        n_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {total_loss / n_batches:.6f}")
        print(f"Train MAE(lat): {total_mae_lat / n_batches:.6f}, "
              f"MAE(lon): {total_mae_lon / n_batches:.6f}")

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                pred_scaled, _ = autoregressive_rollout(
                    model, xb,
                    global_scaler_X, global_scaler_y,
                    horizon=HORIZON, training=False
                )
                pred_t = pred_scaled.float()  # Ensure float32

                val_loss += criterion(pred_t, yb).item()

        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            # Save model for best val loss
            torch.save({
                "model_state": model.state_dict(),
                "segment_scalers": segment_scalers,
                "meta": meta,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "global_scaler_X": global_scaler_X,
                "global_scaler_y": global_scaler_y,
                "X": X, "y": y
            }, "ais_model_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    # Load best model for final testing
    if patience_counter >= PATIENCE:
        print("Loading best model for test evaluation...")
        best_ckpt = torch.load("ais_model_best.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])

    # ------------------- TEST -------------------
    print("\nEvaluating on test set...")
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred_scaled, _ = autoregressive_rollout(
                model, xb,
                global_scaler_X, global_scaler_y,
                horizon=HORIZON, training=False
            )
            pred_t = pred_scaled.float()  # Ensure float32
            test_loss += criterion(pred_t, yb).item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")

    # Final save of the model (could be the best or the last epoch)
    final_save_path = "ais_model_final.pth"
    torch.save({
        "model_state": model.state_dict(),
        "segment_scalers": segment_scalers,
        "meta": meta,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "global_scaler_X": global_scaler_X,
        "global_scaler_y": global_scaler_y,
        "X": X, "y": y
    }, final_save_path)

    print(f"Model state saved to {final_save_path}.")


def predict_sample(idx):
    """
    DEPRECATED: This function uses the non-autoregressive (cheating) prediction
    where all future steps are predicted from a single state representation (z).
    Use predict_sample_autoregressive(idx) instead.
    """
    print("Warning: Using non-autoregressive (cheating) prediction. Use predict_sample_autoregressive.")

    # Load checkpoint ONCE per call
    ckpt = torch.load("ais_model_final.pth", map_location=DEVICE, weights_only=False)

    meta = ckpt["meta"]
    X = ckpt["X"]
    segment_scalers = ckpt["segment_scalers"]

    # Extract scaler
    mmsi = int(meta.loc[idx, "mmsi"])
    seg = int(meta.loc[idx, "segment"])
    _, scaler_y = segment_scalers[(mmsi, seg)]

    # Load model weights correctly
    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Single window, ensure float32
    x = torch.tensor(X[idx:idx + 1], dtype=torch.float32).to(DEVICE)

    # Predict
    with torch.no_grad():
        z = model(x)
        # Prediction of all H steps from a single state (the cheat)
        h_rep = z.unsqueeze(1).repeat(1, HORIZON, 1)
        preds = model.decoder(h_rep)
        # Convert to numpy (HORIZON, 2)
        preds_np = preds.squeeze(0).cpu().numpy()

    # Inverse transform
    deltas = scaler_y.inverse_transform(preds_np)

    lat0 = meta.loc[idx, "last_lat"]
    lon0 = meta.loc[idx, "last_lon"]

    pred_lat = lat0 + np.cumsum(deltas[:, 0])
    pred_lon = lon0 + np.cumsum(deltas[:, 1])
    coords = np.stack([pred_lat, pred_lon], axis=-1)

    return coords


def predict_sample_autoregressive(idx):
    """Autoregressive prediction using ALL physics-aware features."""

    # --- FIX: Ensure the file path is correct ---
    # The file "ais_model_final.pth" contains all necessary context.
    ckpt = torch.load("ais_model_final.pth", map_location=DEVICE, weights_only=False)

    # --- FIX: Load meta and X from the checkpoint ---
    meta = ckpt["meta"]
    X = ckpt["X"]
    segment_scalers = ckpt["segment_scalers"]

    # Extract scalers for this segment
    mmsi = int(meta.loc[idx, "mmsi"])
    seg = int(meta.loc[idx, "segment"])
    # The scalers are saved as a dictionary of tuples in the checkpoint
    scaler_X, scaler_y = ckpt["global_scaler_X"], ckpt["global_scaler_y"]
    # NOTE: The segment_scalers in the checkpoint were redundant as they hold the global scalers.
    # We use the global scalers directly from the checkpoint for simplicity and correctness.

    # Load the model
    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Initial scaled window (already float32 from build_dataset fix)
    # --- FIX: Access the specific window X[idx] ---
    window = X[idx].copy()

    # Last observed actual coordinates
    prev_lat = float(meta.loc[idx, "last_lat"])
    prev_lon = float(meta.loc[idx, "last_lon"])

    # Extract last feature values (scaled → inverse scaled)
    last_feat_scaled = window[-1]
    last_feat = scaler_X.inverse_transform(last_feat_scaled.reshape(1, -1))[0]

    prev_SOG = last_feat[2]
    prev_COG = last_feat[3]

    pred_coords = []

    with torch.no_grad():
        for _ in range(HORIZON):
            # ... (rest of the prediction loop remains as previously corrected,
            # ensuring all haversine inputs are torch tensors) ...

            # Model input: ensure it is float32
            x = torch.tensor(window[np.newaxis, ...], dtype=torch.float32).to(DEVICE)

            # Predict Δlat, Δlon (scaled)
            z = model(x)
            delta_scaled = model.decoder(z)[0].cpu().numpy()

            # Inverse-transform deltas
            delta_real = scaler_y.inverse_transform(delta_scaled.reshape(1, -1))[0]
            dlat, dlon = delta_real

            # Update coordinates (now floats/scalars)
            cur_lat = prev_lat + dlat
            cur_lon = prev_lon + dlon
            pred_coords.append([cur_lat, cur_lon])

            # --- Haversine Fix (Using torch tensors for input) ---
            p_lat = torch.tensor(prev_lat, dtype=torch.float32, device=DEVICE)
            p_lon = torch.tensor(prev_lon, dtype=torch.float32, device=DEVICE)
            c_lat = torch.tensor(cur_lat, dtype=torch.float32, device=DEVICE)
            c_lon = torch.tensor(cur_lon, dtype=torch.float32, device=DEVICE)

            # 1. Compute distance travelled (meters)
            # The haversine function must be the PyTorch-compatible version
            dist_m_tensor = haversine_torch(p_lat, p_lon, c_lat, c_lon)
            dist_m = dist_m_tensor.item()
            # -----------------------------------------------------

            # 2. Compute new SOG (speed over ground)
            SOG = dist_m / 60.0  # AIS = 60 sec interval

            # 3. Compute new COG (direction) (Keep using np/scalars)
            dY = cur_lat - prev_lat
            dX = cur_lon - prev_lon
            COG = (np.degrees(np.arctan2(dX, dY)) + 360) % 360

            # 4. vx, vy: velocity vector components
            vx = SOG * np.cos(np.radians(COG))
            vy = SOG * np.sin(np.radians(COG))

            # 5. dSOG and dCOG (accelerations)
            dSOG = SOG - prev_SOG
            dCOG = (COG - prev_COG + 540) % 360 - 180  # normalize turn rate

            # New raw feature row (unscaled)
            new_row_array = np.array([[
                cur_lat, cur_lon, SOG, COG, vx, vy, dSOG, dCOG
            ]], dtype=np.float32)

            # Scale new row for next window
            new_scaled = scaler_X.transform(new_row_array)[0]

            # Slide window: ensure the new row is float32 before vstack
            window = np.vstack([window[1:], new_scaled.astype(np.float32)])

            # Update prev state
            prev_lat, prev_lon = cur_lat, cur_lon
            prev_SOG, prev_COG = SOG, COG

    return np.array(pred_coords)


def haversine_torch(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Convert degrees to radians using torch.deg2rad
    dlat_rad = torch.deg2rad(dlat)
    dlon_rad = torch.deg2rad(dlon)
    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)

    # Haversine formula using torch functions
    a = torch.sin(dlat_rad / 2) ** 2 + \
        torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon_rad / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c
import folium


def visualize_on_map(actual_coords, predicted_coords, last_lat, last_lon, save_path="ais_map.html"):
    # Center the map around the last observed point
    m = folium.Map(location=[last_lat, last_lon], zoom_start=12)

    # -------------------------------------------------------
    # Last observed position marker
    # -------------------------------------------------------
    folium.Marker(
        [last_lat, last_lon],
        popup="Last Observed Position",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

    # -------------------------------------------------------
    # Actual trajectory line (blue)
    # -------------------------------------------------------
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in actual_coords],
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Actual Trajectory"
    ).add_to(m)

    # Start/End points for actual
    folium.CircleMarker(
        location=[actual_coords[0][0], actual_coords[0][1]],
        radius=5,
        color="green",
        fill=True,
        fill_color="green",
        popup="Actual Start"
    ).add_to(m)

    folium.CircleMarker(
        location=[actual_coords[-1][0], actual_coords[-1][1]],
        radius=5,
        color="blue",
        fill=True,
        fill_color="blue",
        popup="Actual End"
    ).add_to(m)

    # -------------------------------------------------------
    # Predicted trajectory line (orange)
    # -------------------------------------------------------
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in predicted_coords],
        color="orange",
        weight=4,
        opacity=0.8,
        tooltip="Predicted Trajectory"
    ).add_to(m)

    # Start/End points for predicted
    folium.CircleMarker(
        location=[predicted_coords[0][0], predicted_coords[0][1]],
        radius=5,
        color="purple",
        fill=True,
        fill_color="purple",
        popup="Predicted Start"
    ).add_to(m)

    folium.CircleMarker(
        location=[predicted_coords[-1][0], predicted_coords[-1][1]],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        popup="Predicted End"
    ).add_to(m)

    m.save(save_path)
    print(f"Map saved to {save_path}")
    print("Open it in your browser to view the map.")


def evaluate_full_ship(mmsi):
    ckpt = torch.load("ais_model_final.pth", map_location=DEVICE, weights_only=False)

    # Build model and load weights
    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load scalers & meta
    segment_scalers = ckpt["segment_scalers"]
    meta = ckpt["meta"]

    # Load original full dataset (needed for actual trajectory)
    df_full = pd.read_csv(CSV_PATH)
    full_ship_df = df_full[df_full["MMSI"] == mmsi].sort_values("Timestamp")
    full_actual = full_ship_df[["Latitude", "Longtitude"]].to_numpy()

    # Get windows for this ship
    ship_meta = meta[meta["mmsi"] == mmsi].copy()
    if ship_meta.empty:
        print(f"No windows found for MMSI {mmsi}")
        return full_actual, np.array([])

    # Keep consistent order
    ship_meta["window_index"] = ship_meta.index
    ship_meta_sorted = ship_meta.sort_values(["segment", "window_index"])

    all_pred_coords = []
    for idx in ship_meta_sorted["window_index"]:
        # We exclusively use the correct autoregressive prediction function now
        pred_coords = predict_sample_autoregressive(idx)
        all_pred_coords.extend(pred_coords.tolist())

    pred_future = np.array(all_pred_coords)
    return full_actual, pred_future


def visualize_full_ship_route(full_actual, pred_future, mmsi, save_path=None):
    if len(full_actual) == 0:
        print("No actual data to plot.")
        return

    m = folium.Map(location=[full_actual[0][0], full_actual[0][1]], zoom_start=9)

    # FULL ACTUAL TRAJECTORY (blue)
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in full_actual],
        color="blue",
        weight=4,
        opacity=0.9,
        tooltip="Actual Route (Full)"
    ).add_to(m)

    # Start and end markers for actual track
    folium.Marker(
        location=[full_actual[0][0], full_actual[0][1]],
        popup="Start (Actual)",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        location=[full_actual[-1][0], full_actual[-1][1]],
        popup="End (Actual)",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # PREDICTED ROUTE (red)
    if len(pred_future) > 0:
        folium.PolyLine(
            locations=[[lat, lon] for lat, lon in pred_future],
            color="red",
            weight=4,
            opacity=0.9,
            tooltip="Predicted Route"
        ).add_to(m)

    # Save
    if save_path is None:
        save_path = f"ship_{mmsi}_actual_vs_predicted.html"

    m.save(save_path)
    print(f"Map saved to {save_path}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    print(df.head())
    #mmsi = 220464000
    # Good results
    #mmsi = 231025000
    mmsi = 220464000
    # Uncomment the line below to train the corrected model
    #train_model()


    # The evaluation now relies solely on the fixed autoregressive logic
    actual, predicted = evaluate_full_ship(mmsi)
    metrics = evaluate_metrics(predicted, actual)
    # The 'plot_ship_trajectory' utility (if available) and visualization calls remain
    # plot_ship_trajectory(df, mmsi, "temp.html")
    visualize_full_ship_route(actual, predicted, mmsi)