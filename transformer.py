import json
import os
from math import radians, sin, cos, sqrt, atan2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib


# ===========================================================
# GLOBAL CONSTANTS
# ===========================================================
WINDOW = 60          # input sequence length
HORIZON = 10         # predict 10 minutes ahead
STEP = 10            # window stride

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
DEVICE = "cpu"       # set to "cuda" if GPU

CSV_PATH = "data/ais_data_1min_clean.csv"

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

TARGET_DIM = 2       # Δlat, Δlon


# ===========================================================
# HELPERS
# ===========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def compute_physics_features(df):
    """Add vx, vy, dSOG, dCOG features per segment."""
    dfs = []

    for (mmsi, seg), g in df.groupby(["MMSI", "Segment_new"], observed=True):
        g = g.sort_values("Timestamp").copy()

        # Convert COG to radians
        rad = np.radians(g["COG"].values)

        # Velocity components
        g["vx"] = g["SOG"] * np.cos(rad)
        g["vy"] = g["SOG"] * np.sin(rad)

        # Deltas
        g["dSOG"] = g["SOG"].diff().fillna(0)
        g["dCOG"] = g["COG"].diff().fillna(0)

        dfs.append(g)

    return pd.concat(dfs, ignore_index=True)


# ===========================================================
# TRANSFORMER MODEL (autoregressive)
# ===========================================================
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

        # predict 1-step delta at a time (autoregressive decoder)
        self.decoder = nn.Linear(embed_size, TARGET_DIM)

    def forward(self, x):
        """Forward only encodes input window; decoding happens outside."""
        z = self.embedding(x)
        z = self.encoder(z)
        return z[:, -1, :]     # last timestep representation


# ===========================================================
# DATASET CLASS
# ===========================================================
class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================================================
# DATASET BUILDING (A + B + C)
# ===========================================================
def build_dataset():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Adding physics-aware features...")
    df = compute_physics_features(df)

    # Store scalers per segment (A)
    segment_scalers = {}

    X_samples = []
    y_samples = []
    meta_rows = []

    print("Building sliding windows...")
    for (mmsi, seg), g in df.groupby(["MMSI", "Segment_new"], observed=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)

        if len(g) < WINDOW + HORIZON + 1:
            continue

        # -----------------------------
        # Fit feature scaler per segment
        scaler_X = StandardScaler().fit(g[FEATURES])
        X_scaled = scaler_X.transform(g[FEATURES])

        # Compute deltas (target)
        lat = g["Latitude"].values
        lon = g["Longtitude"].values

        dlat = lat[1:] - lat[:-1]
        dlon = lon[1:] - lon[:-1]
        deltas = np.stack([dlat, dlon], axis=-1)   # (N-1, 2)

        # Fit ROBUST scaler for deltas (C)
        scaler_y = RobustScaler().fit(deltas)
        delta_scaled = scaler_y.transform(deltas)

        # save segment scalers
        segment_scalers[(mmsi, seg)] = (scaler_X, scaler_y)

        # -----------------------------
        # Create windows
        for i in range(0, len(g) - WINDOW - HORIZON, STEP):

            X_window = X_scaled[i:i + WINDOW]

            # HORIZON future deltas (scaled)
            future_delta_scaled = delta_scaled[i + WINDOW - 1 : i + WINDOW - 1 + HORIZON]

            if future_delta_scaled.shape[0] != HORIZON:
                continue

            X_samples.append(X_window)
            y_samples.append(future_delta_scaled)

            meta_rows.append({
                "mmsi": int(mmsi),
                "segment": int(seg),
                "last_lat": float(lat[i + WINDOW - 1]),
                "last_lon": float(lon[i + WINDOW - 1])
            })

    X = np.array(X_samples)
    y = np.array(y_samples)

    print("Saving X, y, meta and scalers...")
    np.save("X.npy", X)
    np.save("y.npy", y)
    pd.DataFrame(meta_rows).to_csv("meta.csv", index=False)
    joblib.dump(segment_scalers, "segment_scalers.pkl")

    print(f"Dataset built: X={X.shape}, y={y.shape}")


# ===========================================================
# TRAINING (autoregressive)
# ===========================================================
def train_model():
    if not os.path.exists("X.npy"):
        build_dataset()

    print("Loading dataset...")
    X = np.load("X.npy")
    y = np.load("y.npy")

    dataset = TrajectoryDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=1.0)

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            # Encode input
            z = model(xb)

            # Autoregressive decoding (D)
            preds = []
            h = z
            for t in range(HORIZON):
                step_pred = model.decoder(h)
                preds.append(step_pred)
                h = h   # we keep encoder state fixed

            preds = torch.stack(preds, dim=1)  # (batch, HORIZON, 2)

            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                z = model(xb)

                preds = []
                h = z
                for _ in range(HORIZON):
                    step_pred = model.decoder(h)
                    preds.append(step_pred)
                preds = torch.stack(preds, dim=1)

                val_loss += criterion(preds, yb).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train {tr_loss:.6f} | Val {val_loss:.6f}")

    torch.save(model.state_dict(), "ais_model_final.pth")
    print("Model saved to ais_model_final.pth")


# ===========================================================
# PREDICTION & RECONSTRUCTION
# ===========================================================
def predict_sample(idx):
    X = np.load("X.npy")
    meta = pd.read_csv("meta.csv")
    segment_scalers = joblib.load("segment_scalers.pkl")

    mmsi = int(meta.loc[idx, "mmsi"])
    seg = int(meta.loc[idx, "segment"])
    scaler_X, scaler_y = segment_scalers[(mmsi, seg)]

    model = AISTransformer(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load("ais_model_final.pth", map_location=DEVICE))
    model.eval()

    x = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z = model(x)
        preds = []
        h = z
        for _ in range(HORIZON):
            step_pred = model.decoder(h)
            preds.append(step_pred)
        preds = torch.stack(preds, dim=1)[0].cpu().numpy()

    # inverse-scale deltas
    deltas = scaler_y.inverse_transform(preds)

    # reconstruct trajectory
    lat0 = meta.loc[idx, "last_lat"]
    lon0 = meta.loc[idx, "last_lon"]

    pred_lat = lat0 + np.cumsum(deltas[:, 0])
    pred_lon = lon0 + np.cumsum(deltas[:, 1])
    coords = np.stack([pred_lat, pred_lon], axis=-1)

    json_path = f"pred_{idx}.json"
    json.dump({
        "coords": coords.tolist(),
        "deltas": deltas.tolist(),
        "last_lat": lat0,
        "last_lon": lon0
    }, open(json_path, "w"), indent=4)

    return json_path

import folium

def visualize_on_map(actual_coords, predicted_coords, last_lat, last_lon, save_path="ais_map.html"):
    """
    Visualize actual and predicted AIS trajectories on an interactive OpenStreetMap map.

    Parameters:
        actual_coords:      numpy array (HORIZON, 2)  → [[lat, lon], ...]
        predicted_coords:   numpy array (HORIZON, 2)
        last_lat, last_lon: floats, last observed AIS point before prediction
        save_path:          where to save the HTML file
    """

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

    # -------------------------------------------------------
    # Save map
    # -------------------------------------------------------
    m.save(save_path)
    print(f"✔ Map saved to {save_path}")
    print("Open it in your browser to view the map.")


# ===========================================================
# EVALUATION
# ===========================================================
def evaluate(idx):
    y = np.load("y.npy")
    meta = pd.read_csv("meta.csv")
    segment_scalers = joblib.load("segment_scalers.pkl")

    # run prediction
    json_path = predict_sample(idx)
    pred_data = json.load(open(json_path))

    pred_coords = np.array(pred_data["coords"])
    last_lat = pred_data["last_lat"]
    last_lon = pred_data["last_lon"]

    # True deltas
    mmsi = int(meta.loc[idx, "mmsi"])
    seg = int(meta.loc[idx, "segment"])
    _, scaler_y = segment_scalers[(mmsi, seg)]

    true_delta_scaled = y[idx]
    true_deltas = scaler_y.inverse_transform(true_delta_scaled)
    true_lat = last_lat + np.cumsum(true_deltas[:, 0])
    true_lon = last_lon + np.cumsum(true_deltas[:, 1])
    actual_coords = np.stack([true_lat, true_lon], axis=-1)

    # Compute errors
    errors = [
        haversine(actual_coords[t,0], actual_coords[t,1],
                  pred_coords[t,0], pred_coords[t,1])
        for t in range(HORIZON)
    ]

    print("Mean error (m):", np.mean(errors))
    print("Median error (m):", np.median(errors))

    # Matplotlib plot
    plt.figure(figsize=(12,6))
    plt.plot(actual_coords[:,1], actual_coords[:,0], "o-", label="Actual")
    plt.plot(pred_coords[:,1], pred_coords[:,0], "x-", label="Predicted")
    plt.scatter([last_lon], [last_lat], s=80, label="Last observed")
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Predicted vs Actual AIS Trajectory (Final Model)")
    plt.grid()
    plt.show()

    # 🌍 Visualize on map
    visualize_on_map(actual_coords, pred_coords, last_lat, last_lon)

def evaluate_full_ship(mmsi):
    # Load metadata
    meta = pd.read_csv("meta.csv")
    X = np.load("X.npy")
    y = np.load("y.npy")
    segment_scalers = joblib.load("segment_scalers.pkl")

    # Filter windows for this ship
    ship_meta = meta[meta["mmsi"] == mmsi].copy()
    if ship_meta.empty:
        print(f"No windows found for MMSI {mmsi}")
        return

    # Add explicit window index column
    ship_meta["window_index"] = ship_meta.index

    # Sort windows by segment first, then chronological window index
    ship_meta_sorted = ship_meta.sort_values(by=["segment", "window_index"])

    all_errors = []
    all_actual_coords = []
    all_pred_coords = []

    for idx in ship_meta_sorted["window_index"]:
        # Run prediction for each window
        json_path = predict_sample(idx)
        pred_data = json.load(open(json_path))

        pred_coords = np.array(pred_data["coords"])
        last_lat = pred_data["last_lat"]
        last_lon = pred_data["last_lon"]

        # Get target deltas
        seg = int(ship_meta.loc[idx, "segment"])
        _, scaler_y = segment_scalers[(mmsi, seg)]

        true_delta_scaled = y[idx]
        true_deltas = scaler_y.inverse_transform(true_delta_scaled)

        # reconstruct ground truth path for this window
        true_lat = last_lat + np.cumsum(true_deltas[:, 0])
        true_lon = last_lon + np.cumsum(true_deltas[:, 1])
        actual_coords = np.stack([true_lat, true_lon], axis=-1)

        # error per horizon
        for t in range(HORIZON):
            d = haversine(
                actual_coords[t,0], actual_coords[t,1],
                pred_coords[t,0], pred_coords[t,1]
            )
            all_errors.append(d)

        # Collect full route for visualization
        all_actual_coords.extend(actual_coords.tolist())
        all_pred_coords.extend(pred_coords.tolist())

    # Convert to array
    all_actual_coords = np.array(all_actual_coords)
    all_pred_coords = np.array(all_pred_coords)

    # Print results
    print(f"\n====== Full Trajectory Evaluation for MMSI {mmsi} ======")
    print(f"Total windows: {len(ship_meta_sorted)}")
    print(f"Total predicted points: {len(all_errors)}")
    print(f"Mean Error:   {np.mean(all_errors):.2f} m")
    print(f"Median Error: {np.median(all_errors):.2f} m")
    print(f"90th Percentile: {np.percentile(all_errors, 90):.2f} m")
    print("========================================================\n")

    return all_actual_coords, all_pred_coords

def visualize_full_ship_route(actual_coords, pred_coords, mmsi):
    if len(actual_coords) == 0 or len(pred_coords) == 0:
        print("No coords to visualize.")
        return

    # Center on the first actual point
    center_lat = actual_coords[0][0]
    center_lon = actual_coords[0][1]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

    # Actual route
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in actual_coords],
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Actual Full Route"
    ).add_to(m)

    # Predicted route
    folium.PolyLine(
        locations=[[lat, lon] for lat, lon in pred_coords],
        color="red",
        weight=4,
        opacity=0.7,
        tooltip="Predicted Route"
    ).add_to(m)

    # Start/end markers
    folium.Marker(
        [actual_coords[0][0], actual_coords[0][1]],
        popup="Actual Start",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        [actual_coords[-1][0], actual_coords[-1][1]],
        popup="Actual End",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    save_path = f"ship_{mmsi}_route.html"
    m.save(save_path)
    print(f"✔ Map saved to {save_path}")

    m.save(save_path)
    print(f"✔ Full ship route saved to {save_path}")
    print("Open it in your browser.")


def list_available_segments():
    meta = pd.read_csv("meta.csv")
    grouped = meta.groupby(["mmsi", "segment"]).size().reset_index(name="samples")
    print(grouped)
    return grouped

def list_unique_mmsi():
    meta = pd.read_csv("meta.csv")
    unique_mmsi = meta["mmsi"].unique()
    print(unique_mmsi)
    return unique_mmsi

# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    list_available_segments()
    list_unique_mmsi()
    actual, predicted = evaluate_full_ship(255814000)
    #actual, predicted = evaluate_full_ship(205210000)
    #actual, predicted = evaluate_full_ship(257737000)
    visualize_full_ship_route(actual, predicted, 255814000)

    #train_model()
    #evaluate(0)
