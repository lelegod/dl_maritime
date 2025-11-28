import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def haversine(coord1, coord2):
    """
    Compute the Haversine distance between two (lat, lon) points in meters.
    coord1, coord2: tuples (lat, lon) in degrees.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees → radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371000  # Earth radius in meters

    return R * c


def latlon_to_xy(lat, lon):
    """
    Convert lat/lon to x/y using simple equirectangular projection.
    For small regions this is fine; for global scale, use pyproj.
    """
    R = 6371000  # Earth radius in meters
    x = np.radians(lon) * R * np.cos(np.radians(lat))
    y = np.radians(lat) * R
    return x, y

def xy_to_latlon(x, y):
    R = 6371000.0
    
    lat = np.degrees(y / R)
    lon = np.degrees(x / (R * np.cos(np.radians(lat))))
    
    return lat, lon

def mass_xy_to_latlon(xy):
    """
    Convert many XY points back to lat/lon using the global
    equirectangular inverse transform, with a simple loop.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape (N, 2) where [:,0] = x, [:,1] = y

    Returns
    -------
    latlon : np.ndarray
        Array of shape (N, 2) where [:,0] = lat, [:,1] = lon
    """
    R = 6371000.0  # Earth radius in meters
    
    latlon_list = []
    
    for i in range(xy.shape[0]):
        x = xy[i, 0]
        y = xy[i, 1]
        
        # latitude in degrees
        lat = (180.0 / math.pi) * (y / R)
        
        # longitude in degrees
        lon = math.degrees(x / (R * math.cos(math.radians(lat))))
        
        latlon_list.append([lat, lon])
    
    return np.array(latlon_list)

def add_xy_and_deltas(df):
    """
    Add x,y coordinates and deltas (dx, dy) per MMSI+segment.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["MMSI", "Segment", "Timestamp"])

    x, y = latlon_to_xy(df["Latitude"].values, df["Longtitude"].values)
    df["x"] = x
    df["y"] = y
    
    # Group by MMSI + segment to compute deltas within each trip
    df["dx"] = df.groupby(["MMSI", "Segment"])["x"].diff().fillna(0)
    df["dy"] = df.groupby(["MMSI", "Segment"])["y"].diff().fillna(0)
    return df

def split_train_val_test(df, TEST_SIZE=0.2, VAL_SIZE=0.1, RANDOM_STATE=42):
    """
    Split by MMSI+segment so trips don't leak across sets.
    """
    # Unique trips
    trips = df[["MMSI", "Segment"]].drop_duplicates()
    
    train_trips, test_trips = train_test_split(
        trips, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_trips, val_trips = train_test_split(
        train_trips, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    
    def subset(trips_set):
        merged = df.merge(trips_set, on=["MMSI", "Segment"])
        return merged
    
    return subset(train_trips), subset(val_trips), subset(test_trips)

def create_sequences(df, input_features, output_features, seq_len=10):
    """
    Create sliding window sequences for autoregressive prediction.
    X = seq_len rows of features
    Y = next row (the seq_len+1-th point)
    """
    sequences_X, sequences_y, metadata = [], [], []
    
    # Group by trip (MMSI + segment)
    for (mmsi, seg), group in df.groupby(["MMSI", "Segment"]):
        group = group.sort_values("Timestamp")
        X_vals = group[input_features].values
        y_vals = group[output_features].values
        
        # Sliding window
        for i in range(len(group) - seq_len):
            seq_X = X_vals[i : i + seq_len]
            seq_y = y_vals[i + seq_len]   # the next point
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)

            metadata.append({
                "MMSI": mmsi,
                "Segment": seg,
                "start_index": group.index[i],
                "end_index": group.index[i + seq_len - 1],
                "target_index": group.index[i + seq_len]
            })
    
    return np.array(sequences_X), np.array(sequences_y), metadata

def autoregressive_predict(model, initial_window, steps):
    window = initial_window.copy()
    outputs = np.zeros((steps, window.shape[1]))  # (steps, features)

    for i in range(steps):
        pred = model(window[np.newaxis, ...], training=False).numpy().squeeze()
        outputs[i] = pred
        window = np.vstack([window[1:], pred])

    return outputs

def reconstruct_positions(deltas, start_point=(0,0)):
    """
    Convert sequence of deltas (dx, dy) back to absolute positions.
    start_point: initial (x,y) coordinate, default (0,0).
    """
    positions = [start_point]
    for dx, dy in deltas:
        last_x, last_y = positions[-1]
        positions.append((last_x + dx, last_y + dy))
    return np.array(positions)

def plot_input_and_predictions(input_deltas, preds, start_point=(0,0)):
    """
    Plot input trajectory and predicted continuation.
    input_deltas: array of input deltas (seq_len x 2)
    preds: array of predicted deltas (N x 2)
    """
    # Reconstruct input trajectory
    input_positions = reconstruct_positions(input_deltas, start_point)
    
    # Start predictions from the last input point
    pred_start = input_positions[-1]
    pred_positions = reconstruct_positions(preds, pred_start)
    
    plt.figure(figsize=(8,6))
    plt.plot(input_positions[:,0], input_positions[:,1], '-o', label="Input sequence", alpha=0.7)
    plt.plot(pred_positions[:,0], pred_positions[:,1], '-o', label="Predicted continuation", alpha=0.7, color="red")
    
    # Mark start and transition point
    plt.scatter(input_positions[0,0], input_positions[0,1], c='green', s=80, label="Start")
    plt.scatter(pred_positions[0,0], pred_positions[0,1], c='orange', marker='x', s=80, label="Prediction start")
    plt.scatter(pred_positions[-1,0], pred_positions[-1,1], c='red', s=80, label="Final prediction")
    
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory: Input vs Predicted Continuation")
    plt.grid(True)
    plt.show()

def folium_plot_trip_with_prediction(
        df, preds, seq_start_idx,
        feature_dx="dx", feature_dy="dy"):
    """
    Plot true 10-point input sequence + predicted continuation.

    df: DataFrame containing at least [Latitude, Longitude, x, y, dx, dy]
    preds: (N x 2) predicted deltas (dx, dy)
    seq_start_idx: index in df where the 10-point input sequence starts
    input_len: number of input points (default = 10)
    """

    # -------------------------------
    # 1. Extract TRUE segment (10 points)
    # -------------------------------
    true_seg = df.iloc[seq_start_idx : seq_start_idx + preds.shape[0]].copy()

    true_coords = list(zip(true_seg["Latitude"], true_seg["Longtitude"]))

    # Starting base position in XY
    base_x, base_y = true_seg.iloc[-1][["x", "y"]]   # last of true trip

    # -------------------------------
    # 2. Convert predicted deltas → XY positions
    # -------------------------------
    # preds is e.g. shape (N, 2) with columns [dx, dy]
    pred_positions = np.zeros((len(preds) + 1, 2))
    pred_positions[0] = [base_x, base_y]

    # Accumulate deltas
    for i in range(len(preds)):
        pred_positions[i+1] = pred_positions[i] + preds[i]

    # Drop the very first element (base point)
    pred_positions = pred_positions[1:]

    # -------------------------------
    # 3. Convert predicted XY → lat/lon
    # -------------------------------
    ref_lat = true_seg["Latitude"].mean()

    pred_lat, pred_lon = xy_to_latlon(
        pred_positions[:, 0], pred_positions[:, 1], ref_lat
    )

    pred_coords = list(zip(pred_lat, pred_lon))

    # -------------------------------
    # 4. Build the Folium map
    # -------------------------------
    m = folium.Map(location=true_coords[0], zoom_start=10)

    # True blue line
    folium.PolyLine(true_coords, color="blue", weight=4,
                    opacity=0.8, tooltip="True Input").add_to(m)

    # Pred red line
    folium.PolyLine(pred_coords, color="red", weight=4,
                    opacity=0.8, tooltip="Predicted").add_to(m)

    # Markers
    folium.Marker(true_coords[0],
                  popup="Start", icon=folium.Icon(color="green")).add_to(m)

    folium.Marker(true_coords[-1],
                  popup="Last True (Prediction base)",
                  icon=folium.Icon(color="blue")).add_to(m)

    folium.Marker(pred_coords[-1],
                  popup="Predicted End",
                  icon=folium.Icon(color="red")).add_to(m)

    return m

def normalized_folium_plot_trip_with_prediction(
        df, preds_norm, seq_start_idx,
        dxy_scaler,
        input_len=10):
    """
    Plot true 10-point input sequence and predicted continuation.
    
    df: DataFrame with [Latitude, Longtitude, x, y, dx, dy]
    preds_norm: normalized predicted deltas (N x 2)
    seq_start_idx: index where the 10-point true input sequence begins
    dxy_scaler: scaler used for dx, dy normalization (for inverse_transform)
    input_len: number of true points to plot before predictions
    """

    # -------------------------------------------------
    # 1. Extract TRUE segment (10 real points)
    # -------------------------------------------------
    true_seg = df.iloc[seq_start_idx : seq_start_idx + input_len].copy()

    true_coords = list(zip(true_seg["Latitude"], true_seg["Longtitude"]))

    # Base XY point = last true position
    base_x, base_y = true_seg.iloc[-1][["x", "y"]]

    # -------------------------------------------------
    # 2. INVERSE TRANSFORM predicted deltas
    # -------------------------------------------------
    # preds_norm shape (N,2) → real dx, dy
    preds_real = dxy_scaler.inverse_transform(preds_norm)

    # -------------------------------------------------
    # 3. Convert predicted deltas → predicted XY positions
    # -------------------------------------------------
    pred_positions = np.zeros((len(preds_real) + 1, 2))
    pred_positions[0] = [base_x, base_y]

    # accumulate dx, dy
    for i in range(len(preds_real)):
        pred_positions[i + 1] = pred_positions[i] + preds_real[i]

    # drop first base point → pure predictions
    pred_positions = pred_positions[1:]

    # -------------------------------------------------
    # 4. Convert predicted XY → lat/lon
    # -------------------------------------------------
    ref_lat = true_seg["Latitude"].mean()

    pred_lat, pred_lon = xy_to_latlon(
        pred_positions[:, 0],
        pred_positions[:, 1],
        ref_lat
    )

    pred_coords = list(zip(pred_lat, pred_lon))

    # -------------------------------------------------
    # 5. Build Folium Map
    # -------------------------------------------------
    m = folium.Map(location=true_coords[0], zoom_start=12)

    # True path (blue)
    folium.PolyLine(
        true_coords, color="blue", weight=4,
        opacity=0.8, tooltip="True 10-point Input"
    ).add_to(m)

    # Prediction (red)
    folium.PolyLine(
        pred_coords, color="red", weight=4,
        opacity=0.8, tooltip="Predicted"
    ).add_to(m)

    # Markers
    folium.Marker(
        true_coords[0],
        popup="Start", icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        true_coords[-1],
        popup="Prediction Base (last true)",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    folium.Marker(
        pred_coords[-1],
        popup="Predicted End",
        icon=folium.Icon(color="red")
    ).add_to(m)

    return m

def compute_errors(true_latlon, pred_latlon):
    """
    Compute haversine distances between true and predicted points.
    """
    errors = [haversine(tuple(true_latlon[i]), tuple(pred_latlon[i])) for i in range(len(true_latlon))]
    return {
        "mean_error_km": np.mean(errors),
        "rmse_km": np.sqrt(mean_squared_error([0]*len(errors), errors)),
        "max_error_km": np.max(errors),
        "errors": errors
    }