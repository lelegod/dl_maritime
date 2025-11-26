import pandas as pd
import numpy as np
import folium
import torch

def create_mmsi_dict_from_file(file_path):
    mmsi_type_dict = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    mmsi_part, type_part = line.split(',', 1)

                    mmsi_key = mmsi_part.split(':', 1)[1].strip()
                    ship_type_value = type_part.split(':', 1)[1].strip()

                    mmsi_type_dict[mmsi_key] = ship_type_value
                    
                except (ValueError, IndexError):
                    print(f"Skipping malformed line: '{line}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return mmsi_type_dict
    
def plot_ship_trajectory(df, mmsi, save_path=None):
    """
    Plots the trajectory of a ship given its MMSI.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['MMSI', 'Latitude', 'Longitude'] columns.
    mmsi : int or str
        The MMSI of the ship to plot.
    save_path : str, optional
        Path to save the interactive HTML map. If None, it will just return the map object. When working with Jupyter, no need for save file.
    
    Returns
    -------
    folium.Map
        Folium map object with the trajectory plotted.
    """
    # Filter for the requested MMSI
    df_ship = df[df['MMSI'] == mmsi].sort_values('Timestamp')

    if df_ship.empty:
        print(f"No data found for MMSI {mmsi}.")
        return None

    # Compute map center
    center_lat = df_ship['Latitude'].mean()
    center_lon = df_ship['Longtitude'].mean()

    # Create folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Plot trajectory line
    coords = list(zip(df_ship['Latitude'], df_ship['Longtitude']))
    folium.PolyLine(coords, color="blue", weight=2.5, opacity=0.8).add_to(m)

    # Optional: add markers for each point
    for _, row in df_ship.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longtitude']],
            radius=2,
            color='red',
            fill=True
        ).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"Map saved to {save_path}")

    return m
    
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- Segment and renumber per MMSI
def segment_and_renumber(df, GAP_BREAK_MIN):
    segmented = []
    for mmsi, g in df.groupby("MMSI", observed=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        dt = g["Timestamp"].diff().dt.total_seconds().fillna(0)
        seg_raw = (dt > GAP_BREAK_MIN * 60).cumsum()
        g["Segment"] = seg_raw - seg_raw.min() + 1
        segmented.append(g)
    return pd.concat(segmented, ignore_index=True)

def filter_stationary_ships(df, radius_threshold=1000, speed_threshold=2.0):
    """
    Removes ships that remain within a small area and never exceed a given speed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must include ['MMSI', 'Latitude', 'Longtitude', 'SOG'] columns.
    radius_threshold : float
        Maximum radius (m) around mean position to consider stationary.
    speed_threshold : float
        Maximum SOG (knots) allowed for stationary ships.
    
    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with stationary ships removed.
    """
    stationary_mmsi = []

    for mmsi, group in df.groupby("MMSI"):
        mean_lat = group["Latitude"].mean()
        mean_lon = group["Longtitude"].mean()
        
        distances = haversine_m(group["Latitude"], group["Longtitude"], mean_lat, mean_lon)
        max_dist = distances.max()
        max_speed = group["SOG"].max()

        if max_dist < radius_threshold and max_speed < speed_threshold:
            stationary_mmsi.append(mmsi)

    print(f"Found {len(stationary_mmsi)} stationary ships out of {df['MMSI'].nunique()}.")

    # Return df with stationary ships removed
    df_clean = df[~df["MMSI"].isin(stationary_mmsi)].copy()

    print(f"Cleaned DF contains {df_clean['MMSI'].nunique()} ships.")
    return df_clean

def plot_ship_trajectory_with_prediction(df_obs, df_pred, mmsi, save_path=None):
    """
    Plots both observed and predicted trajectories for a ship.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observed data containing ['MMSI', 'Latitude', 'Longtitude'].
    df_pred : pd.DataFrame
        Predicted data containing ['MMSI', 'Latitude', 'Longtitude'].
    mmsi : int or str
        MMSI of the ship.
    save_path : str, optional
        Output path for HTML map.

    Returns
    -------
    folium.Map
    """

    # Filter observed data
    df_obs_ship = df_obs[df_obs['MMSI'] == mmsi].sort_values('Timestamp')
    if df_obs_ship.empty:
        print(f"No observed data found for MMSI {mmsi}.")
        return None

    # Filter predicted data
    df_pred_ship = df_pred[df_pred['MMSI'] == mmsi].sort_values('Timestamp')
    if df_pred_ship.empty:
        print(f"No predicted data found for MMSI {mmsi}.")
        return None

    # Compute map center from observed data
    center_lat = df_obs_ship['Latitude'].mean()
    center_lon = df_obs_ship['Longtitude'].mean()

    # Create folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Observed trajectory
    obs_coords = list(zip(df_obs_ship['Latitude'], df_obs_ship['Longtitude']))
    folium.PolyLine(obs_coords, color="blue", weight=3, opacity=0.9,
                    tooltip="Observed Trajectory").add_to(m)

    # Predicted trajectory
    pred_coords = list(zip(df_pred_ship['Latitude'], df_pred_ship['Longtitude']))
    folium.PolyLine(pred_coords, color="green", weight=3, opacity=0.9,
                    tooltip="Predicted Trajectory").add_to(m)

    # Markers for observed points
    for _, row in df_obs_ship.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longtitude']],
            radius=2,
            color='blue',
            fill=True
        ).add_to(m)

    # Optional markers for predicted points (commented out)
    for _, row in df_pred_ship.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longtitude']],
            radius=2,
            color='green',
            fill=True
        ).add_to(m)

    if save_path:
        m.save(save_path)
        print(f"Map saved to {save_path}")

    return m

def create_sequences(data, sequence_length, features, target_features):
    """
    Create sequences for time series prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe for a single segment
    sequence_length : int
        Number of timesteps to use as input
    features : list
        List of feature column names to use as input
    target_features : list
        List of feature column names to predict
        
    Returns:
    --------
    X : np.array
        Input sequences of shape (n_samples, sequence_length, n_features)
    y : np.array
        Target values of shape (n_samples, n_target_features)
    """
    X, y = [], []
    
    data_values = data[features].values
    target_values = data[target_features].values
    
    for i in range(len(data_values) - sequence_length):
        X.append(data_values[i:i+sequence_length])
        y.append(target_values[i+sequence_length])
    
    return np.array(X), np.array(y)


def prepare_training_data(df, sequence_length, features, target_features, min_segment_length):
    """
    Prepare training data from the entire dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned AIS data
    sequence_length : int
        Number of timesteps for input sequences
    features : list
        Input feature names
    target_features : list
        Target feature names
    min_segment_length : int
        Minimum segment length to include
        
    Returns:
    --------
    X : np.array
        All input sequences
    y : np.array
        All target values
    segment_info : list
        Information about each sequence (one entry per sequence in X/y)
    """
    X_all, y_all = [], []
    segment_info = []
    
    for (mmsi, seg), group in df.groupby(["MMSI", "Segment"]):
        # Skip short segments
        if len(group) < min_segment_length:
            continue
            
        # Sort by timestamp to ensure correct order
        group = group.sort_values("Timestamp")
        
        # Create sequences for this segment
        X_seg, y_seg = create_sequences(group, sequence_length, features, target_features)
        
        if len(X_seg) > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
            # Add one entry per sequence (not per segment)
            for i in range(len(X_seg)):
                segment_info.append({
                    'mmsi': mmsi,
                    'segment': seg,
                    'length': len(group),
                    'seq_idx_in_segment': i
                })
    
    # Concatenate all sequences
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    return X, y, segment_info

# Function for iterative prediction
def iterative_predict(model, initial_sequence, n_steps, scaler_X, scaler_y, device):
    """
    Perform multi-step iterative prediction.
    
    Args:
        model: Trained GRU model
        initial_sequence: Input sequence (normalized), shape (seq_len, n_features)
        n_steps: Number of prediction steps
        scaler_X: Input scaler
        scaler_y: Target scaler
        device: PyTorch device
    
    Returns:
        predictions: Array of predictions (original scale), shape (n_steps, n_features)
    """
    model.eval()
    current_sequence = initial_sequence.copy()
    predictions = []
    
    with torch.no_grad():
        for step in range(n_steps):
            # Prepare input tensor
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            
            # Predict next step (normalized)
            pred_normalized = model(input_tensor).cpu().numpy()[0]
            
            # Store prediction (convert to original scale)
            pred_original = scaler_y.inverse_transform(pred_normalized.reshape(1, -1))[0]
            predictions.append(pred_original)
            
            # Update sequence: shift and append prediction
            # The prediction needs to be in normalized form for the next input
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_normalized  # Already normalized
    
    return np.array(predictions)