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
    df_ship = df[df['MMSI'] == mmsi].sort_values('Timestamp')
    if df_ship.empty:
        print(f"No data found for MMSI {mmsi}.")
        return None
    center_lat = df_ship['Latitude'].mean()
    center_lon = df_ship['Longtitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    coords = list(zip(df_ship['Latitude'], df_ship['Longtitude']))
    folium.PolyLine(coords, color="blue", weight=2.5, opacity=0.8).add_to(m)
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
    df_clean = df[~df["MMSI"].isin(stationary_mmsi)].copy()
    print(f"Cleaned DF contains {df_clean['MMSI'].nunique()} ships.")
    return df_clean

def plot_ship_trajectory_with_prediction(df_obs, df_pred, mmsi, save_path=None):
    df_obs_ship = df_obs[df_obs['MMSI'] == mmsi].sort_values('Timestamp')
    if df_obs_ship.empty:
        print(f"No observed data found for MMSI {mmsi}.")
        return None
    df_pred_ship = df_pred[df_pred['MMSI'] == mmsi].sort_values('Timestamp')
    if df_pred_ship.empty:
        print(f"No predicted data found for MMSI {mmsi}.")
        return None
    center_lat = df_obs_ship['Latitude'].mean()
    center_lon = df_obs_ship['Longtitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    obs_coords = list(zip(df_obs_ship['Latitude'], df_obs_ship['Longtitude']))
    folium.PolyLine(obs_coords, color="blue", weight=3, opacity=0.9,
                    tooltip="Observed Trajectory").add_to(m)
    pred_coords = list(zip(df_pred_ship['Latitude'], df_pred_ship['Longtitude']))
    folium.PolyLine(pred_coords, color="green", weight=3, opacity=0.9,
                    tooltip="Predicted Trajectory").add_to(m)
    for _, row in df_obs_ship.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longtitude']],
            radius=2,
            color='blue',
            fill=True
        ).add_to(m)
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
    X, y = [], []
    data_values = data[features].values
    target_values = data[target_features].values
    for i in range(len(data_values) - sequence_length):
        X.append(data_values[i:i+sequence_length])
        y.append(target_values[i+sequence_length])
    return np.array(X), np.array(y)

def prepare_training_data(df, sequence_length, features, target_features, min_segment_length):
    X_all, y_all = [], []
    segment_info = []
    for (mmsi, seg), group in df.groupby(["MMSI", "Segment"]):
        if len(group) < min_segment_length:
            continue
        group = group.sort_values("Timestamp")
        X_seg, y_seg = create_sequences(group, sequence_length, features, target_features)
        if len(X_seg) > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
            for i in range(len(X_seg)):
                segment_info.append({
                    'mmsi': mmsi,
                    'segment': seg,
                    'length': len(group),
                    'seq_idx_in_segment': i
                })
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, segment_info

def prepare_delta_sequences(df, seq_length, features, target_features, min_length):
    X_all, y_all = [], []
    segment_info = []
    for (mmsi, seg), group in df.groupby(['MMSI', 'Segment']):
        if len(group) < min_length:
            continue
        group = group.sort_values('Timestamp')
        X_seg, y_seg = create_sequences(group, seq_length, features, target_features)
        if len(X_seg) > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
            for i in range(len(X_seg)):
                segment_info.append({
                    'mmsi': mmsi,
                    'segment': seg,
                    'length': len(group),
                    'seq_idx_in_segment': i
                })
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, segment_info

def iterative_predict(model, initial_sequence, n_steps, scaler_X, scaler_y, device):
    model.eval()
    current_sequence = initial_sequence.copy()
    predictions = []
    with torch.no_grad():
        for step in range(n_steps):
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            pred_normalized_y = model(input_tensor).cpu().numpy()[0]
            pred_original = scaler_y.inverse_transform(pred_normalized_y.reshape(1, -1))[0]
            predictions.append(pred_original)
            pred_normalized_x = scaler_X.transform(pred_original.reshape(1, -1))[0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_normalized_x
    return np.array(predictions)

def iterative_predict_delta(model, initial_sequence, n_steps, scaler_X, scaler_y, device, 
                            last_lat, last_lon):
    model.eval()
    current_sequence = initial_sequence.copy()
    predictions_delta = []
    predictions_abs = []
    current_lat = last_lat
    current_lon = last_lon
    with torch.no_grad():
        for step in range(n_steps):
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            pred_normalized_y = model(input_tensor).cpu().numpy()[0]
            pred_delta = scaler_y.inverse_transform(pred_normalized_y.reshape(1, -1))[0]
            predictions_delta.append(pred_delta)
            new_lat = current_lat + pred_delta[0]
            new_lon = current_lon + pred_delta[1]
            predictions_abs.append([new_lat, new_lon, pred_delta[2], pred_delta[3]])
            current_lat = new_lat
            current_lon = new_lon
            pred_normalized_x = scaler_X.transform(pred_delta.reshape(1, -1))[0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_normalized_x
    return np.array(predictions_abs), np.array(predictions_delta)

def iterative_predict_delta_cog_sincos(model, initial_sequence, n_steps, scaler_X, scaler_y, device, 
                                        last_lat, last_lon):
    model.eval()
    current_sequence = initial_sequence.copy()
    predictions_raw = []
    predictions_abs = []
    current_lat = last_lat
    current_lon = last_lon
    with torch.no_grad():
        for step in range(n_steps):
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            pred_normalized_y = model(input_tensor).cpu().numpy()[0]
            pred_original = scaler_y.inverse_transform(pred_normalized_y.reshape(1, -1))[0]
            predictions_raw.append(pred_original)
            delta_lat = pred_original[0]
            delta_lon = pred_original[1]
            sog = pred_original[2]
            cog_sin = pred_original[3]
            cog_cos = pred_original[4]
            cog_degrees = np.rad2deg(np.arctan2(cog_sin, cog_cos))
            if cog_degrees < 0:
                cog_degrees += 360
            new_lat = current_lat + delta_lat
            new_lon = current_lon + delta_lon
            predictions_abs.append([new_lat, new_lon, sog, cog_degrees])
            current_lat = new_lat
            current_lon = new_lon
            pred_normalized_x = scaler_X.transform(pred_original.reshape(1, -1))[0]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_normalized_x
    return np.array(predictions_abs), np.array(predictions_raw)

def iterative_predict_delta_cog_sincos_complex(model, initial_sequence, n_steps, scaler_X, scaler_y, device,
                                       last_lat, last_lon):
    model.eval()
    encoder_input = torch.FloatTensor(initial_sequence).unsqueeze(0).to(device)
    try:
        max_target_len = max(n_steps, model.future_step)
    except AttributeError:
        max_target_len = n_steps
    batch_size = 1
    output_dim = initial_sequence.shape[1]
    target_input_for_inference = torch.zeros(
        batch_size,
        max_target_len,
        output_dim
    ).to(device)
    predictions_raw = []
    predictions_abs = []
    current_lat = last_lat
    current_lon = last_lon
    with torch.no_grad():
        for t in range(n_steps):
            outputs = model(encoder_input, target_input_for_inference)
            predicted_step_t = outputs[:, t, :]
            if t + 1 < n_steps:
                target_input_for_inference[:, t + 1, :] = predicted_step_t.clone()
            pred_normalized_y = predicted_step_t.cpu().numpy()[0]
            pred_original = scaler_y.inverse_transform(pred_normalized_y.reshape(1, -1))[0]
            predictions_raw.append(pred_original)
            delta_lat = pred_original[0]
            delta_lon = pred_original[1]
            sog = pred_original[2]
            cog_sin = pred_original[3]
            cog_cos = pred_original[4]
            cog_degrees = np.rad2deg(np.arctan2(cog_sin, cog_cos))
            if cog_degrees < 0:
                cog_degrees += 360
            new_lat = current_lat + delta_lat
            new_lon = current_lon + delta_lon
            predictions_abs.append([new_lat, new_lon, sog, cog_degrees])
            current_lat = new_lat
            current_lon = new_lon
    return np.array(predictions_abs), np.array(predictions_raw)

def create_seq2seq_sequences(df, seq_len, pred_horizon, input_features, target_features, min_length):
    X_list = []
    y_list = []
    segment_info = []
    for (mmsi, seg), g in df.groupby(['MMSI', 'Segment']):
        if len(g) < min_length:
            continue
        data_input = g[input_features].values
        data_target = g[target_features].values
        for i in range(len(g) - seq_len - pred_horizon + 1):
            X_list.append(data_input[i:i + seq_len])
            y_list.append(data_target[i + seq_len:i + seq_len + pred_horizon])
            segment_info.append({
                'mmsi': mmsi,
                'segment': seg,
                'seq_idx_in_segment': i,
                'length': len(g)
            })
    return np.array(X_list), np.array(y_list), segment_info