import folium
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import random


def load_model_and_predict(model_path: str, model_class, input_tensor, scaler_y, target_features, prediction_horizon, device):
    model = model_class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        seq2seq_pred = model(input_tensor, teacher_forcing_ratio=0.0)
    
    pred_raw = scaler_y.inverse_transform(
        seq2seq_pred.cpu().numpy().reshape(-1, len(target_features))
    ).reshape(prediction_horizon, len(target_features))
    
    return pred_raw


def deltas_to_positions(pred_raw, last_lat, last_lon, last_cog, prediction_horizon):
    pred_lats = [last_lat]
    pred_lons = [last_lon]
    pred_cogs = [last_cog]
    
    for t in range(prediction_horizon):
        pred_lats.append(pred_lats[-1] + pred_raw[t, 0])
        pred_lons.append(pred_lons[-1] + pred_raw[t, 1])
        new_cog = (pred_cogs[-1] + pred_raw[t, 3]) % 360
        pred_cogs.append(new_cog)
    
    return pred_lats, pred_lons, pred_cogs


def compare_multiple_models(
    model_configs: List[Dict],
    X_test: np.ndarray,
    test_indices: List[int],
    segment_info: List[Dict],
    mmsi_test_set: set,
    df_delta: pd.DataFrame,
    original_positions_test: pd.DataFrame,
    scaler_y,
    target_features: List[str],
    prediction_horizon: int,
    sequence_length: int,
    interval: int,
    device,
    random_mmsi: int = None,
    save_path: str = None
):
    print(f"Comparing {len(model_configs)} models")
    print("="*60)
    
    if random_mmsi is None:
        test_mmsis = list(mmsi_test_set)
        random_mmsi = random.choice(test_mmsis)
    
    ship_test_indices = [i for i, seg in enumerate(segment_info) 
                        if seg['mmsi'] == random_mmsi and i in test_indices]
    
    if len(ship_test_indices) == 0:
        print(f"No test sequences found for MMSI {random_mmsi}")
        return None
    
    first_seq_orig_idx = ship_test_indices[0]
    first_seq_test_idx = test_indices.index(first_seq_orig_idx)
    seg_info = segment_info[first_seq_orig_idx]
    
    print(f"Using MMSI: {random_mmsi}, Segment: {seg_info['segment']}")
    
    input_sequence = X_test[first_seq_test_idx:first_seq_test_idx+1]
    input_tensor = torch.FloatTensor(input_sequence).to(device)
    
    df_ship_segment = df_delta[(df_delta['MMSI'] == random_mmsi) & 
                                (df_delta['Segment'] == seg_info['segment'])].sort_values('Timestamp')
    
    seq_start_idx = seg_info['seq_idx_in_segment']
    seq_end_idx = seq_start_idx + sequence_length
    gt_end_idx = min(seq_end_idx + prediction_horizon, len(df_ship_segment))
    
    df_observed = df_ship_segment.iloc[seq_start_idx:gt_end_idx].copy()
    last_input_point = df_ship_segment.iloc[seq_end_idx - 1]
    
    last_lat = original_positions_test.iloc[first_seq_test_idx]['last_lat']
    last_lon = original_positions_test.iloc[first_seq_test_idx]['last_lon']
    last_cog = original_positions_test.iloc[first_seq_test_idx]['last_cog']
    
    center_lat = df_observed['Latitude'].mean()
    center_lon = df_observed['Longtitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    observed_coords = df_observed[['Latitude', 'Longtitude']].values.tolist()
    folium.PolyLine(
        observed_coords,
        color='blue',
        weight=3,
        opacity=0.8
    ).add_to(m)
    
    gt_future_coords = df_observed.iloc[sequence_length:][['Latitude', 'Longtitude']].values.tolist()
    if len(gt_future_coords) > 0:
        folium.PolyLine(
            gt_future_coords,
            color='black',
            weight=3,
            opacity=0.6,
            dash_array='10',
            popup='Ground Truth'
        ).add_to(m)
    
    all_predictions = {}
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['model_path']
        model_instance = config['model_instance']
        color = config.get('color', 'red')
        
        print(f"\nProcessing {model_name}...")
        
        try:
            pred_raw = load_model_and_predict(
                model_path, model_instance, input_tensor, 
                scaler_y, target_features, prediction_horizon, device
            )
            
            pred_lats, pred_lons, pred_cogs = deltas_to_positions(
                pred_raw, last_lat, last_lon, last_cog, prediction_horizon
            )
            
            all_predictions[model_name] = {
                'lats': pred_lats,
                'lons': pred_lons,
                'cogs': pred_cogs,
                'color': color
            }
            
            pred_coords = [[pred_lats[i], pred_lons[i]] for i in range(len(pred_lats))]
            folium.PolyLine(
                pred_coords,
                color=color,
                weight=2.5,
                opacity=0.7,
                dash_array='5, 10'
            ).add_to(m)
            
            if len(gt_future_coords) > 0:
                from utils import haversine_m
                gt_future = df_ship_segment.iloc[seq_end_idx:gt_end_idx][['Latitude', 'Longtitude']].values
                pred_future = np.column_stack([pred_lats[1:len(gt_future)+1], pred_lons[1:len(gt_future)+1]])
                
                if len(pred_future) > 0 and len(gt_future) > 0:
                    min_len = min(len(pred_future), len(gt_future))
                    distances = haversine_m(
                        pred_future[:min_len, 0], pred_future[:min_len, 1], 
                        gt_future[:min_len, 0], gt_future[:min_len, 1]
                    )
                    mean_error = np.mean(distances)
                    print(f"  {model_name} - Mean Error: {mean_error:.1f} m")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {str(e)}")
            continue
    
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px">
    <p style="margin:0"><b>Trajectory Legend</b></p>
    <p style="margin:5px 0"><span style="color:blue">━━━</span> Observed</p>
    <p style="margin:5px 0"><span style="color:black">╍╍╍</span> Ground Truth</p>
    '''
    
    for model_name, pred_data in all_predictions.items():
        color = pred_data['color']
        legend_html += f'<p style="margin:5px 0"><span style="color:{color}">━━━</span> {model_name}</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_path:
        m.save(save_path)
        print(f"\nMap saved to: {save_path}")
    
    print("\n" + "="*60)
    print(f"Comparison complete! {len(all_predictions)} models visualized.")
    
    return m


def multi_horizon_iterative_prediction(
    model_instance,
    initial_sequence: np.ndarray,
    last_lat: float,
    last_lon: float,
    last_cog: float,
    last_sog: float,
    max_steps: int,
    prediction_horizon: int,
    scaler_y,
    scaler_lat_lon,
    target_features: List[str],
    input_features: List[str],
    device
):
    model_instance.eval()
    
    current_sequence = initial_sequence.copy()
    
    pred_lats = [last_lat]
    pred_lons = [last_lon]
    pred_cogs = [last_cog]
    pred_sogs = [last_sog]
    
    num_iterations = (max_steps + prediction_horizon - 1) // prediction_horizon
    
    for iteration in range(num_iterations):
        with torch.no_grad():
            input_tensor = torch.FloatTensor(current_sequence).to(device)
            seq2seq_pred = model_instance(input_tensor, teacher_forcing_ratio=0.0)
        
        pred_batch = scaler_y.inverse_transform(
            seq2seq_pred.cpu().numpy().reshape(-1, len(target_features))
        ).reshape(prediction_horizon, len(target_features))
        
        steps_to_use = min(prediction_horizon, max_steps - iteration * prediction_horizon)
        
        for step in range(steps_to_use):
            delta_lat = pred_batch[step, 0]
            delta_lon = pred_batch[step, 1]
            next_sog = pred_batch[step, 2]
            delta_cog = pred_batch[step, 3]
            
            next_lat = pred_lats[-1] + delta_lat
            next_lon = pred_lons[-1] + delta_lon
            next_cog = (pred_cogs[-1] + delta_cog) % 360
            
            pred_lats.append(next_lat)
            pred_lons.append(next_lon)
            pred_cogs.append(next_cog)
            pred_sogs.append(next_sog)
        
        if iteration < num_iterations - 1:
            new_inputs = []
            for step in range(prediction_horizon):
                idx = len(pred_lats) - prediction_horizon + step
                lat_lon_normalized = scaler_lat_lon.transform([[pred_lats[idx], pred_lons[idx]]])[0]
                
                if idx > 0:
                    delta_lat_seq = pred_lats[idx] - pred_lats[idx-1]
                    delta_lon_seq = pred_lons[idx] - pred_lons[idx-1]
                    delta_cog_seq = (pred_cogs[idx] - pred_cogs[idx-1]) % 360
                    if delta_cog_seq > 180:
                        delta_cog_seq -= 360
                else:
                    delta_lat_seq = 0
                    delta_lon_seq = 0
                    delta_cog_seq = 0
                
                new_input = np.array([
                    lat_lon_normalized[0],
                    lat_lon_normalized[1],
                    delta_lat_seq,
                    delta_lon_seq,
                    pred_sogs[idx],
                    delta_cog_seq
                ])
                new_inputs.append(new_input)
            
            new_inputs = np.array(new_inputs)
            current_sequence = np.concatenate([
                current_sequence[0, prediction_horizon:, :],
                new_inputs
            ], axis=0).reshape(1, initial_sequence.shape[1], len(input_features))
    
    return pred_lats, pred_lons, pred_cogs, pred_sogs


def compare_multiple_models_multihorizon(
    model_configs: List[Dict],
    X_test: np.ndarray,
    test_indices: List[int],
    segment_info: List[Dict],
    mmsi_test_set: set,
    df_delta: pd.DataFrame,
    original_positions_test: pd.DataFrame,
    scaler_y,
    scaler_lat_lon,
    target_features: List[str],
    input_features: List[str],
    prediction_horizon: int,
    sequence_length: int,
    interval: int,
    device,
    max_steps: int = 30,
    random_mmsi: Optional[int] = None,
    save_path: Optional[str] = None
):
    print(f"Multi-Horizon Model Comparison ({max_steps * interval} minutes)")
    print("="*60)
    
    min_required_points = sequence_length + max_steps
    
    valid_test_indices = []
    for i in test_indices:
        seg_info = segment_info[i]
        if seg_info['length'] >= min_required_points:
            seq_start = seg_info['seq_idx_in_segment']
            if seq_start + sequence_length + max_steps <= seg_info['length']:
                valid_test_indices.append(i)
    
    if len(valid_test_indices) == 0:
        print(f"No test sequences with {max_steps * interval}+ minutes of future data found")
        return None
    
    if random_mmsi is not None:
        mmsi_valid = [i for i in valid_test_indices if segment_info[i]['mmsi'] == random_mmsi]
        if len(mmsi_valid) == 0:
            print(f"No valid sequences found for MMSI {random_mmsi}")
            return None
        selected_orig_idx = random.choice(mmsi_valid)
    else:
        selected_orig_idx = random.choice(valid_test_indices)
    
    selected_test_idx = test_indices.index(selected_orig_idx)
    seg_info = segment_info[selected_orig_idx]
    selected_mmsi = seg_info['mmsi']
    
    print(f"Selected MMSI: {selected_mmsi}, Segment: {seg_info['segment']}")
    print(f"Segment length: {seg_info['length']} points ({seg_info['length'] * interval} min)")
    
    df_ship_segment = df_delta[(df_delta['MMSI'] == selected_mmsi) & 
                                (df_delta['Segment'] == seg_info['segment'])].sort_values('Timestamp')
    
    seq_start_idx = seg_info['seq_idx_in_segment']
    seq_end_idx = seq_start_idx + sequence_length
    
    df_observed = df_ship_segment.iloc[seq_start_idx:seq_end_idx].copy()
    last_input_point = df_ship_segment.iloc[seq_end_idx - 1]
    df_ground_truth = df_ship_segment.iloc[seq_end_idx:seq_end_idx + max_steps].copy()
    
    print(f"Observed: {len(df_observed)} points ({sequence_length * interval} min)")
    print(f"Ground truth: {len(df_ground_truth)} points ({len(df_ground_truth) * interval} min)")
    
    initial_sequence = X_test[selected_test_idx:selected_test_idx+1]
    last_lat = original_positions_test.iloc[selected_test_idx]['last_lat']
    last_lon = original_positions_test.iloc[selected_test_idx]['last_lon']
    last_cog = original_positions_test.iloc[selected_test_idx]['last_cog']
    last_sog = last_input_point['SOG']
    
    center_lat = df_ship_segment['Latitude'].mean()
    center_lon = df_ship_segment['Longtitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    observed_coords = df_observed[['Latitude', 'Longtitude']].values.tolist()
    folium.PolyLine(
        observed_coords,
        color='blue',
        weight=5,
        opacity=0.8
    ).add_to(m)
    
    gt_coords = df_ground_truth[['Latitude', 'Longtitude']].values.tolist()
    if len(gt_coords) > 0:
        folium.PolyLine(
            gt_coords,
            color='black',
            weight=5,
            opacity=0.6,
            dash_array='10',
            popup='Ground Truth'
        ).add_to(m)
    
    all_predictions = {}
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['model_path']
        model_instance = config['model_instance']
        color = config.get('color', 'red')
        
        print(f"\nProcessing {model_name}...")
        
        try:
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
            
            pred_lats, pred_lons, pred_cogs, pred_sogs = multi_horizon_iterative_prediction(
                model_instance=model_instance,
                initial_sequence=initial_sequence,
                last_lat=last_lat,
                last_lon=last_lon,
                last_cog=last_cog,
                last_sog=last_sog,
                max_steps=max_steps,
                prediction_horizon=prediction_horizon,
                scaler_y=scaler_y,
                scaler_lat_lon=scaler_lat_lon,
                target_features=target_features,
                input_features=input_features,
                device=device
            )
            
            all_predictions[model_name] = {
                'lats': pred_lats,
                'lons': pred_lons,
                'cogs': pred_cogs,
                'sogs': pred_sogs,
                'color': color
            }
            
            pred_coords = [[pred_lats[i], pred_lons[i]] for i in range(len(pred_lats))]
            folium.PolyLine(
                pred_coords,
                color=color,
                weight=4,
                opacity=0.8,
                dash_array='5, 10'
            ).add_to(m)
            
            from utils import haversine_m
            all_errors = []
            for idx in range(1, min(len(pred_lats), len(df_ground_truth) + 1)):
                gt_lat = df_ground_truth.iloc[idx-1]['Latitude']
                gt_lon = df_ground_truth.iloc[idx-1]['Longtitude']
                pred_lat = pred_lats[idx]
                pred_lon = pred_lons[idx]
                error = haversine_m(pred_lat, pred_lon, gt_lat, gt_lon)
                all_errors.append(error)
            
            if len(all_errors) > 0:
                print(f"  {model_name} - Mean Error: {np.mean(all_errors):>8.1f} m")
                print(f"             - Median Error: {np.median(all_errors):>8.1f} m")
                print(f"             - Max Error: {np.max(all_errors):>8.1f} m")
            
        except Exception as e:
            print(f"  Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: white; z-index:9999; font-size:18px;
                border:2px solid grey; padding: 12px">
    <p style="margin:6px 0"><span style="color:blue">━━━</span> Observed</p>
    <p style="margin:6px 0"><span style="color:black">╍╍╍</span> Ground Truth</p>
    '''
    
    for model_name, pred_data in all_predictions.items():
        color = pred_data['color']
        legend_html += f'<p style="margin:6px 0"><span style="color:{color}">━━━</span> {model_name}</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_path:
        m.save(save_path)
        print(f"\nMap saved to: {save_path}")
    
    print("\n" + "="*60)
    print(f"Multi-horizon comparison complete! {len(all_predictions)} models visualized.")
    
    return m
