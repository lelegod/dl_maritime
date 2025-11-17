import pandas as pd
import numpy as np
import folium

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

