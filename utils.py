import pandas as pd
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
