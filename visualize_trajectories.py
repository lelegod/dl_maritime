import pandas as pd
import folium
from folium import plugins
import numpy as np
import os


def generate_colors(n):
    """Generate n distinct colors for segments"""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors


def plot_trajectories_by_segment(csv_file, output_html="trajectory_map.html", 
                                   sample_mmsi=None, max_segments=None):
    """
    Visualize ship trajectories from CSV file, color-coded by segments.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing AIS data with columns: MMSI, Segment, Timestamp, Latitude, Longitude
    output_html : str
        Output HTML file path for the interactive map
    sample_mmsi : list, optional
        List of specific MMSIs to plot. If None, plot all vessels.
    max_segments : int, optional
        Maximum number of segments to plot per vessel. If None, plot all segments.
    
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    print("="*70)
    print("TRAJECTORY VISUALIZATION BY SEGMENTS")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check required columns
    required_cols = ['MMSI', 'Segment', 'Timestamp', 'Latitude', 'Longtitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try alternative spellings
        if 'Longitude' in df.columns:
            df = df.rename(columns={'Longitude': 'Longtitude'})
        else:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['MMSI', 'Segment', 'Timestamp'])
    
    print(f"  Total records: {len(df):,}")
    print(f"  Unique vessels (MMSI): {df['MMSI'].nunique():,}")
    print(f"  Total segments: {df.groupby(['MMSI', 'Segment']).ngroups:,}")
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Filter by MMSI if specified
    if sample_mmsi is not None:
        print(f"\nFiltering to {len(sample_mmsi)} specific vessels...")
        df = df[df['MMSI'].isin(sample_mmsi)]
        if len(df) == 0:
            print("  WARNING: No data found for specified MMSIs!")
            return None
    
    # Calculate center of map
    center_lat = df['Latitude'].mean()
    center_lon = df['Longtitude'].mean()
    
    print(f"\nMap center: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Create base map
    print("\nCreating map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Group trajectories by MMSI and Segment
    grouped = df.groupby(['MMSI', 'Segment'])
    total_segments = len(grouped)
    
    print(f"  Total segments to plot: {total_segments}")
    
    # Limit segments if specified
    if max_segments and total_segments > max_segments:
        print(f"  Limiting to first {max_segments} segments...")
        groups_to_plot = list(grouped.groups.keys())[:max_segments]
    else:
        groups_to_plot = list(grouped.groups.keys())
    
    # Generate colors for each vessel
    unique_mmsis = df['MMSI'].unique()
    vessel_colors = {mmsi: color for mmsi, color in zip(unique_mmsis, generate_colors(len(unique_mmsis)))}
    
    # Create feature groups for each vessel
    vessel_groups = {}
    for mmsi in unique_mmsis:
        vessel_groups[mmsi] = folium.FeatureGroup(name=f'MMSI: {mmsi}')
    
    # Plot each segment
    segments_plotted = 0
    for (mmsi, segment) in groups_to_plot:
        segment_data = grouped.get_group((mmsi, segment))
        
        if len(segment_data) < 2:
            continue
        
        # Get coordinates
        coords = segment_data[['Latitude', 'Longtitude']].values.tolist()
        
        # Get segment info
        start_time = segment_data['Timestamp'].iloc[0]
        end_time = segment_data['Timestamp'].iloc[-1]
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        # Add polyline for trajectory
        polyline = folium.PolyLine(
            coords,
            color=vessel_colors[mmsi],
            weight=3,
            opacity=0.7,
            popup=f"""
            <b>MMSI:</b> {mmsi}<br>
            <b>Segment:</b> {segment}<br>
            <b>Points:</b> {len(segment_data)}<br>
            <b>Start:</b> {start_time.strftime('%Y-%m-%d %H:%M')}<br>
            <b>End:</b> {end_time.strftime('%Y-%m-%d %H:%M')}<br>
            <b>Duration:</b> {duration:.1f} hours
            """,
            tooltip=f"MMSI {mmsi} - Segment {segment}"
        )
        polyline.add_to(vessel_groups[mmsi])
        
        # Add start marker (green)
        folium.CircleMarker(
            location=[segment_data['Latitude'].iloc[0], segment_data['Longtitude'].iloc[0]],
            radius=5,
            color='green',
            fill=True,
            fillColor='green',
            fillOpacity=0.8,
            popup=f"Start - MMSI {mmsi} Seg {segment}",
            tooltip="Start"
        ).add_to(vessel_groups[mmsi])
        
        # Add end marker (red)
        folium.CircleMarker(
            location=[segment_data['Latitude'].iloc[-1], segment_data['Longtitude'].iloc[-1]],
            radius=5,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.8,
            popup=f"End - MMSI {mmsi} Seg {segment}",
            tooltip="End"
        ).add_to(vessel_groups[mmsi])
        
        segments_plotted += 1
    
    # Add all vessel groups to map
    for mmsi in unique_mmsis:
        vessel_groups[mmsi].add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure control
    plugins.MeasureControl(position='topleft').add_to(m)
    
    # Add mouse position
    plugins.MousePosition().add_to(m)
    
    # Save map
    print(f"\nSaving map to: {output_html}")
    m.save(output_html)
    file_size_mb = os.path.getsize(output_html) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Vessels plotted: {len(unique_mmsis)}")
    print(f"Segments plotted: {segments_plotted}")
    print(f"Output: {output_html}")
    print("="*70)
    print("\n✅ Visualization complete! Open the HTML file in a browser.")
    
    return m


def plot_single_vessel_segments(csv_file, mmsi, output_html=None):
    """
    Plot all segments for a single vessel.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    mmsi : int or str
        MMSI of the vessel to plot
    output_html : str, optional
        Output HTML file path. If None, auto-generate filename.
    """
    if output_html is None:
        output_html = f"trajectory_mmsi_{mmsi}.html"
    
    return plot_trajectories_by_segment(csv_file, output_html, sample_mmsi=[mmsi])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize ship trajectories by segments')
    parser.add_argument('csv_file', help='Path to CSV file with AIS data')
    parser.add_argument('-o', '--output', default='trajectory_map.html', 
                       help='Output HTML file (default: trajectory_map.html)')
    parser.add_argument('-m', '--mmsi', nargs='+', type=int,
                       help='Specific MMSI(s) to plot (default: all vessels)')
    parser.add_argument('-l', '--limit', type=int,
                       help='Maximum number of segments to plot')
    
    args = parser.parse_args()
    
    # Example usage:
    # python visualize_trajectories.py data/ais_combined_merged.csv
    # python visualize_trajectories.py data/ais_combined_merged.csv -m 219000358 219001684
    # python visualize_trajectories.py data/ais_combined_merged.csv -l 50
    
    plot_trajectories_by_segment(
        args.csv_file,
        output_html=args.output,
        sample_mmsi=args.mmsi,
        max_segments=args.limit
    )
