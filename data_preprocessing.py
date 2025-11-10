import pandas as pd
import numpy as np

DATA_PATH = "data/ais_combined.csv"
OUTPUT_PATH = "data/ais_data_1min_clean.csv"

GAP_BREAK_MIN = 10          # minutes to start a new segment
INTERP_LIMIT_MIN = 5        # interpolate gaps up to 5 minutes
MAX_DISTANCE_M = 3000       # ~97 knots
MAX_SOG_KNOTS = 40
NUM_COLS = ["SOG", "COG", "Longtitude", "Latitude"]
# ----------------------------------------

# --- Helper: haversine distance (meters)
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- Segment and renumber per MMSI
def segment_and_renumber(df):
    segmented = []
    for mmsi, g in df.groupby("MMSI", observed=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        dt = g["Timestamp"].diff().dt.total_seconds().fillna(0)
        seg_raw = (dt > GAP_BREAK_MIN * 60).cumsum()
        g["Segment"] = seg_raw - seg_raw.min() + 1
        segmented.append(g)
    return pd.concat(segmented, ignore_index=True)

# --- Load data
df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)
# --- Segment first (sequential per MMSI)
df = segment_and_renumber(df)

# --- Downsample & interpolate per segment
results = []
for (mmsi, seg), g in df.groupby(["MMSI", "Segment"], observed=True):
    g = g.set_index("Timestamp")

    # Downsample to 1-minute intervals (keep last)
    g1 = g.resample("1min").last()

    # Interpolate numeric columns for short gaps only
    g1[NUM_COLS] = g1[NUM_COLS].interpolate(
        method="time", limit=INTERP_LIMIT_MIN, limit_direction="both"
    )

    # Drop minutes still NaN (beyond real range or long gaps)
    g1 = g1.dropna(subset=NUM_COLS, how="all")

    # Fill identifiers
    g1["MMSI"] = mmsi
    g1["Segment"] = seg

    # --- Outlier guards ---
    lat = g1["Latitude"].to_numpy()
    lon = g1["Longtitude"].to_numpy()
    lat_prev, lon_prev = np.roll(lat, 1), np.roll(lon, 1)
    lat_prev[0], lon_prev[0] = lat[0], lon[0]

    g1["distance_m"] = haversine_m(lat, lon, lat_prev, lon_prev)
    g1.loc[g1.index[0], "distance_m"] = 0.0
    g1["speed_mps_track"] = g1["distance_m"] / 60.0

    # Filter unrealistic movement or SOG
    g1 = g1[(g1["distance_m"] < MAX_DISTANCE_M) & (g1["SOG"] <= MAX_SOG_KNOTS)]

    results.append(g1)


# --- Combine & save
df_clean = pd.concat(results).reset_index()
print("Before deleting", len(df_clean))
missing = df_clean[df_clean[["SOG", "COG", "Latitude", "Longtitude"]].isna().any(axis=1)]
print(f"Missing numeric data rows: {len(missing)}")
# Removing rows with empty data approximately 6%
df_clean = df_clean.dropna(subset=["SOG", "COG", "Latitude", "Longtitude", "MMSI", "Segment"])
print("After deleting", len(df_clean))
df_clean.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned 1-minute AIS data saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df_clean):,}, Ships: {df_clean['MMSI'].nunique()}, Segments: {df_clean['Segment'].nunique()}")