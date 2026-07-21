from pathlib import Path
import pandas as pd
import numpy as np
import reverse_geocoder as rg
from typing import Tuple
from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.feature_definitions import (
    haversine_array,
    dummy_manhattan_distance,
    bearing_array,
)

app = typer.Typer()

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts raw columns to optimized, memory-efficient data types."""
    df = df.copy()
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == "Y").astype('int8')
    df['vendor_id'] = df['vendor_id'].astype('int8')
    df['passenger_count'] = df['passenger_count'].astype('int8')
    
    return df

def filter_coordinates(
    df: pd.DataFrame, 
    lat_bounds: Tuple[float, float] = (40.477398, 40.917577), 
    lng_bounds: Tuple[float, float] = (-74.259090, -73.700272)
) -> pd.DataFrame:
    """
    Dynamically removes spatial outliers outside a specified bounding box (defaults to NYC).
    Replaces brittle hardcoded index dropping.
    """
    in_lat = df['pickup_latitude'].between(lat_bounds[0], lat_bounds[1])
    in_lng = df['pickup_longitude'].between(lng_bounds[0], lng_bounds[1])
    
    return df[in_lat & in_lng].copy()

def add_reverse_geocoding(df: pd.DataFrame) -> pd.DataFrame:
    """Performs batch offline reverse geocoding and extracts fields as categories."""
    df = df.copy()
    
    pickup_coords = list(zip(df['pickup_latitude'], df['pickup_longitude']))
    dropoff_coords = list(zip(df['dropoff_latitude'], df['dropoff_longitude']))

    pickup_results = rg.search(pickup_coords)
    dropoff_results = rg.search(dropoff_coords)

    # pd.Categorical minimizes memory usage for highly repetitive string fields
    df['pickup_zone'] = pd.Categorical([res['name'] for res in pickup_results])
    df['pickup_state'] = pd.Categorical([res['admin1'] for res in pickup_results])
    df['dropoff_zone'] = pd.Categorical([res['name'] for res in dropoff_results])
    df['dropoff_state'] = pd.Categorical([res['admin1'] for res in dropoff_results])

    return df

def compute_spatial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers distance and directional vector metrics."""
    df = df.copy()
    
    df['distance_haversine_km'] = haversine_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['distance_manhattan_km'] = dummy_manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['direction'] = bearing_array(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    return df

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives optimized temporal properties from pickup timestamps."""
    df = df.copy()

    df["hour_of_day"] = df['pickup_datetime'].dt.hour.astype('int8')
    df["day_of_week"] = df['pickup_datetime'].dt.dayofweek.astype('int8')
    df["month_of_year"] = df['pickup_datetime'].dt.month.astype('int8')
    
    is_weekday = df['day_of_week'] < 5
    df['is_rush_hour'] = (
        is_weekday & 
        ((df['hour_of_day'].between(7, 10)) | (df['hour_of_day'].between(16, 19)))
    ).astype(int)

    season = df['pickup_datetime'].dt.month % 12 // 3 + 1
    season_mapping = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df["season_name"] = pd.Categorical(season.map(season_mapping))

    # Safely transform trip duration column without side-effect issues
    df['trip_duration_min'] = (df['trip_duration'] / 60).round(2)
    df = df.drop(columns=['trip_duration'])

    return df

def remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out impossible speeds, invalid states, and unrealistic distances."""
    df = df.copy()
    
    # Avoid zero-division bugs by enforcing trip_duration_min > 0 first
    valid_duration = df['trip_duration_min'] > 0
    
    # Implied speed calculation (km / hours)
    implied_speed = df['distance_haversine_km'] / (df['trip_duration_min'] / 60)
    
    valid_states = ["New York", "New Jersey", "Connecticut"]

    mask = (
        valid_duration &
        (df['distance_haversine_km'].between(0.5, 150)) & 
        (df['trip_duration_min'].between(1, 180)) &
        (implied_speed < 130) & 
        (df['pickup_state'].isin(valid_states)) & 
        (df['dropoff_state'].isin(valid_states))
    )
    
    return df[mask].copy()

def passenger_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['passenger_count'].between(1, 6)].copy()

def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely removes raw or highly correlated columns that are no longer 
    needed after feature engineering.
    """
    df = df.copy()
    
    columns_to_drop = [
        "id",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude"
    ]
    
    # errors='ignore' ensures the pipeline won't crash if a column was missing
    return df.drop(columns=columns_to_drop, errors='ignore')

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "NYC.csv",
    output_path: Path = PROCESSED_DATA_DIR / "baseline" / "processed_NYC.csv",
):
    """
    Main execution pipeline that loads, cleans, engineers features, 
    and saves the optimized dataset.
    """

    # Explicitly ensure the baseline directory exists right away
    baseline_dir = PROCESSED_DATA_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ensured baseline directory exists at {}", baseline_dir)

    logger.info(f"Validating dataset entry at {input_path}...")
    
    if not input_path.exists():
        logger.error(f"Execution terminated: Input file not found at {input_path}")
        raise typer.Exit(code=1)

    try:
        logger.info("Loading raw dataset...")
        df_raw = pd.read_csv(input_path)
        
        logger.info(f"Executing processing pipeline on {len(df_raw):,} records...")
        
        # Linear sequence using method chaining via .pipe()
        processed_df = (
            df_raw
            .pipe(change_dtypes)
            .pipe(filter_coordinates)
            .pipe(add_reverse_geocoding)
            .pipe(compute_spatial_metrics)
            .pipe(extract_temporal_features)
            .pipe(remove_anomalies)
            .pipe(passenger_anomaly)
            .pipe(drop_features)
        )
        
        logger.info(f"Writing polished dataset to final storage target: {output_path}")
        
        # Ensure parent directory exists dynamically if it doesn't already
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # index=False drops the uninformative default row index integer
        processed_df.to_csv(output_path, index=False)
        
        logger.success(f"Processing complete! Final shape: {processed_df.shape}")
        
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
