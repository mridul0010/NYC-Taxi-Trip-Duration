from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PARAMS_FILE = PROJ_ROOT / "params.yaml"

TRAIN_INPUT_PATH = PROCESSED_DATA_DIR / "osrm_boosted"
TEST_INPUT_PATH = PROCESSED_DATA_DIR / "osrm_boosted"


AVG_EARTH_RADIUS = 6371

# features.py
TARGET_COLUMN = "trip_duration_min"

HIGH_CARDINALITY_FEATURES = ["pickup_zone", "dropoff_zone", "route_time_density"]

OHE_FEATURES = ["dropoff_state", "pickup_state", "travel_quadrant", "season_name"]

NUMERICAL_COLS = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "distance_haversine_km",
    "distance_manhattan_km",
    "direction",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
    "is_rush_hour",
    "is_interstate_trip",
    "is_late_night",
    "is_weekend_night"
]

LOG_TRANSFORM_FEATURES_BASELINE = [
    'distance_haversine_km',
    'distance_manhattan_km'
]

NUMERICAL_COLS_OSRM = [
    "vendor_id",
    "passenger_count",
    "store_and_fwd_flag",
    "distance_haversine_km",
    "distance_manhattan_km",
    "direction",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
    "is_rush_hour",
    "is_interstate_trip",
    "is_late_night",
    "is_weekend_night",
    'total_distance_km',
    'total_travel_time_min'
]

LOG_TRANSFORM_FEATURES_OSRM = [
    'distance_haversine_km',
    'distance_manhattan_km',
    'total_distance_km',
    'total_travel_time_min'
]





# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
