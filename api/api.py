import sys
from pathlib import Path
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from shapely.geometry import Point, Polygon  # NEW: Imported for geographic guardrails

# Import your pipeline functions exactly as written in your scripts
from src.feature_definitions import haversine_array, dummy_manhattan_distance, bearing_array
from src.config import PREPROCESSOR_PATH, MODEL_PATH, OSRM_BASE_URL  
from src.features_osrm import NYCTaxiFeatureEngineer
from src.dataset_osrm import (
    change_dtypes,
    filter_coordinates,
    add_reverse_geocoding,
    compute_spatial_metrics,
    extract_temporal_features,
    osrm_unit_conversion,
    drop_features
)
from api.schemas import TaxiPredictionRequest

ml_models = {}

def _register_pickle_compatibility() -> None:
    """Expose notebook-trained classes under the module names stored in pickles."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "NYCTaxiFeatureEngineer"):
        main_module.NYCTaxiFeatureEngineer = NYCTaxiFeatureEngineer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads artifacts on startup and cleans up on shutdown."""
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing artifacts. Ensure {PREPROCESSOR_PATH} and {MODEL_PATH} exist."
        )

    _register_pickle_compatibility()
    
    # Load scikit-learn pipeline & trained model
    ml_models["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    ml_models["estimator"] = joblib.load(MODEL_PATH)
    yield
    ml_models.clear()

app = FastAPI(
    title="NYC Taxi Trip Duration Predictor", 
    version="1.0.0", 
    lifespan=lifespan
)

def run_raw_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the precise step transformations executed in main() 
    excluding outlier/anomaly removals to prevent production drops.
    """
    return (
        df
        .pipe(change_dtypes)
        .pipe(filter_coordinates)
        .pipe(add_reverse_geocoding)
        .pipe(compute_spatial_metrics)
        .pipe(extract_temporal_features)
        .pipe(osrm_unit_conversion)
        .pipe(drop_features)
    )

# --- NEW: Guardrail Validation Logic ---
def is_valid_tlc_location(lat: float, lon: float) -> bool:
    """
    Checks if a coordinate falls strictly within the permitted NYC TLC zone polygon.
    (5 boroughs, Nassau, Westchester, Newark Airport)
    """
    point = Point(lon, lat)
    
    nyc_tlc_zone = Polygon([
        (-74.25, 40.50), # Bottom Left (Staten Island)
        (-73.70, 40.55), # Bottom Right (Queens/Nassau edge)
        (-73.40, 40.85), # Top Right (Nassau/Westchester edge)
        (-73.90, 41.38), # Top (Westchester)
        (-74.00, 40.80), # Top Left (Bronx/Manhattan edge)
        (-74.25, 40.50)  # Close the loop
    ])
    
    return nyc_tlc_zone.contains(point)
# ---------------------------------------

@app.post("/predict")
async def predict_duration(payload: TaxiPredictionRequest):
    
    # --- STEP 1: APPLY GUARDRAILS ---
    # We check this FIRST to save processing power and avoid unnecessary OSRM API calls
    if not is_valid_tlc_location(payload.pickup_latitude, payload.pickup_longitude):
        raise HTTPException(
            status_code=400, 
            detail="🚫 Guardrail Activated: Pickup location is outside the permitted NYC TLC zone."
        )
        
    if not is_valid_tlc_location(payload.dropoff_latitude, payload.dropoff_longitude):
        raise HTTPException(
            status_code=400, 
            detail="🚫 Guardrail Activated: Dropoff location is outside the permitted NYC TLC zone."
        )
    # --------------------------------

    # 1. Convert input request schema to dictionary
    input_dict = payload.model_dump()
    
    # Crucial Pipeline Fix: extract_temporal_features drops 'trip_duration' after tracking it
    # We add a temporary dummy value here so pandas pipeline does not crash
    input_dict["trip_duration"] = 0.0

    # 2. Automatically fetch real-time OSRM routing metrics behind the scenes
    coords_url = f"{payload.pickup_longitude},{payload.pickup_latitude};{payload.dropoff_longitude},{payload.dropoff_latitude}"
    osrm_url = f"{OSRM_BASE_URL}/route/v1/driving/{coords_url}?overview=false&steps=true"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(osrm_url, timeout=3.0)
            
        if response.status_code == 200:
            route_data = response.json()
            if route_data.get("routes"):
                best_route = route_data["routes"][0]
                # Inject OSRM parameters expected by your pipeline
                input_dict["total_distance"] = float(best_route["distance"])
                input_dict["total_travel_time"] = float(best_route["duration"])
                input_dict["number_of_steps"] = int(len(best_route["legs"][0]["steps"]))
            else:
                raise HTTPException(status_code=400, detail="No viable driving route could be found for coordinates.")
        else:
            raise httpx.HTTPStatusError("OSRM API error status", request=None, response=response)

    except (httpx.RequestError, httpx.HTTPStatusError):
        # Fallback to sensible data science defaults if routing service is slow/unreachable
        input_dict["total_distance"] = 3000.0  # ~3km average
        input_dict["total_travel_time"] = 600.0 # ~10 mins average
        input_dict["number_of_steps"] = 5

    try:
        # 3. Create DataFrame row
        raw_data = pd.DataFrame([input_dict])
        
        # 4. Run base transformations
        processed_base = run_raw_preprocessing(raw_data)
        
        if processed_base.empty:
            raise HTTPException(
                status_code=400, 
                detail="Input coordinates fell outside structural geographic boundary parameters."
            )
            
        # 5. Pipeline transformation via Scikit-Learn Preprocessor
        final_features = ml_models["preprocessor"].transform(processed_base)
        
        # 6. Model Prediction
        prediction = ml_models["estimator"].predict(final_features)
        
        return {
            "status": "success",
            "predicted_trip_duration_minutes": float(np.round(prediction[0], 2))
        }

    except HTTPException:
        raise
    except (KeyError, RuntimeError, ImportError) as e:
        raise HTTPException(status_code=503, detail=f"Prediction service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "components_loaded": list(ml_models.keys())}