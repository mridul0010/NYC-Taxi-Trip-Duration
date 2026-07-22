import json
import sys
from pathlib import Path
from datetime import datetime

import httpx
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, OSRM_BASE_URL, PREPROCESSOR_PATH
from src.dataset_osrm import (
    add_reverse_geocoding,
    change_dtypes,
    compute_spatial_metrics,
    drop_features,
    extract_temporal_features,
    filter_coordinates,
    osrm_unit_conversion,
)

# -----------------------------------------------------------------------------
# NYC Predefined Hotspots (Piche exact coordinate jayenge, user ko sirf naam dikhega)
# -----------------------------------------------------------------------------
NYC_HOTSPOTS = {
    "Times Square (Midtown)": {"lat": 40.7580, "lon": -73.9855},
    "Central Park (The Mall)": {"lat": 40.7829, "lon": -73.9654},
    "JFK International Airport": {"lat": 40.6413, "lon": -73.7781},
    "Empire State Building": {"lat": 40.7484, "lon": -73.9857},
    "LaGuardia Airport (LGA)": {"lat": 40.7769, "lon": -73.8740},
    "Penn Station / Madison Square Garden": {"lat": 40.7506, "lon": -73.9935},
    "Grand Central Terminal": {"lat": 40.7527, "lon": -73.9772},
    "Battery Park (Lower Manhattan)": {"lat": 40.7033, "lon": -74.0170},
    "Brooklyn Bridge Park": {"lat": 40.7023, "lon": -73.9964},
    "Metropolitan Museum of Art (The Met)": {"lat": 40.7794, "lon": -73.9632}
}

# -----------------------------------------------------------------------------
# Pipeline Core Logic
# -----------------------------------------------------------------------------
def _register_pickle_compatibility() -> None:
    """Expose notebook-trained classes under the module names stored in pickles."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "NYCTaxiFeatureEngineer"):
        from src.features_osrm import NYCTaxiFeatureEngineer
        main_module.NYCTaxiFeatureEngineer = NYCTaxiFeatureEngineer

@st.cache_resource(show_spinner=False)
def load_models():
    """Load the trained preprocessing pipeline and estimator once per session."""
    _register_pickle_compatibility()
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Ensure {PREPROCESSOR_PATH} and {MODEL_PATH} exist."
        )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    estimator = joblib.load(MODEL_PATH)
    return preprocessor, estimator

def run_raw_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing flow used by the API pipeline."""
    return (
        df.pipe(change_dtypes)
        .pipe(filter_coordinates)
        .pipe(add_reverse_geocoding)
        .pipe(compute_spatial_metrics)
        .pipe(extract_temporal_features)
        .pipe(osrm_unit_conversion)
        .pipe(drop_features)
    )

def fetch_route_features(payload: dict) -> dict:
    """Try to fetch live OSRM features and fall back to sensible defaults."""
    coords_url = (
        f"{payload['pickup_longitude']},{payload['pickup_latitude']};"
        f"{payload['dropoff_longitude']},{payload['dropoff_latitude']}"
    )
    osrm_url = f"{OSRM_BASE_URL}/route/v1/driving/{coords_url}?overview=false&steps=true"
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(osrm_url)
        if response.status_code == 200:
            route_data = response.json()
            if route_data.get("routes"):
                best_route = route_data["routes"][0]
                return {
                    "total_distance": float(best_route["distance"]),
                    "total_travel_time": float(best_route["duration"]),
                    "number_of_steps": int(len(best_route["legs"][0]["steps"])),
                }
    except Exception:
        pass
    return {
        "total_distance": 3000.0,
        "total_travel_time": 600.0,
        "number_of_steps": 5,
    }

@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    """Load evaluation metrics from the project report file."""
    metrics_path = Path(__file__).resolve().parent / "reports" / "evaluation_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "test_mae": float(data.get("test_mae", 0.0)),
        "test_rmse": float(data.get("test_rmse", 0.0)),
        "test_r2": float(data.get("test_r2", 0.0)),
        "cv_score_mae": float(data.get("cv_score_mae", 0.0)),
        "prediction_bias": float(data.get("prediction_bias", 0.0)),
    }

def predict_trip_duration(payload: dict) -> dict:
    """Produce a trip duration prediction from the trained model."""
    input_dict = payload.copy()
    input_dict["trip_duration"] = 0.0
    route_features = fetch_route_features(payload)
    input_dict.update(route_features)
    raw_data = pd.DataFrame([input_dict])
    processed_base = run_raw_preprocessing(raw_data)
    if processed_base.empty:
        raise ValueError("The selected hotspots fall outside the supported NYC boundary.")
    preprocessor, estimator = load_models()
    final_features = preprocessor.transform(processed_base)
    prediction = estimator.predict(final_features)
    return {
        "predicted_trip_duration_minutes": float(np.round(prediction[0], 2)),
        "route_distance_km": float(np.round(route_features["total_distance"] / 1000.0, 1)),
        "route_time_minutes": float(np.round(route_features["total_travel_time"] / 60.0, 1)),
        "route_steps": int(route_features["number_of_steps"]),
    }

def render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0f2b4d 45%, #1f5a87 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .info-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .info-card-title {
            color: #9dc7ff;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
        }
        .info-card-value {
            color: #f7fbff;
            font-size: 1.35rem;
            font-weight: 600;
            word-wrap: break-word;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 0.7rem 0.8rem;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background: rgba(255,255,255,0.10);
            color: #f7fbff;
            border-radius: 10px;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        .submit-btn-container {
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='text-align:center; padding:0.2rem 0 1rem 0;'>
            <h1 style='margin-bottom:0.2rem; color:#f9fcff;'>🚕 NYC Taxi Trip Duration Studio</h1>
            <p style='margin-top:0; color:#9dc7ff; font-size:1.05rem;'>A refined, route-aware experience for estimating taxi ride length in minutes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar() -> None:
    with st.sidebar:
        st.header("Model status")
        st.info("The app uses the trained preprocessing pipeline and model artifacts already stored in the project.")
        
        if st.button("🧪 Load sample NYC trip", use_container_width=True):
            st.session_state["vendor_id"] = 2
            st.session_state["p_date"] = datetime(2023, 1, 15).date()
            st.session_state["p_time"] = datetime(2023, 1, 15, 8, 30).time()
            st.session_state["d_date"] = datetime(2023, 1, 15).date()
            st.session_state["d_time"] = datetime(2023, 1, 15, 8, 55).time()
            st.session_state["passenger_count"] = 2
            st.session_state["pickup_landmark"] = "Times Square (Midtown)"
            st.session_state["dropoff_landmark"] = "Central Park (The Mall)"
            st.session_state["store_and_fwd_flag"] = "N"
            st.rerun()

        st.markdown("### Metric definitions")
        st.markdown("**⏱️ Predicted Duration:** The total travel time in minutes estimated by our Gradient Boosting ML model.")
        st.markdown("**📏 Estimated Distance:** The actual driving route distance in kilometers fetched in real-time from OSRM.")
        st.markdown("**🔄 Route Steps:** The total number of navigation maneuvers (turns/intersections) required, capturing route complexity for better prediction accuracy.")

        st.markdown("---")
        st.markdown("### What powers this app")
        st.caption("- Route-aware features from OSRM when available")
        st.caption("- Temporal and spatial feature engineering")
        st.caption("- A trained gradient boosting model for duration estimation")

def main() -> None:
    st.set_page_config(page_title="NYC Taxi Duration", page_icon="🚕", layout="wide")
    render_header()
    render_sidebar()

    # Session State Initialization
    if "vendor_id" not in st.session_state: st.session_state["vendor_id"] = 2
    if "p_date" not in st.session_state: st.session_state["p_date"] = datetime(2023, 1, 15).date()
    if "p_time" not in st.session_state: st.session_state["p_time"] = datetime(2023, 1, 15, 8, 30).time()
    if "d_date" not in st.session_state: st.session_state["d_date"] = datetime(2023, 1, 15).date()
    if "d_time" not in st.session_state: st.session_state["d_time"] = datetime(2023, 1, 15, 8, 55).time()
    if "passenger_count" not in st.session_state: st.session_state["passenger_count"] = 2
    if "pickup_landmark" not in st.session_state: st.session_state["pickup_landmark"] = "Times Square (Midtown)"
    if "dropoff_landmark" not in st.session_state: st.session_state["dropoff_landmark"] = "Central Park (The Mall)"
    if "store_and_fwd_flag" not in st.session_state: st.session_state["store_and_fwd_flag"] = "N"

    left_column, right_column = st.columns([1.2, 0.8], gap="large")

    with left_column:
        st.markdown("### Trip details")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                vendor_id = st.number_input(
                    "Vendor ID", min_value=1, max_value=2, value=int(st.session_state["vendor_id"]), step=1
                )
                st.write("**Pickup Datetime**")
                p_date = st.date_input("Pickup Date", value=st.session_state["p_date"], label_visibility="collapsed")
                p_time = st.time_input("Pickup Time", value=st.session_state["p_time"], label_visibility="collapsed")
                
                st.write("**Dropoff Datetime**")
                d_date = st.date_input("Dropoff Date", value=st.session_state["d_date"], label_visibility="collapsed")
                d_time = st.time_input("Dropoff Time", value=st.session_state["d_time"], label_visibility="collapsed")

            with col2:
                passenger_count = st.number_input(
                    "Passenger count", min_value=1, max_value=6, value=int(st.session_state["passenger_count"]), step=1
                )
                store_and_fwd_flag = st.selectbox(
                    "Store and forward flag", ["Y", "N"], index=["Y", "N"].index(st.session_state["store_and_fwd_flag"])
                )
                
                st.write("**Pickup Hotspot**")
                pickup_selection = st.selectbox(
                    "Select Pickup Spot",
                    options=list(NYC_HOTSPOTS.keys()),
                    index=list(NYC_HOTSPOTS.keys()).index(st.session_state["pickup_landmark"]),
                    label_visibility="collapsed"
                )
                
                st.write("**Dropoff Hotspot**")
                dropoff_selection = st.selectbox(
                    "Select Dropoff Spot",
                    options=list(NYC_HOTSPOTS.keys()),
                    index=list(NYC_HOTSPOTS.keys()).index(st.session_state["dropoff_landmark"]),
                    label_visibility="collapsed"
                )

            st.markdown('<div class="submit-btn-container"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict trip duration", use_container_width=True)

        if submitted:
            pickup_coords = NYC_HOTSPOTS[pickup_selection]
            dropoff_coords = NYC_HOTSPOTS[dropoff_selection]

            pickup_str = f"{p_date.strftime('%Y-%m-%d')} {p_time.strftime('%H:%M:%S')}"
            dropoff_str = f"{d_date.strftime('%Y-%m-%d')} {d_time.strftime('%H:%M:%S')}"

            payload = {
                "vendor_id": int(vendor_id),
                "pickup_datetime": pickup_str,
                "dropoff_datetime": dropoff_str,
                "passenger_count": int(passenger_count),
                "pickup_longitude": float(pickup_coords["lon"]),
                "pickup_latitude": float(pickup_coords["lat"]),
                "dropoff_longitude": float(dropoff_coords["lon"]),
                "dropoff_latitude": float(dropoff_coords["lat"]),
                "store_and_fwd_flag": store_and_fwd_flag,
            }

            try:
                result = predict_trip_duration(payload)
                st.success("🎉 Prediction complete!")

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Predicted duration", f"{result['predicted_trip_duration_minutes']} min")
                col_b.metric("Estimated distance", f"{result['route_distance_km']} km")
                col_c.metric("Route steps", result["route_steps"])

                if result["predicted_trip_duration_minutes"] < 15:
                    st.info("⚡ This looks like a quick city hop.")
                elif result["predicted_trip_duration_minutes"] < 30:
                    st.info("🗺️ This is a typical midtown-style ride.")
                else:
                    st.info("🌙 This looks like a longer cross-borough trip.")

                with st.expander("View full request payload JSON"):
                    st.json(payload)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with right_column:
        st.markdown("### Project highlights")
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Model artifacts</div>
                <div class="info-card-value">Loaded from /models</div>
            </div>
            <div class="info-card">
                <div class="info-card-title">Route intel</div>
                <div class="info-card-value">Live OSRM map routing data</div>
            </div>
            <div class="info-card">
                <div class="info-card-title">Prediction target</div>
                <div class="info-card-value">Trip duration in minutes</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown("### Model performance snapshot")
        metrics = load_model_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            col1.metric("Test MAE", f"{metrics['test_mae']:.2f} min")
            col2.metric("Test RMSE", f"{metrics['test_rmse']:.2f} min")

            col3, col4 = st.columns(2)
            col3.metric("Test R²", f"{metrics['test_r2']:.3f}")
            col4.metric("CV MAE", f"{metrics['cv_score_mae']:.2f} min")

            st.caption("Industry regression standards: lower MAE/RMSE and higher R² indicate premium accuracy.")
            st.caption(f"Prediction bias: {metrics['prediction_bias']:.2f} minutes")
        else:
            st.info("Model metrics report compilation was not found in the project folder structure.")

if __name__ == "__main__":
    main()