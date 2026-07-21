from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder , FunctionTransformer

from src.config import (
    PROCESSED_DATA_DIR,
    PARAMS_FILE,
    TARGET_COLUMN,
    HIGH_CARDINALITY_FEATURES,
    OHE_FEATURES,
    NUMERICAL_COLS_OSRM,
    LOG_TRANSFORM_FEATURES_OSRM
) 

app = typer.Typer()


class NYCTaxiFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the engineered features defined in the notebook."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        route_comb = X["pickup_zone"].astype(str) + " -> " + X["dropoff_zone"].astype(str)
        time_bin = pd.cut(
            X["hour_of_day"],
            bins=[0, 6, 12, 16, 20, 24],
            labels=["Night", "Morning", "Midday", "Evening_Rush", "Late_Night"],
            include_lowest=True,
        ).astype(str)

        X["route_time_density"] = route_comb + " @ " + time_bin
        X["is_interstate_trip"] = (
            X["pickup_state"].astype(str) != X["dropoff_state"].astype(str)
        ).astype(int)
        X["travel_quadrant"] = pd.cut(
            X["direction"],
            bins=[-180, -90, 0, 90, 180],
            labels=["South-West", "North-West", "North-East", "South-East"],
            include_lowest=True,
        ).astype(str)
        X["is_late_night"] = X["hour_of_day"].between(0, 5).astype(int)
        X["is_weekend_night"] = (X["is_late_night"] & X["day_of_week"].isin([4, 5, 6])).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None

        input_features_list = list(input_features)
        new_features = [
            "route_time_density",
            "is_interstate_trip",
            "travel_quadrant",
            "is_late_night",
            "is_weekend_night",
        ]
        return np.array(input_features_list + new_features)


def build_pipeline() -> Pipeline:
    updated_high_cardinality = HIGH_CARDINALITY_FEATURES 
    updated_ohe = OHE_FEATURES 
    updated_numerical = NUMERICAL_COLS_OSRM
    updated_log_col = LOG_TRANSFORM_FEATURES_OSRM

    column_transformer = ColumnTransformer(
        transformers=[
            ("target_encoded", TargetEncoder(smooth="auto"), updated_high_cardinality),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), updated_ohe),
            ("log_transform" , FunctionTransformer(np.log1p , validate=True , feature_names_out='one-to-one') , updated_log_col),
            ("standard_scaled", StandardScaler(), updated_numerical),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", NYCTaxiFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
    
    pipeline.set_output(transform="pandas")
    return pipeline


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_params(file_path: Path) -> dict:
    if not file_path.exists():
        logger.warning(f"Parameters file not found at {file_path}. Using defaults.")
        return {}
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file

def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir / transformer_name
    joblib.dump(value=transformer, filename=save_location)
    logger.info("Successfully serialized transformer state to {}", save_location)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "osrm_boosted" / "processed_osrm_NYC.csv",
    output_path: Path = PROCESSED_DATA_DIR / "osrm_boosted" / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "osrm_boosted" / "labels.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "osrm_boosted" / "features_test.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "osrm_boosted" / "labels_test.csv",
    target_column: str = TARGET_COLUMN,
    test_size: float | None = None,
    random_state: int | None = None,
):

    logger.info("Loading processed data from {}", input_path)
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {input_path}")

    # Load parameters from yaml config
    params = read_params(PARAMS_FILE)
    
    features_params = params.get("features_osrm", {})
    if test_size is None:
        test_size = features_params.get("test_size", 0.2)
    if random_state is None:
        random_state = features_params.get("random_state", 42)

    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")

    X = df.drop(columns=target_column)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_pipeline()

    logger.info("Fitting feature engineering pipeline on the training split")
    pipeline.fit(X_train, y_train)

    logger.info("Transforming training and test splits")
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    X_train_trans.index = X_train.index
    X_test_trans.index = X_test.index

    # DataFrames are saved inside the newly confirmed baseline directory
    _save_frame(X_train_trans, output_path)
    _save_frame(y_train.to_frame(name=target_column), labels_path)
    _save_frame(X_test_trans, test_output_path)
    _save_frame(y_test.to_frame(name=target_column), test_labels_path)

    transformer_filename = "preprocessor_osrm.joblib"
    try:
        root_path = Path(__file__).resolve().parent.parent
    except NameError:
        root_path = Path(".").resolve()
        
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(parents=True, exist_ok=True)
    
    save_transformer(
        transformer=pipeline,
        save_dir=transformer_save_dir,
        transformer_name=transformer_filename
    )

    logger.success("Feature engineering complete")
    logger.info("Training features saved to {}", output_path)
    logger.info("Training labels saved to {}", labels_path)
    logger.info("Test features saved to {}", test_output_path)
    logger.info("Test labels saved to {}", test_labels_path)


if __name__ == "__main__":
    app()