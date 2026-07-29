from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
import typer

# Fallback project root resolution if src.config is unavailable
try:
    from src.config import PROJ_ROOT
except ImportError:
    PROJ_ROOT = Path(__file__).resolve().parents[1]

load_dotenv()

app = typer.Typer(help="Register a trained ML model in MLflow/DagsHub and manage lifecycle metadata.")

try:
    import dagshub
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:
    dagshub = None
    mlflow = None
    MlflowClient = None
    MlflowException = Exception


DEFAULT_RUN_INFO_PATH = PROJ_ROOT / "run_information.json"
DEFAULT_OUTPUT_JSON_PATH = PROJ_ROOT / "models" / "registered_model_info.json"
DEFAULT_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
DEFAULT_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Staging")  # Kept for legacy compatibility


def _require_dependencies() -> None:
    if dagshub is None or mlflow is None or MlflowClient is None:
        raise RuntimeError(
            "mlflow and dagshub must be installed before registering a model."
        )


def _load_run_information(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Run information file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file_handle:
        run_info = json.load(file_handle)

    if not isinstance(run_info, dict):
        raise ValueError(f"Run information file must contain a JSON object: {file_path}")

    missing_keys = [key for key in ("run_id", "model_name") if key not in run_info]
    if missing_keys:
        raise ValueError(
            f"Run information file is missing required keys {missing_keys}: {file_path}"
        )

    return run_info


def _resolve_tracking_uri() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    if dagshub_owner and dagshub_repo:
        return f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow"

    raise ValueError(
        "Set MLFLOW_TRACKING_URI or both DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME in .env"
    )


def _configure_mlflow() -> str:
    _require_dependencies()

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not repo_owner or not repo_name:
        raise ValueError("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME in .env")

    tracking_uri = _resolve_tracking_uri()
    dagshub_username = os.getenv("DAGSHUB_USERNAME", repo_owner)
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    else:
        logger.warning("DAGSHUB_TOKEN is not set; registry access may fail for private repos.")

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(tracking_uri)

    return tracking_uri


def _wait_for_model_version_ready(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> None:
    deadline = time.monotonic() + timeout_seconds

    while True:
        model_version = client.get_model_version(name=model_name, version=version)
        status = (model_version.status or "").upper()

        if status == "READY":
            return

        if status in {"FAILED_REGISTRATION", "FAILED"}:
            raise RuntimeError(
                f"Model registration failed for {model_name} v{version}: {model_version.status_message}"
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for model version {version} of {model_name} to become READY"
            )

        time.sleep(poll_interval_seconds)


def _resolve_logged_model_id(client: MlflowClient, run_id: str, model_name: str) -> str:
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise ValueError(f"Unable to load MLflow run {run_id}: {exc}") from exc

    search_experiment_ids = [str(run.info.experiment_id)]
    filter_string = f"source_run_id = '{run_id}'"
    
    logged_models = client.search_logged_models(
        experiment_ids=search_experiment_ids,
        filter_string=filter_string,
    )

    if not logged_models:
        raise ValueError(f"No logged models found whatsoever for run {run_id}.")

    # Attempt exact match by name attribute or artifact path match
    for lm in logged_models:
        if getattr(lm, "name", None) == model_name or model_name in getattr(lm, "artifact_path", ""):
            return str(lm.model_id)

    # Informative fallback if name matching fails
    logger.warning(
        "Exact match for model name '{}' not found in logged models. Defaulting to first available model ID.",
        model_name
    )
    return str(logged_models[0].model_id)


@app.command()
def main(
    run_info_path: Path = DEFAULT_RUN_INFO_PATH,
    output_json_path: Path = DEFAULT_OUTPUT_JSON_PATH,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    model_stage: str = DEFAULT_MODEL_STAGE,
    wait_timeout_seconds: int = 300,
    poll_interval_seconds: int = 5,
) -> None:
    try:
        _configure_mlflow()

        logger.info("Loading run metadata from {}", run_info_path)
        run_info = _load_run_information(run_info_path)

        run_id = str(run_info["run_id"])
        model_name = str(run_info["model_name"])
        
        client = MlflowClient()
        model_id = _resolve_logged_model_id(client=client, run_id=run_id, model_name=model_name)
        model_uri = f"models:/{model_id}"

        logger.info("Registering model from URI: {}", model_uri)
        try:
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        except MlflowException as exc:
            raise RuntimeError(
                f"MLflow could not register the model from {model_uri}: {exc}"
            ) from exc

        logger.info(
            "Waiting for registry to finalize model version {} for {}",
            model_version.version,
            model_name,
        )
        _wait_for_model_version_ready(
            client=client,
            model_name=model_name,
            version=model_version.version,
            timeout_seconds=wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

        # IMPROVISATION: Automatic Metadata & Metrics Enrichment
        try:
            source_run = client.get_run(run_id)
            metrics_summary = {k: float(v) for k, v in source_run.data.metrics.items()}
            
            # Tag the model version with metrics for easy UI sorting/filtering
            for metric_name, metric_val in metrics_summary.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=f"metric.{metric_name}",
                    value=str(metric_val)
                )
            
            # Add a rich markdown description to the version documentation
            description_md = (
                f"### Auto-Registered Version\n"
                f"- **Source Run ID:** `{run_id}`\n"
                f"- **Key Metrics:** {json.dumps(metrics_summary)}\n"
                f"- **Registered On:** {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description_md
            )
            logger.info("Successfully enriched model version metadata and tags with run metrics.")
        except Exception as meta_exc:
            logger.warning("Failed to automatically enrich model metadata: {}", meta_exc)
            metrics_summary = {}

        # IMPROVISATION: Modern Aliases Configuration (Preferred over legacy Stages)
        if model_alias:
            logger.info("Setting alias '{}' on model {} version {}", model_alias, model_name, model_version.version)
            client.set_registered_model_alias(
                name=model_name,
                alias=model_alias,
                version=model_version.version
            )

        # Legacy Stage Transition (kept as a backward compatibility option)
        if model_stage:
            logger.info("Transitioning model {} version {} to stage {}", model_name, model_version.version, model_stage)
            try:
                # Suppress modern deprecation warnings natively if using newer MLflow versions
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=model_stage,
                )
            except Exception as stage_exc:
                logger.debug("Legacy stage transition skipped or unsupported: {}", stage_exc)

        registration_summary = {
            "registered_model_name": model_name,
            "logged_model_id": model_id,
            "model_version": model_version.version,
            "assigned_alias": model_alias or "none",
            "promoted_stage": model_stage or "none",
            "extracted_metrics": metrics_summary,
            "timestamp": time.time()
        }
        
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(registration_summary, f, indent=4)
        logger.info("DVC registration artifact safely saved to {}", output_json_path)

        logger.success(
            "Model {} version {} successfully registered, tagged, and set to alias '{}'.",
            model_name,
            model_version.version,
            model_alias or "no-alias",
        )

    except FileNotFoundError as exc:
        logger.exception("Registration failed because an input file is missing: {}", exc)
        raise typer.Exit(code=1)
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        logger.exception("Model registration failed: {}", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected error occurred while registering the model")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()