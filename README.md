# NYC Taxi Trip Duration Studio 🚕

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
</a>
<a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</a>
<a href="https://dvc.org/">
    <img src="https://img.shields.io/badge/DVC-13E4C6?logo=dvc&logoColor=white" alt="DVC">
</a>

A comprehensive MLOps project to predict the total trip duration of taxi rides in New York City. This repository features a full end-to-end Machine Learning pipeline utilizing Data Version Control (DVC) for reproducible data processing and model training, and a Streamlit web application to provide real-time route-aware predictions.

## Features ✨

* **End-to-end Data Pipeline**: Managed via DVC (`dvc.yaml`) for processing raw data, generating OSRM features, training models (LightGBM/XGBoost), and evaluating performance.
* **Streamlit Web Application**: An interactive interface to input pickup/dropoff coordinates and receive instant trip duration predictions.
* **Geospatial & Route Intel**: Incorporates real-time map routing data (distances, steps, duration) via OSRM to drastically improve prediction accuracy.
* **Containerized Deployment**: Ready to be launched using Docker and Docker Compose for a seamless setup experience.
* **Pre-configured Environment**: Easy setup with `Makefile` commands to manage dependencies and environments.

## Project Organization 📂

```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make create_environment`
├── README.md          <- The top-level README for developers using this project.
├── data/              <- Data directory managed by DVC (raw, processed, external)
├── docs/              <- A default mkdocs project; see www.mkdocs.org for details
├── models/            <- Trained and serialized models (.joblib files)
├── notebooks/         <- Jupyter notebooks for data exploration and prototyping.
├── src/               <- Source code for use in this project (ML pipeline, feature engineering, config).
├── app.py             <- Streamlit Web Application entry point.
├── dvc.yaml           <- DVC Pipeline definition file.
├── params.yaml        <- Parameters for model training and feature engineering.
├── Dockerfile         <- Docker image configuration.
├── compose.yaml       <- Docker Compose configuration to easily run the app.
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
└── pyproject.toml     <- Project configuration file with package metadata.
```

## Getting Started 🚀

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### 1. Prerequisites

You will need the following installed on your machine:
- **Python 3.10+** (Python 3.13 is the default in Makefile)
- **Git** & **DVC**
- **Docker & Docker Compose** (Optional, but recommended for running the web app easily)
- **Make** (For utilizing the provided Makefile)

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd NYC-Taxi-Trip-Duration
```

### 3. Setup Python Environment

You can use the provided `Makefile` to quickly create a conda environment and install dependencies:

```bash
# Create a conda environment named NYC-Taxi-Trip-Duration
make create_environment

# Activate the environment
conda activate NYC-Taxi-Trip-Duration

# Install dependencies
make requirements
```

Alternatively, you can manually create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Running the Machine Learning Pipeline (DVC)

This project uses DVC to manage the data processing and training pipeline. The pipeline steps are defined in `dvc.yaml`.

To reproduce the entire pipeline (data processing, feature engineering, model training, and evaluation), run:

```bash
dvc repro
```
This will automatically track changes and re-run only the necessary stages that were modified. Models will be saved in the `models/` directory and evaluation metrics will be outputted to `reports/evaluation_metrics.json`.

*Note: Ensure you have the raw data available in `data/raw/` or configured remote storage via `make sync_data_down`.*

### 5. Running the Web Application (Streamlit)

You can run the web application either locally or via Docker.

#### Option A: Running Locally via Streamlit

With your virtual environment activated, run:

```bash
streamlit run app.py
```
The app should automatically open in your default browser at `http://localhost:8501`.

#### Option B: Running via Docker Compose (Recommended)

If you have Docker installed, you can easily spin up the application in an isolated container. From the root of the project, run:

```bash
docker compose up --build
```
This will build the Docker image and start the Streamlit service. Access the application in your browser (usually available at `http://localhost:8501` or `http://localhost:8000`).

## Development 🛠️

- **Linting & Formatting**: Ensure your code follows the standard style guidelines by running:
  ```bash
  make lint
  make format
  ```
- **Modifying the Model**: Adjust model hyperparameters inside `params.yaml` and re-run `dvc repro` to retrain the model and log new metrics.

## License 📜

This project is licensed under the terms provided in the `LICENSE` file.
