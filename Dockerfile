# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.16
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by LightGBM/Estimators (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies using cache mounts for faster builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy the source code, configs, models, and reports into the container AND grant ownership to appuser
COPY --chown=appuser:appuser . .

# Explicitly ensure models directory exists, has bundled files, and belongs to appuser
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models && chmod -R 755 /app/models

# Ensure Streamlit local configuration runtime directories are writeable by appuser if needed
RUN mkdir -p /app/.streamlit && chown -R appuser:appuser /app/.streamlit

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --- NOW SWITCH TO THE NON-PRIVILEGED USER ---
USER appuser

# Expose the standard port that Streamlit listens on.
EXPOSE 8501

# Run the updated Streamlit application directly.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]