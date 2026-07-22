# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.14
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

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
    python -m pip install -r requirements.txt

# Copy the source code into the container AND grant ownership to appuser
COPY --chown=appuser:appuser . .

# --- RUN THESE AS ROOT FIRST ---
# Ensure the models directory exists inside the container's working directory
RUN mkdir -p /app/models
# Grant ownership of the folder to appuser so it can write to it
RUN chown -R appuser:appuser /app/models

# --- NOW SWITCH TO THE NON-PRIVILEGED USER ---
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application (fetches model from S3, then starts FastAPI).
CMD ["sh", "-c", "python fetch_artifacts.py && uvicorn app.api:app --host=0.0.0.0 --port=8000"]