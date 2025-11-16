# syntax=docker/dockerfile:1.7

# Use slim Python image so build remains lightweight while keeping Debian tooling.
FROM python:3.11-slim AS base

# Avoid python .pyc files and buffer stdout for better logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system packages that Streamlit/Altair rely on (fonts, git for debugging).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Streamlit defaults to port 8501; expose for docker run convenience.
EXPOSE 8501

# Start the Streamlit dashboard when the container launches.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
