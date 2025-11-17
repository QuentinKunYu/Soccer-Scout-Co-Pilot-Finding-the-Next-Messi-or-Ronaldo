# syntax=docker/dockerfile:1.7

# Use slim Python image to keep build lightweight while retaining Debian tools.
FROM python:3.11-slim AS base

# Avoid generating .pyc files and make stdout unbuffered for better logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install required system packages needed by Streamlit/Altair (fonts, git for debugging).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Copy and set permissions for the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directories to ensure they exist
RUN mkdir -p /app/data /app/data/processed

# Streamlit defaults to port 8501; expose for docker run convenience.
EXPOSE 8501

# Use entrypoint script to handle data processing and app startup
ENTRYPOINT ["/app/entrypoint.sh"]
