#!/bin/bash
# entrypoint.sh: Automated setup script for Docker container
# This script ensures data is processed before starting the Streamlit app

set -e  # Exit immediately if any command fails

echo "=================================================="
echo "Soccer Scout Co-Pilot - Container Initialization"
echo "=================================================="
echo ""

# Check if processed data already exists
if [ -f "/app/data/processed/player_recommendations.csv" ]; then
    echo "✓ Processed data already exists"
    echo "  Skipping data processing pipeline..."
    echo ""
else
    echo "⚠ Processed data not found"
    echo "  Running prediction pipeline to generate data..."
    echo ""
    
    # Check if raw data exists
    if [ ! -d "/app/data" ] || [ -z "$(ls -A /app/data/*.csv 2>/dev/null)" ]; then
        echo "=================================================="
        echo "ERROR: Raw data files not found!"
        echo "=================================================="
        echo ""
        echo "Please download the dataset from:"
        echo "https://www.kaggle.com/datasets/davidcariboo/player-scores"
        echo ""
        echo "Then place the following CSV files in the 'data/' directory:"
        echo "  - players.csv"
        echo "  - clubs.csv"
        echo "  - competitions.csv"
        echo "  - games.csv"
        echo "  - appearances.csv"
        echo "  - game_events.csv"
        echo "  - game_lineups.csv"
        echo "  - club_games.csv"
        echo "  - player_valuations.csv"
        echo "  - transfers.csv"
        echo ""
        echo "After adding the data files, restart the container:"
        echo "  docker compose restart"
        echo ""
        exit 1
    fi
    
    echo "✓ Raw data files found"
    echo "  Starting prediction pipeline..."
    echo ""
    
    # Run the prediction pipeline to generate processed data
    python /app/run_predictions.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Data processing completed successfully!"
        echo ""
    else
        echo ""
        echo "=================================================="
        echo "ERROR: Data processing failed!"
        echo "=================================================="
        echo ""
        echo "Please check the error messages above and ensure:"
        echo "  1. All required CSV files are present in data/"
        echo "  2. CSV files are not corrupted"
        echo "  3. You have sufficient memory available"
        echo ""
        exit 1
    fi
fi

echo "=================================================="
echo "Starting Streamlit Application"
echo "=================================================="
echo ""
echo "🚀 Application will be available at:"
echo "   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=================================================="
echo ""

# Start Streamlit application
exec streamlit run /app/app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true

