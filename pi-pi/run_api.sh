#!/bin/bash
# Run the maximum power pi calculator

echo "🚀 Maximum Power Pi Calculator for Raspberry Pi 5"
echo "================================================="

# Set performance mode
echo "Setting CPU to performance mode..."
sudo cpufreq-set -g performance 2>/dev/null || echo "CPU governor setting skipped"

# Set process priority
echo "Starting calculation with high priority..."

# Choice: API server or direct calculation
echo "Choose mode:"
echo "1) Direct calculation (text file output)"
echo "2) API server (REST endpoints)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "2" ]; then
    echo "Starting API server..."
    nice -n -10 python3 pi_api_server.py
else
    echo "Starting direct calculation..."
    nice -n -10 python3 pi_calculator_api.py
fi