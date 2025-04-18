#!/bin/bash
# Script to install the StockScan Dash application as a systemd service

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Get the absolute path of the project directory
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Create stockscan user if it doesn't exist
if ! id -u stockscan &>/dev/null; then
    echo "Creating stockscan user..."
    useradd -m -s /bin/bash stockscan
fi

# Update the service file with the correct paths
echo "Configuring service file..."
sed -i "s|/path/to/stockscan|$PROJECT_DIR|g" "$PROJECT_DIR/scripts/stockscan-dash.service"

# Copy the service file to systemd directory
echo "Installing systemd service..."
cp "$PROJECT_DIR/scripts/stockscan-dash.service" /etc/systemd/system/

# Set proper permissions
chown -R stockscan:stockscan "$PROJECT_DIR"

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable stockscan-dash.service

echo "Service installed. You can start it with: sudo systemctl start stockscan-dash.service"
echo "To check status: sudo systemctl status stockscan-dash.service"
echo "To view logs: sudo journalctl -u stockscan-dash.service"

exit 0