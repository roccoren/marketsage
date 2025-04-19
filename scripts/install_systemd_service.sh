#!/bin/bash
# Script to install the MarketSage Dash application as a systemd service

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Get the absolute path of the project directory
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Create marketsage user if it doesn't exist
if ! id -u marketsage &>/dev/null; then
    echo "Creating marketsage user..."
    useradd -m -s /bin/bash marketsage
fi

# Update the service file with the correct paths
echo "Configuring service file..."
sed -i "s|/path/to/marketsage|$PROJECT_DIR|g" "$PROJECT_DIR/scripts/marketsage-dash.service"

# Copy the service file to systemd directory
echo "Installing systemd service..."
cp "$PROJECT_DIR/scripts/marketsage-dash.service" /etc/systemd/system/

# Set proper permissions
chown -R marketsage:marketsage "$PROJECT_DIR"

# Reload systemd to recognize the new service
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable marketsage-dash.service

echo "Service installed. You can start it with: sudo systemctl start marketsage-dash.service"
echo "To check status: sudo systemctl status marketsage-dash.service"
echo "To view logs: sudo journalctl -u marketsage-dash.service"

exit 0