#!/bin/bash
# Script to prepare the deployment package for Azure App Service

set -e

# Navigate to the project root directory
cd "$(dirname "$0")/../../.."

# Check if zip command is available
if ! command -v zip &> /dev/null; then
    echo "Error: zip command is not installed. Please install it first."
    exit 1
fi

echo "Preparing deployment package for Azure App Service..."

# Create a temporary directory for the deployment package
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Copy necessary files to the temporary directory
echo "Copying files to temporary directory..."

# Copy Python files
cp dash_app.py "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"

# Copy modules directory
mkdir -p "$TEMP_DIR/modules"
cp -r modules/* "$TEMP_DIR/modules/"

# Copy config directory
mkdir -p "$TEMP_DIR/config"
cp config/default_config.yaml "$TEMP_DIR/config/"

# Copy assets directory if it exists
if [ -d "assets" ]; then
    mkdir -p "$TEMP_DIR/assets"
    cp -r assets/* "$TEMP_DIR/assets/"
fi

# Create .deployment file for Azure
cat > "$TEMP_DIR/.deployment" << EOL
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT=true
EOL

# Create startup command file
cat > "$TEMP_DIR/startup.txt" << EOL
gunicorn --bind=0.0.0.0 --timeout 600 dash_app:server
EOL

# Create web.config file for Azure App Service
cat > "$TEMP_DIR/web.config" << EOL
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <appSettings>
    <add key="PYTHONPATH" value="%HOME%" />
    <add key="WSGI_HANDLER" value="dash_app.server" />
    <add key="WSGI_LOG" value="d:\\home\\LogFiles\\application.log" />
  </appSettings>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="FastCgiModule" scriptProcessor="d:\\home\\python364x86\\python.exe|d:\\home\\python364x86\\wfastcgi.py" resourceType="Unspecified" requireAccess="Script" />
    </handlers>
  </system.webServer>
</configuration>
EOL

# Create the deployment package
echo "Creating deployment package..."
cd "$TEMP_DIR"
zip -r deploy.zip ./*
mv deploy.zip "$(cd - > /dev/null && pwd)/"

# Clean up
echo "Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"

echo "Deployment package created: $(pwd)/deploy.zip"
echo "You can now deploy this package using the deploy_app_service.sh script."

exit 0