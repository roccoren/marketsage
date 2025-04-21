#!/bin/bash
set -e

# Replace environment variables in nginx configuration
envsubst '${API_URL}' < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf.tmp
mv /etc/nginx/conf.d/default.conf.tmp /etc/nginx/conf.d/default.conf

# Replace environment variables in the JavaScript files
# Find all JS files in the build directory
find /usr/share/nginx/html -type f -name "*.js" | while read -r file; do
    # Replace environment variables in each file
    sed -i "s|__VITE_API_URL__|${VITE_API_URL}|g" "$file"
    sed -i "s|__VITE_AUTH_TOKEN_KEY__|${VITE_AUTH_TOKEN_KEY}|g" "$file"
    sed -i "s|__VITE_REFRESH_TOKEN_KEY__|${VITE_REFRESH_TOKEN_KEY}|g" "$file"
done

# Execute the main container command
exec "$@"