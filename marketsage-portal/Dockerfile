# Build stage
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy all files
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY ./nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Add bash for entrypoint script
RUN apk add --no-cache bash

# Copy entrypoint script
COPY ./nginx/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 80

# Set environment variables that can be overridden at runtime
ENV VITE_API_URL=http://localhost:8000/api
ENV VITE_AUTH_TOKEN_KEY=marketsage_auth_token
ENV VITE_REFRESH_TOKEN_KEY=marketsage_refresh_token

# Use entrypoint script to substitute environment variables in the frontend code
ENTRYPOINT ["/entrypoint.sh"]

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]