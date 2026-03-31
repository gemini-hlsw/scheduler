# Use a Node.js base image
FROM node:lts-alpine AS builder

# Enable corepack
RUN corepack enable && corepack prepare pnpm@latest --activate

# Set the working directory inside the container
WORKDIR /app

# Create directories
RUN mkdir -p frontend/

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY frontend/package*.json frontend/

# Install dependencies
RUN pnpm install

# Copy the rest of the application code
COPY frontend/ frontend/
COPY schema/ schema/

# Build the Vite application for production
RUN pnpm frontend build

# --- Stage 2: Serve with Nginx ---
# Use a lightweight, stable Nginx image
FROM nginx:stable-alpine AS production

# Copy only the built static files from the 'builder' stage to the Nginx root directory
COPY --from=builder /app/frontend/dist /usr/share/nginx/html

# Copy a custom Nginx configuration file for Single Page Applications (SPAs)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Command to run Nginx, daemon off keeps the container running in the foreground
CMD ["nginx", "-g", "daemon off;"]