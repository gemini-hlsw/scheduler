# Use a Node.js base image
FROM node:lts-alpine

# Enable corepack
RUN corepack enable && corepack prepare pnpm@latest --activate

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Copy the rest of the application code
COPY . .

# Build the Vite application for production
RUN pnpm build

# Expose the port your Vite app will run on (e.g., 5173 for development, 80 for production with a server)
EXPOSE 4173

# Command to run the application (e.g., serving static files with a web server like Nginx)
CMD ["pnpm", "preview"]