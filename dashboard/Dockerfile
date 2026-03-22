FROM node:20-alpine

WORKDIR /app

# Ensure fast dependency caching
COPY package.json package-lock.json* ./
RUN npm install

# Copy source
COPY . .

# Expose Vite's exact development port
EXPOSE 5173

# Expose to outside docker network via 0.0.0.0
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
