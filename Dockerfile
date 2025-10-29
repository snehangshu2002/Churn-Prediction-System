# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project including artifacts
COPY . .

# Give permissions to the app folder BEFORE switching user
RUN chown -R root:root /app

# Expose port (Render will override)
EXPOSE 8000

# Run FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
