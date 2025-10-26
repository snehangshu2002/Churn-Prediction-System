# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY app.py .
COPY templates/ ./templates/

# Create artifacts directory
RUN mkdir -p artifacts

# Copy specific model files
COPY artifacts/model.pkl ./artifacts/
COPY artifacts/preprocessor.pkl ./artifacts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Make port 8000 available
EXPOSE 8000

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' app_user && \
    chown -R app_user:app_user /app
USER app_user

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application
CMD ["python", "app.py"]
