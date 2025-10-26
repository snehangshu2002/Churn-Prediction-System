# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create necessary directories if they don't exist
RUN mkdir -p artifacts templates

# Make port 8000 available
EXPOSE 8000

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' app_user
RUN chown -R app_user:app_user /app
USER app_user

# Command to run the application
CMD ["python", "app.py"]
