# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker caching)
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# Copy remaining project files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
