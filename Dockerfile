# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model files
COPY artifacts/model.pkl artifacts/
COPY artifacts/preprocessor.pkl artifacts/

# Copy remaining project files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
