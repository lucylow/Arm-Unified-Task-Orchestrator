
# Use a Python base image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=api_server.py
ENV FLASK_RUN_HOST=0.0.0.0

# Install system dependencies for Tesseract OCR and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app/autorl_project

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the Flask API port and Prometheus metrics port
EXPOSE 5000
EXPOSE 9000

# Command to run the Flask API server
CMD ["python", "api_server.py"]

