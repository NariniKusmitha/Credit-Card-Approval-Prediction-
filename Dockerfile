# Use official lightweight Python image.
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in logs
ENV PYTHONUNBUFFERED=true

# Set working directory inside the container
WORKDIR /app

# Copy application files
COPY . /app

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the model training script to generate the dataset and train/save the model.pkl
# This guarantees the container has the serialized model ready on startup.
RUN python train_model.py

# Expose port 8080 (standard for IBM Cloud Code Engine and other cloud providers)
EXPOSE 8080

# Run the web service on container startup.
# We use gunicorn to serve the Flask application and bind to the PORT environment variable.
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 0 app:app
