# Use a pinned Debian base so package names stay stable.
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 5000

# Install system dependencies for Tesseract OCR and TensorFlow/Pillow runtime support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the application with Gunicorn on Railway's injected PORT.
CMD ["sh", "-c", "gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:${PORT:-5000} backend.app:app"]
