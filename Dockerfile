# Fetch the runtime model via Git LFS so Docker builds don't depend on LFS support
# in the source checkout itself.
FROM python:3.11-slim-bookworm AS model-fetch

ARG MODEL_REPO_URL=https://github.com/html-proof/fake.git
ARG MODEL_GIT_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/model-repo

RUN git clone --depth 1 --branch "${MODEL_GIT_REF}" "${MODEL_REPO_URL}" . \
    && git lfs install --local \
    && git lfs pull --include="backend/model/fake_job_bilstm.keras" --exclude="" \
    && test -s backend/model/fake_job_bilstm.keras

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

# Replace the LFS pointer from the build context with the real model file.
COPY --from=model-fetch /tmp/model-repo/backend/model/fake_job_bilstm.keras /app/backend/model/fake_job_bilstm.keras

# Ensure the startup script is executable inside the container.
RUN chmod +x /app/start.sh

# Expose the port the app runs on
EXPOSE 5000

# Run the application with the shared startup script.
CMD ["./start.sh"]
