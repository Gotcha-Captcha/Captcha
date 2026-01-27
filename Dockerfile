# Use the official Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
# build-essential is kept for potential C-extensions (scikit-image/torch/etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Cloud Run sets PORT environment variable, default to 80)
ENV PORT=80
EXPOSE 80

# Command to run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
