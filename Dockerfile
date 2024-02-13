# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         git \
         libglib2.0-0 \
         libsm6 \
         libxext6 \
         libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port 8080 available outside this container
EXPOSE 8080

# Define environment variable for model directory (adjust as needed)
ENV MODEL_DIR=/app/models

# Use ENTRYPOINT to specify the executable and CMD for default arguments
ENTRYPOINT ["python3", "main.py"]

# CMD provides default arguments if not overridden
CMD ["--config", "config.yaml", "--lora-name", "easter_egg", "--base-model", "runwayml/stable-diffusion-v1-5", "--text-prompt", "easter, A pokemon with blue eyes", "--fuse-lora-scale", "0.9", "--height", "640", "--width", "512", "--num-inference-steps", "15", "--guidance-scale", "7.6", "--seed-num", "3", "--num-imgs", "3"]