# Set the base image with the required platform and a slim Python version
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main processing script into the container
COPY main.py .

# Set the command to run when the container starts
CMD ["python", "main.py"]