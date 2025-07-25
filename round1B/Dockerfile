# 1. Set the base image with the required platform
FROM --platform=linux/amd64 python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ALL requirements files
COPY requirements.txt .
COPY torch_requirements.txt .

# 4. Install libraries in two steps
RUN pip install --no-cache-dir -r torch_requirements.txt --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and the input files
COPY main.py .
COPY input_payload.json .
COPY documents/ ./documents/

# 6. Set the command to run when the container starts
CMD ["python", "main.py"]