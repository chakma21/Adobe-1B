# 1. Set the base image with the required platform
FROM --platform=linux/amd64 python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Install Python libraries, ensuring CPU-only PyTorch for a smaller image
# This respects the CPU-only constraint [cite: 1519]
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# 5. Copy and run the model download script
# This ensures the model is pre-packaged in the image to meet the "no internet access" rule 
COPY download_model.py .
RUN python download_model.py

# 6. Copy the rest of your application files
# This includes your main logic and the required explanation file [cite: 1526]
COPY main.py .
COPY approach_explanation.md .

# 7. Set the command to run when the container starts
CMD ["python", "main.py"]