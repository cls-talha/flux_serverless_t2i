FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# Install all dependencies
RUN pip install --upgrade pip
RUN pip install rembg onnxruntime asyncer filetype
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-u", "rp_handler.py"]
