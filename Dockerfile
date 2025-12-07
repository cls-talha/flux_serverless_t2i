FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    wget \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision --upgrade
RUN pip install libcom

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "rp_handler.py"]
