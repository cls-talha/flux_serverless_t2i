FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Install git and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & setup tools
RUN python -m ensurepip --upgrade
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install torch & torchvision
RUN pip install --no-cache-dir torch torchvision

# Install libcom (will also pull chumpy and other deps)
RUN pip install --no-cache-dir libcom

WORKDIR /workspace

# Install your remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Start handler
CMD ["python", "-u", "rp_handler.py"]
