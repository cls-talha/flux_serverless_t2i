FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install all dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your code (handler.py, any helper scripts)
COPY . .

CMD ["python", "-u", "rp_handler.py"]
