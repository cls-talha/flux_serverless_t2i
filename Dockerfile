FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

    
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision
RUN pip install libcom

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "rp_handler.py"]
