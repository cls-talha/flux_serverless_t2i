FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install libcom rembg onnxruntime asyncer filetype
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-u", "rp_handler.py"]
