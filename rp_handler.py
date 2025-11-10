import os
import io
import base64
import random
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from PIL import Image
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = None
HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token from environment
LORA_REPO = "strangerzonehf/Flux-Super-Realism-LoRA"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"

def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN environment variable is not set")
        login(token=HF_TOKEN)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        PIPELINE = DiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            use_auth_token=True
        ).to(DEVICE)
        PIPELINE.load_lora_weights(LORA_REPO)
    return PIPELINE

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
    try:
        inputs = job.get("input", {})
        prompt = inputs.get("prompt", "Close-up portrait of a girl, photorealistic")
        width = int(inputs.get("width", 720))
        height = int(inputs.get("height", 1024))
        num_inference_steps = int(inputs.get("num_inference_steps", 28))
        guidance_scale = float(inputs.get("guidance_scale", 4.5))
        seed = inputs.get("seed")
        if seed is None:
            seed = random.randint(0, 999999)
        else:
            seed = int(seed)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        pipe = load_pipeline()

        img = pipe(
            prompt="Super Realism " + prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]

        img_base64 = pil_to_base64(img)
        return {"status": "success", "seed": seed, "image_base64": img_base64}

    except Exception as e:
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": handler})
