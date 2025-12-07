import os
import io
import base64
import random
import torch
import cv2
import numpy as np
from PIL import Image
from libcom import ImageHarmonizationModel
from diffusers import DiffusionPipeline
from huggingface_hub import login
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE = None
HF_TOKEN = os.environ.get("HF_TOKEN")
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

def base64_to_image(b64: str) -> str:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    temp_path = "/tmp/temp_input.png"
    img.save(temp_path)
    return temp_path

def erode_mask(mask, shrink_pixels):
    if shrink_pixels <= 0:
        return mask
    k = shrink_pixels * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)

def process_pipeline(foreground_path, background_path, shrink_pixels=5, out_dir="/tmp/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    out_fg = f"{out_dir}/out_fg.png"
    rgba_path = f"{out_dir}/foreground_rgba.png"
    mask_raw_path = f"{out_dir}/mask_raw.png"
    mask_shrink_path = f"{out_dir}/mask_shrink.png"
    composite_path = f"{out_dir}/composite.png"
    harmonized_path = f"{out_dir}/harmonized.png"

    os.system(f"rembg i -m birefnet-portrait {foreground_path} {out_fg}")
    Image.open(out_fg).convert("RGBA").save(rgba_path)

    fg = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(background_path)
    if fg.shape[:2] != bg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    alpha = fg[:, :, 3]
    mask_raw = np.where(alpha > 0, 255, 0).astype(np.uint8)
    mask_shrink = erode_mask(mask_raw, shrink_pixels)

    fg_rgb = fg[:, :, :3]
    alpha_norm = alpha.astype(np.float32) / 255.0
    composite = bg.copy()
    for c in range(3):
        composite[:, :, c] = alpha_norm * fg_rgb[:, :, c] + (1 - alpha_norm) * composite[:, :, c]

    cv2.imwrite(mask_raw_path, mask_raw)
    cv2.imwrite(mask_shrink_path, mask_shrink)
    cv2.imwrite(composite_path, composite)

    img = cv2.imread(composite_path)
    mask_h = cv2.imread(mask_shrink_path, cv2.IMREAD_GRAYSCALE)
    PCTNet = ImageHarmonizationModel(device=0, model_type='PCTNet')
    harmonized = PCTNet(img, mask_h)
    cv2.imwrite(harmonized_path, harmonized)

    return harmonized_path

def handler(job):
    try:
        inputs = job.get("input", {})
        case = inputs.get("case", "generate")
        shrink_pixels = int(inputs.get("shrink_pixels", 5))

        if case == "generate":
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

        elif case == "generate_foreground":
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
                prompt="Super Realism " + prompt + ", white background",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil"
            ).images[0]
            img_base64 = pil_to_base64(img)
            return {"status": "success", "seed": seed, "image_base64": img_base64}

        elif case == "harmonize":
            fg_b64 = inputs.get("foreground_image")
            bg_b64 = inputs.get("background_image")
            if fg_b64 is None or bg_b64 is None:
                return {"status": "failed", "error": "foreground_image or background_image missing"}
            fg_path = base64_to_image(fg_b64)
            bg_path = base64_to_image(bg_b64)
            harmonized_path = process_pipeline(fg_path, bg_path, shrink_pixels)
            harmonized_img = Image.open(harmonized_path)
            harmonized_b64 = pil_to_base64(harmonized_img)
            return {"status": "success", "image_base64": harmonized_b64}

        else:
            return {"status": "failed", "error": f"Unknown case: {case}"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": handler})
