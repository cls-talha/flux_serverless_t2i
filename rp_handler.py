import os
import io
import base64
import random
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FluxKontextPipeline

from huggingface_hub import login
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIPELINE_T2I = None
PIPELINE_I2I = None

HF_TOKEN = os.environ.get("HF_TOKEN")
LORA_REPO = "strangerzonehf/Flux-Super-Realism-LoRA"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"


# ------------------------
# Pipeline Loaders t2i
# ------------------------
def load_t2i_pipeline():
    free_i2i()
    global PIPELINE_T2I
    if PIPELINE_T2I is None:
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN environment variable is not set")
        login(token=HF_TOKEN)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        PIPELINE_T2I = DiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            use_auth_token=True
        ).to(DEVICE)
    return PIPELINE_T2I


def enable_lora(pipe):
    pipe.load_lora_weights(LORA_REPO)


def disable_lora(pipe):
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

# ------------------------
# Pipeline Loaders i2i
# ------------------------
def load_i2i_pipeline():
    free_t2i()
    global PIPELINE_I2I
    if PIPELINE_I2I is None:
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN environment variable is not set")

        login(token=HF_TOKEN)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        PIPELINE_I2I = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=dtype,
            use_auth_token=True
        ).to(DEVICE)

    return PIPELINE_I2I


# ------------------------
# Helpers
# ------------------------
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

# ------------------------
# CUDA memory cleanup
# ------------------------
def free_t2i():
    global PIPELINE_T2I
    if PIPELINE_T2I is not None:
        del PIPELINE_T2I
        PIPELINE_T2I = None
        torch.cuda.empty_cache()


def free_i2i():
    global PIPELINE_I2I
    if PIPELINE_I2I is not None:
        del PIPELINE_I2I
        PIPELINE_I2I = None
        torch.cuda.empty_cache()


# ------------------------
# Handler
# ------------------------
def handler(job):
    try:
        inputs = job.get("input", {})
        case = inputs.get("case", "generate")

        prompt = inputs.get("prompt", "Close-up portrait of a girl, photorealistic")
        width = int(inputs.get("width", 720))
        height = int(inputs.get("height", 1024))
        steps = int(inputs.get("num_inference_steps", 28))
        cfg = float(inputs.get("guidance_scale", 4.5))
        strength = float(inputs.get("strength", 0.8))
        seed = inputs.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 999999)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # -------------------------
        # TEXT → IMAGE
        # -------------------------
        if case in ["generate", "generate_foreground"]:
            pipe = load_t2i_pipeline()
        
            # Only enable LoRA if you want it
            if case == "generate_foreground":   # or some other condition
                enable_lora(pipe)
        
            if case == "generate_foreground":
                prompt_full = f"Super Realism {prompt}, pure white background"
            else:
                prompt_full = f"Super Realism {prompt}"
        
            img = pipe(
                prompt=prompt_full,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                output_type="pil"
            ).images[0]
            img.save("img.png")

        # -------------------------
        # IMAGE → IMAGE
        # -------------------------
        elif case == "image_to_image":
            pipe = load_i2i_pipeline()
        
            image_b64 = inputs.get("image")
            if image_b64 is None:
                return {"status": "failed", "error": "image (base64) required for i2i"}
        
            init_image = base64_to_pil(image_b64)
        
            orig_w, orig_h = init_image.size
            gen_w = (orig_w // 16) * 16
            gen_h = (orig_h // 16) * 16
        
            negative_prompt = (
                "deformed face, distorted facial features, bad anatomy, "
                "asymmetrical face, extra facial features, duplicate face, "
                "disfigured nose, warped eyes, cross-eyed, "
                "unnatural skin texture, melted face, blurry face, low detail, "
                "oversharpened, plastic skin, uncanny"
            )
        
            img = pipe(
                image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                num_inference_steps=steps,
                width=gen_w,
                height=gen_h
            ).images[0]
            img.save("img2img.png")

        else:
            return {"status": "failed", "error": f"Unknown case: {case}"}

        return {"status": "success", "seed": seed, "image_base64": pil_to_base64(img)}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


runpod.serverless.start({"handler": handler})
