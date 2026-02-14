import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import base64
import random
import torch
from PIL import Image

from diffusers import DiffusionPipeline, FluxKontextPipeline
from huggingface_hub import login
import runpod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_MODEL = "black-forest-labs/FLUX.1-dev"
I2I_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
LORA_REPO = "strangerzonehf/Flux-Super-Realism-LoRA"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set")

login(token=HF_TOKEN)

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def hard_cleanup():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate_t2i(prompt, width, height, steps, cfg, seed, use_lora=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        use_auth_token=True
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("max")

    if use_lora:
        pipe.load_lora_weights(LORA_REPO)

    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            output_type="pil"
        ).images[0]

    del pipe
    hard_cleanup()
    return img

def generate_i2i(init_image, prompt, steps, cfg):
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

    pipe = FluxKontextPipeline.from_pretrained(
        I2I_MODEL,
        torch_dtype=DTYPE,
        use_auth_token=True
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing("max")

    with torch.inference_mode():
        img = pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=cfg,
            num_inference_steps=steps,
            width=gen_w,
            height=gen_h
        ).images[0]

    del pipe
    hard_cleanup()
    return img

def handler(job):
    try:
        inputs = job.get("input", {})
        case = inputs.get("case", "generate")

        prompt = inputs.get("prompt", "Close-up portrait of a girl, photorealistic")
        width = int(inputs.get("width", 720))
        height = int(inputs.get("height", 1024))
        steps = int(inputs.get("num_inference_steps", 24))
        cfg = float(inputs.get("guidance_scale", 4.5))
        seed = inputs.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, 999999)

        if case in ["generate", "generate_foreground"]:
            use_lora = case == "generate_foreground"
            prompt = (
                f"Super Realism {prompt}, pure white background"
                if use_lora else f"Super Realism {prompt}"
            )

            img = generate_t2i(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                use_lora=use_lora
            )

        elif case == "image_to_image":
            image_b64 = inputs.get("image")
            if image_b64 is None:
                return {"status": "failed", "error": "image (base64) required"}

            init_image = base64_to_pil(image_b64)

            img = generate_i2i(
                init_image=init_image,
                prompt=prompt,
                steps=steps,
                cfg=cfg
            )

        else:
            return {"status": "failed", "error": f"Unknown case: {case}"}

        return {
            "status": "success",
            "seed": seed,
            "image_base64": pil_to_base64(img)
        }

    except Exception as e:
        hard_cleanup()
        return {"status": "failed", "error": str(e)}

    finally:
        hard_cleanup()

runpod.serverless.start({"handler": handler})
