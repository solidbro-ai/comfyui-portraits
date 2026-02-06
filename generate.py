#!/usr/bin/env python3
"""
ComfyUI Portrait Generator

Generate realistic photos of people via ComfyUI API.
"""

import os
import json
import random
import requests
import uuid
import time
import argparse
from pathlib import Path
from urllib.parse import urljoin

# Configuration
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://localhost:8188")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# === PROMPT BUILDING OPTIONS ===

GENDERS = ["male", "female"]
AGES = ["young adult", "middle-aged", "elderly", "teenage", "20s", "30s", "40s", "50s", "60s"]
ETHNICITIES = ["caucasian", "african", "asian", "hispanic", "middle eastern", "south asian", "mixed race"]

FRAMING = {
    "profile": "profile photo, head and shoulders, passport style",
    "headshot": "professional headshot, face and neck visible",
    "shoulders": "shoulders up portrait, upper chest visible", 
    "torso": "torso up, waist up portrait, upper body visible",
    "three_quarter": "three quarter body shot, knees up",
    "full_body": "full body portrait, entire person visible head to toe"
}

POSES = [
    "looking directly at camera", "looking slightly left", "looking slightly right",
    "looking up", "looking down", "profile view", "three quarter view",
    "arms crossed", "hands in pockets", "leaning against wall", "sitting",
    "standing confidently", "walking", "candid natural pose", "laughing",
    "smiling warmly", "serious expression", "thoughtful expression", "relaxed"
]

LIGHTING = [
    "natural soft lighting", "golden hour sunlight", "studio lighting",
    "dramatic side lighting", "backlit silhouette rim light", "overcast diffused light",
    "harsh midday sun", "neon lights", "warm interior lighting", "cool blue hour",
    "ring light", "rembrandt lighting", "butterfly lighting", "split lighting",
    "window light", "candlelight", "fluorescent office lighting"
]

TIME_OF_DAY = [
    "sunrise", "morning", "midday", "afternoon", "golden hour", "sunset",
    "blue hour", "dusk", "evening", "night"
]

LOCATIONS = [
    "professional studio with neutral backdrop", "modern office",
    "urban city street", "cafe interior", "park with trees",
    "beach", "mountain landscape", "forest", "rooftop with city skyline",
    "home interior living room", "library", "gym", "restaurant",
    "art gallery", "industrial warehouse", "garden", "university campus",
    "hotel lobby", "train station", "countryside", "desert"
]

BACKGROUNDS = [
    "solid neutral gray", "solid white", "solid black", "blurred bokeh",
    "gradient", "textured wall", "brick wall", "window with natural light",
    "outdoor scenery", "indoor setting", "abstract", "minimalist"
]

CAMERAS = [
    "Canon EOS R5", "Sony A7R IV", "Nikon Z9", "Hasselblad X2D",
    "Leica M11", "Fujifilm GFX 100S", "Phase One IQ4", "Canon 5D Mark IV",
    "Sony A1", "Nikon D850"
]

LENSES = [
    "85mm f/1.4 portrait lens", "50mm f/1.2", "35mm f/1.4",
    "70-200mm f/2.8 telephoto", "24-70mm f/2.8", "105mm f/2.8 macro",
    "135mm f/2", "200mm f/2"
]

STYLES = [
    "photorealistic", "editorial fashion", "commercial advertising",
    "lifestyle candid", "corporate professional", "artistic portrait",
    "documentary", "glamour", "natural authentic", "cinematic"
]

CLOTHING = [
    "business suit", "casual t-shirt and jeans", "formal dress",
    "athletic wear", "smart casual", "streetwear", "traditional cultural attire",
    "summer dress", "winter coat", "professional uniform", "elegant evening wear"
]

QUALITIES = {
    "draft": {"steps": 20, "cfg": 7},
    "normal": {"steps": 30, "cfg": 7.5},
    "high": {"steps": 40, "cfg": 8},
    "ultra": {"steps": 50, "cfg": 8.5}
}

RESOLUTIONS = {
    "sd": (512, 512),
    "hd": (768, 768),
    "full_hd": (1024, 1024),
    "2k": (1280, 1280),
    "4k": (2048, 2048)
}

ASPECT_RATIOS = {
    "square": (1, 1),
    "portrait": (3, 4),
    "portrait_tall": (2, 3),
    "landscape": (4, 3),
    "landscape_wide": (16, 9),
    "instagram": (4, 5),
    "story": (9, 16)
}


def build_prompt(
    gender: str = None,
    age: str = None,
    ethnicity: str = None,
    framing: str = "headshot",
    pose: str = None,
    lighting: str = None,
    time_of_day: str = None,
    location: str = None,
    background: str = None,
    camera: str = None,
    lens: str = None,
    style: str = "photorealistic",
    clothing: str = None,
    additional: str = None
) -> str:
    """Build a detailed prompt for portrait generation."""
    
    # Random defaults
    gender = gender or random.choice(GENDERS)
    age = age or random.choice(AGES)
    ethnicity = ethnicity or random.choice(ETHNICITIES)
    pose = pose or random.choice(POSES)
    lighting = lighting or random.choice(LIGHTING)
    time_of_day = time_of_day or random.choice(TIME_OF_DAY)
    location = location or random.choice(LOCATIONS)
    camera = camera or random.choice(CAMERAS)
    lens = lens or random.choice(LENSES)
    
    framing_desc = FRAMING.get(framing, FRAMING["headshot"])
    
    parts = [
        f"professional {style} photograph",
        f"{framing_desc}",
        f"of a {age} {ethnicity} {gender}",
        f"{pose}",
        clothing and f"wearing {clothing}",
        f"{lighting}",
        f"during {time_of_day}",
        f"at {location}",
        background and f"with {background} background",
        f"shot on {camera} with {lens}",
        "8k uhd, highly detailed, sharp focus, professional photography",
        additional
    ]
    
    prompt = ", ".join(p for p in parts if p)
    return prompt


def get_workflow(prompt: str, negative: str, width: int, height: int, 
                 steps: int, cfg: float, seed: int = None) -> dict:
    """Generate ComfyUI workflow JSON."""
    
    seed = seed or random.randint(1, 2**32-1)
    
    # Basic SDXL workflow - adjust checkpoint name to match your model
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": steps
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "RealVisXL_V4.0.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": height,
                "width": width
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": prompt
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": negative
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "portrait",
                "images": ["8", 0]
            }
        }
    }
    
    return {"prompt": workflow}


def queue_prompt(workflow: dict) -> str:
    """Queue a prompt in ComfyUI and return prompt_id."""
    resp = requests.post(
        urljoin(COMFYUI_HOST, "/prompt"),
        json=workflow,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Wait for prompt to complete and return result."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(urljoin(COMFYUI_HOST, f"/history/{prompt_id}"))
        if resp.ok:
            history = resp.json()
            if prompt_id in history:
                return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Generation timed out after {timeout}s")


def download_image(filename: str, subfolder: str = "") -> Path:
    """Download generated image from ComfyUI."""
    params = {"filename": filename}
    if subfolder:
        params["subfolder"] = subfolder
    
    resp = requests.get(
        urljoin(COMFYUI_HOST, "/view"),
        params=params,
        timeout=60
    )
    resp.raise_for_status()
    
    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    output_path.write_bytes(resp.content)
    return output_path


def generate(
    # Person attributes
    gender: str = None,
    age: str = None, 
    ethnicity: str = None,
    clothing: str = None,
    
    # Shot settings
    framing: str = "headshot",
    pose: str = None,
    
    # Environment
    lighting: str = None,
    time_of_day: str = None,
    location: str = None,
    background: str = None,
    
    # Camera settings
    camera: str = None,
    lens: str = None,
    
    # Style
    style: str = "photorealistic",
    additional: str = None,
    
    # Technical
    quality: str = "normal",
    resolution: str = "hd",
    aspect: str = "portrait",
    seed: int = None,
    
    # Output
    negative: str = None
) -> Path:
    """Generate a portrait photo."""
    
    # Build prompt
    prompt = build_prompt(
        gender=gender, age=age, ethnicity=ethnicity, clothing=clothing,
        framing=framing, pose=pose, lighting=lighting, time_of_day=time_of_day,
        location=location, background=background, camera=camera, lens=lens,
        style=style, additional=additional
    )
    
    # Default negative prompt
    if negative is None:
        negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, missing fingers, extra fingers, mutated, disfigured, watermark, text, signature, cartoon, anime, illustration, painting, drawing, cgi, 3d render"
    
    # Calculate dimensions
    base_res = RESOLUTIONS.get(resolution, RESOLUTIONS["hd"])
    ratio = ASPECT_RATIOS.get(aspect, ASPECT_RATIOS["portrait"])
    
    if ratio[0] > ratio[1]:  # Landscape
        width = base_res[0]
        height = int(base_res[0] * ratio[1] / ratio[0])
    else:  # Portrait or square
        height = base_res[1]
        width = int(base_res[1] * ratio[0] / ratio[1])
    
    # Round to nearest 8 (required by most models)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    qual = QUALITIES.get(quality, QUALITIES["normal"])
    
    print(f"📸 Generating portrait...")
    print(f"   Prompt: {prompt[:100]}...")
    print(f"   Size: {width}x{height}, Steps: {qual['steps']}")
    
    # Generate workflow
    workflow = get_workflow(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps=qual["steps"],
        cfg=qual["cfg"],
        seed=seed
    )
    
    # Queue and wait
    prompt_id = queue_prompt(workflow)
    print(f"   Queued: {prompt_id}")
    
    result = wait_for_completion(prompt_id)
    
    # Download image
    outputs = result.get("outputs", {})
    for node_id, output in outputs.items():
        if "images" in output:
            for img in output["images"]:
                path = download_image(img["filename"], img.get("subfolder", ""))
                print(f"   ✅ Saved: {path}")
                return path
    
    raise RuntimeError("No image generated")


def main():
    parser = argparse.ArgumentParser(description="Generate portrait photos via ComfyUI")
    
    # Person
    parser.add_argument("--gender", choices=GENDERS)
    parser.add_argument("--age", choices=AGES)
    parser.add_argument("--ethnicity", choices=ETHNICITIES)
    parser.add_argument("--clothing")
    
    # Shot
    parser.add_argument("--framing", choices=list(FRAMING.keys()), default="headshot")
    parser.add_argument("--pose")
    
    # Environment  
    parser.add_argument("--lighting")
    parser.add_argument("--time", dest="time_of_day")
    parser.add_argument("--location")
    parser.add_argument("--background")
    
    # Camera
    parser.add_argument("--camera")
    parser.add_argument("--lens")
    
    # Style
    parser.add_argument("--style", default="photorealistic")
    parser.add_argument("--additional", help="Additional prompt text")
    
    # Technical
    parser.add_argument("--quality", choices=list(QUALITIES.keys()), default="normal")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS.keys()), default="hd")
    parser.add_argument("--aspect", choices=list(ASPECT_RATIOS.keys()), default="portrait")
    parser.add_argument("--seed", type=int)
    
    # Random mode
    parser.add_argument("--random", action="store_true", help="Randomize all options")
    
    args = parser.parse_args()
    
    try:
        path = generate(
            gender=args.gender,
            age=args.age,
            ethnicity=args.ethnicity,
            clothing=args.clothing,
            framing=args.framing,
            pose=args.pose,
            lighting=args.lighting,
            time_of_day=args.time_of_day,
            location=args.location,
            background=args.background,
            camera=args.camera,
            lens=args.lens,
            style=args.style,
            additional=args.additional,
            quality=args.quality,
            resolution=args.resolution,
            aspect=args.aspect,
            seed=args.seed
        )
        print(f"\n✨ Generated: {path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
