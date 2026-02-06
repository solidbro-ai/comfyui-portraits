#!/usr/bin/env python3
"""
ComfyUI Reference Portrait Generator with Face Consistency via IP-Adapter-FaceID
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

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://10.10.10.13:8188")
REFERENCE_DIR = Path("./reference")
OUTPUT_DIR = Path(__file__).parent / "output"

OUTPUT_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

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

HAIR_STYLES = [
    "short", "buzz cut", "crew cut", "undercut", "pompadour", "quiff",
    "slicked back", "messy", "wavy", "curly", "long straight", "long layered",
    "ponytail", "bun", "braids", "dreads", "afro", "pixie cut", "bob"
]

HAIR_COLORS = [
    "black", "dark brown", "brown", "light brown", "blonde", "platinum blonde",
    "strawberry blonde", "red", "auburn", "gray", "silver", "white",
    "dyed blue", "dyed purple", "dyed pink"
]

EYE_COLORS = [
    "brown", "hazel", "green", "blue", "gray", "amber", "heterochromia"
]

def build_prompt(
    gender=None, age=None, ethnicity=None, framing="headshot", pose=None,
    lighting=None, time_of_day=None, location=None, background=None,
    camera=None, lens=None, style="photorealistic", clothing=None,
    hair=None, hair_color=None, eye_color=None, additional=None
):
    gender = gender or random.choice(GENDERS)
    age = age or random.choice(AGES)
    ethnicity = ethnicity or random.choice(ETHNICITIES)
    pose = pose or random.choice(POSES)
    lighting = lighting or random.choice(LIGHTING)
    time_of_day = time_of_day or random.choice(TIME_OF_DAY)
    location = location or random.choice(LOCATIONS)
    camera = camera or random.choice(CAMERAS)
    lens = lens or random.choice(LENSES)
    hair = hair or random.choice(HAIR_STYLES)
    hair_color = hair_color or random.choice(HAIR_COLORS)
    eye_color = eye_color or random.choice(EYE_COLORS)

    framing_desc = FRAMING.get(framing, FRAMING["headshot"])

    parts = [
        f"professional {style} photograph",
        f"{framing_desc}",
        f"of a {age} {ethnicity} {gender}",
        f"{pose}",
        clothing and f"wearing {clothing}",
        f"with {hair_color} {hair} hair",
        f"{eye_color} eyes",
        f"{lighting}",
        f"during {time_of_day}",
        f"at {location}",
        background and f"with {background} background",
        f"shot on {camera} with {lens}",
        "8k uhd, highly detailed, sharp focus, professional photography",
        additional
    ]

    return ", ".join(p for p in parts if p)

def get_reference_images():
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [p for p in REFERENCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]

def build_workflow(prompt, negative, width, height, steps, cfg, seed=None, ref_image=None):
    seed = seed or random.randint(0, 2**32 - 1)
    
    # Get first reference image filename
    refs = get_reference_images()
    if not refs:
        raise ValueError("No reference images in ./reference/")
    ref_name = refs[0].name

    return {
        "prompt": {
            "4": {
                "inputs": {"ckpt_name": "RealVisXL_V4.0.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {"text": prompt, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {"text": negative, "clip": ["4", 1]},
                "class_type": "CLIPTextEncode"
            },
            "load_ref": {
                "inputs": {"image": ref_name, "upload": "image"},
                "class_type": "LoadImage"
            },
            "ip_loader": {
                "inputs": {
                    "model": ["4", 0],
                    "preset": "FACEID PLUS V2",
                    "lora_strength": 0.6,
                    "provider": "CUDA"
                },
                "class_type": "IPAdapterUnifiedLoaderFaceID"
            },
            "insightface": {
                "inputs": {"provider": "CUDA"},
                "class_type": "IPAdapterInsightFaceLoader"
            },
            "faceid": {
                "inputs": {
                    "model": ["ip_loader", 0],
                    "ipadapter": ["ip_loader", 1],
                    "image": ["load_ref", 0],
                    "weight": 0.85,
                    "weight_faceidv2": 1.0,
                    "weight_type": "linear",
                    "combine_embeds": "concat",
                    "start_at": 0.0,
                    "end_at": 1.0,
                    "embeds_scaling": "V only",
                    "insightface": ["insightface", 0]
                },
                "class_type": "IPAdapterFaceID"
            },
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["faceid", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "8": {
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {"filename_prefix": "ref_portrait", "images": ["8", 0]},
                "class_type": "SaveImage"
            }
        }
    }

def queue_prompt(workflow):
    resp = requests.post(urljoin(COMFYUI_HOST, "/prompt"), json=workflow, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(urljoin(COMFYUI_HOST, f"/history/{prompt_id}"))
        if resp.ok and prompt_id in resp.json():
            return resp.json()[prompt_id]
        time.sleep(2)
    raise TimeoutError("Generation timed out")

def download_image(filename, subfolder=""):
    params = {"filename": filename}
    if subfolder:
        params["subfolder"] = subfolder
    resp = requests.get(urljoin(COMFYUI_HOST, "/view"), params=params, timeout=60)
    resp.raise_for_status()
    path = OUTPUT_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    path.write_bytes(resp.content)
    return path

def generate_reference(**kwargs):
    refs = get_reference_images()
    if not refs:
        raise ValueError("No images found in ./reference/")

    print(f"Using {len(refs)} reference image(s)")

    prompt = build_prompt(
        gender=kwargs.get('gender'),
        age=kwargs.get('age'),
        ethnicity=kwargs.get('ethnicity'),
        framing=kwargs.get('framing', 'headshot'),
        pose=kwargs.get('pose'),
        lighting=kwargs.get('lighting'),
        time_of_day=kwargs.get('time_of_day'),
        location=kwargs.get('location'),
        background=kwargs.get('background'),
        camera=kwargs.get('camera'),
        lens=kwargs.get('lens'),
        style=kwargs.get('style', 'photorealistic'),
        clothing=kwargs.get('clothing'),
        hair=kwargs.get('hair'),
        hair_color=kwargs.get('hair_color'),
        eye_color=kwargs.get('eye_color'),
        additional=kwargs.get('additional')
    )

    negative = kwargs.get("negative") or "blurry, low quality, deformed, bad anatomy, cartoon, 3d render, ugly"

    resolution = kwargs.get("resolution", "hd")
    aspect = kwargs.get("aspect", "portrait")
    base_res = RESOLUTIONS.get(resolution, (768, 768))
    ratio = ASPECT_RATIOS.get(aspect, (3, 4))

    if ratio[0] > ratio[1]:
        width = base_res[0]
        height = int(width * ratio[1] / ratio[0])
    else:
        height = base_res[1]
        width = int(height * ratio[0] / ratio[1])

    width = (width // 8) * 8
    height = (height // 8) * 8

    quality = kwargs.get("quality", "normal")
    qual = QUALITIES.get(quality, {"steps": 30, "cfg": 7.5})

    print(f"Prompt: {prompt[:100]}...")
    print(f"Size: {width}x{height}, Steps: {qual['steps']}")

    wf = build_workflow(prompt, negative, width, height, qual["steps"], qual["cfg"], kwargs.get("seed"))

    pid = queue_prompt(wf)
    result = wait_for_completion(pid)

    for node_id, output in result.get("outputs", {}).items():
        if "images" in output:
            for img in output["images"]:
                path = download_image(img["filename"], img.get("subfolder", ""))
                print(f"Saved: {path}")
                return path

    raise RuntimeError("No image generated")

def main():
    parser = argparse.ArgumentParser(description="Generate portrait using reference images")
    parser.add_argument("--gender", choices=GENDERS)
    parser.add_argument("--age", choices=AGES)
    parser.add_argument("--ethnicity", choices=ETHNICITIES)
    parser.add_argument("--clothing")
    parser.add_argument("--hair", choices=HAIR_STYLES)
    parser.add_argument("--hair-color")
    parser.add_argument("--eye-color", choices=EYE_COLORS)
    parser.add_argument("--framing", choices=list(FRAMING.keys()), default="headshot")
    parser.add_argument("--pose")
    parser.add_argument("--lighting")
    parser.add_argument("--time", dest="time_of_day")
    parser.add_argument("--location")
    parser.add_argument("--background")
    parser.add_argument("--camera")
    parser.add_argument("--lens")
    parser.add_argument("--style", default="photorealistic")
    parser.add_argument("--additional")
    parser.add_argument("--quality", choices=list(QUALITIES.keys()), default="normal")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS.keys()), default="hd")
    parser.add_argument("--aspect", choices=list(ASPECT_RATIOS.keys()), default="portrait")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    try:
        path = generate_reference(
            gender=args.gender, age=args.age, ethnicity=args.ethnicity,
            clothing=args.clothing, framing=args.framing, pose=args.pose,
            lighting=args.lighting, time_of_day=args.time_of_day, location=args.location,
            background=args.background, camera=args.camera, lens=args.lens,
            style=args.style, additional=args.additional,
            hair=args.hair, hair_color=args.hair_color, eye_color=args.eye_color,
            quality=args.quality, resolution=args.resolution, aspect=args.aspect,
            seed=args.seed
        )
        print(f"Generated: {path}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
