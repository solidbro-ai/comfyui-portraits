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
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "http://10.10.10.13:8188")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# === PROMPT BUILDING OPTIONS ===

GENDERS = ["male", "female"]
AGES = ["late 20s", "30s", "40s", "50s", "60s", "70s", "middle-aged", "mature adult", "senior"]
ETHNICITIES = ["caucasian", "african", "asian", "hispanic", "middle eastern", "south asian", "mixed race"]

FRAMING = {
    "profile": "profile photo, head and shoulders, passport style",
    "headshot": "professional headshot, face and neck visible",
    "shoulders": "shoulders up portrait, upper chest visible", 
    "torso": "torso up, waist up portrait, upper body visible",
    "three_quarter": "three quarter body shot, knees up",
    "full_body": "full body portrait, entire person visible head to toe",
    "close_up": "extreme close up portrait, face filling frame",
    "tight_headshot": "tight headshot, chin to forehead",
    "environmental": "environmental portrait, person in context with surroundings",
    "bust": "bust shot, chest up portrait",
    "cowboy": "cowboy shot, mid-thigh up",
    "medium": "medium shot, hips up",
    "american": "american shot, knees up classic hollywood framing",
    "wide": "wide portrait, full body with environment",
    "over_shoulder": "over the shoulder portrait shot",
    "dutch_angle": "dutch angle portrait, tilted frame",
    "low_angle": "low angle portrait, shot from below",
    "high_angle": "high angle portrait, shot from above",
    "birds_eye": "birds eye view portrait from directly above",
    "worms_eye": "worms eye view portrait from ground level",
    "profile_silhouette": "profile silhouette portrait",
    "back_portrait": "back of head portrait, looking away",
    "candid_crop": "candid crop, natural unposed framing",
    "editorial_crop": "editorial crop, fashion magazine style framing",
    "passport": "passport photo style, neutral front facing",
    "linkedin": "linkedin professional headshot style"
}

POSES = [
    "looking directly at camera", "looking slightly left", "looking slightly right",
    "looking up", "looking down", "profile view", "three quarter view",
    "arms crossed", "hands in pockets", "leaning against wall", "sitting",
    "standing confidently", "walking", "candid natural pose", "laughing",
    "smiling warmly", "serious expression", "thoughtful expression", "relaxed",
    "hand on chin thinking", "hand touching hair", "arms behind back",
    "one hand on hip", "both hands on hips power pose", "leaning forward engaged",
    "leaning back relaxed", "sitting cross legged", "sitting on edge of chair",
    "perched on stool", "kneeling", "crouching", "mid-stride walking",
    "running motion", "jumping", "dancing", "stretching", "yoga pose",
    "meditation pose", "reading a book", "holding coffee cup", "on phone",
    "typing on laptop", "writing", "painting", "playing guitar",
    "holding microphone", "pointing", "waving", "peace sign",
    "thumbs up", "clapping", "praying hands", "arms raised celebration",
    "fist pump", "shrugging", "head tilt curious", "chin up confident",
    "looking over shoulder", "glancing sideways", "eyes closed peaceful",
    "squinting", "surprised expression", "shocked expression", "crying",
    "yelling", "whispering", "blowing kiss", "winking"
]

LIGHTING = [
    "natural soft lighting", "golden hour sunlight", "studio lighting",
    "dramatic side lighting", "backlit silhouette rim light", "overcast diffused light",
    "harsh midday sun", "neon lights", "warm interior lighting", "cool blue hour",
    "ring light", "rembrandt lighting", "butterfly lighting", "split lighting",
    "window light", "candlelight", "fluorescent office lighting",
    "moonlight", "streetlight at night", "car headlights",
    "fire light campfire glow", "christmas lights bokeh", "stadium lighting",
    "theatre spotlight", "disco ball reflections", "underwater light rays",
    "foggy diffused light", "stormy dramatic sky light"
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
    "hotel lobby", "train station", "countryside", "desert",
    "new york times square", "paris eiffel tower", "tokyo shibuya crossing",
    "london big ben", "sydney harbour bridge", "dubai skyline",
    "venice canals", "santorini greece", "bali rice terraces",
    "machu picchu", "grand canyon", "northern lights iceland",
    "african savanna", "amazon rainforest", "swiss alps",
    "maldives beach", "hong kong neon streets", "las vegas strip",
    "hollywood hills", "miami south beach", "barcelona gothic quarter",
    "amsterdam canals", "rome colosseum", "cairo pyramids",
    "moscow red square", "rio de janeiro", "shanghai pudong",
    "singapore marina bay", "seoul gangnam", "mumbai gateway",
    "toronto cn tower", "san francisco golden gate", "chicago skyline",
    "seattle space needle", "austin texas downtown", "nashville broadway",
    "new orleans french quarter", "boston harbor", "washington dc monuments",
    "denver mountains", "phoenix desert", "hawaii volcanic beach"
]

BACKGROUNDS = [
    "solid neutral gray", "solid white", "solid black", "blurred bokeh",
    "gradient", "textured wall", "brick wall", "window with natural light",
    "outdoor scenery", "indoor setting", "abstract", "minimalist",
    "concrete wall", "graffiti wall", "ivy covered wall", "wooden panels",
    "marble texture", "velvet curtain", "silk fabric backdrop",
    "neon signs", "city lights bokeh", "nature bokeh leaves",
    "water reflections", "sunset sky", "stormy clouds", "starry night sky",
    "aurora borealis", "fog and mist", "rain drops on glass",
    "snow falling", "cherry blossoms", "autumn leaves",
    "tropical plants", "cactus desert", "ocean waves",
    "mountain peaks", "urban skyline", "industrial pipes",
    "library bookshelves", "art paintings gallery", "mirror reflections",
    "glass architecture", "geometric patterns", "vintage wallpaper",
    "modern art installation", "holographic", "prismatic rainbow",
    "smoke and haze", "fire sparks", "glitter sparkles"
]

CAMERAS = [
    "Canon EOS R5", "Sony A7R IV", "Nikon Z9", "Hasselblad X2D",
    "Leica M11", "Fujifilm GFX 100S", "Phase One IQ4", "Canon 5D Mark IV",
    "Sony A1", "Nikon D850", "Canon EOS R3", "Sony A7S III",
    "Nikon Z8", "Panasonic S1R", "Sigma fp L", "Leica SL2",
    "Fujifilm X-T5", "Olympus OM-1", "Pentax K-1 II", "Canon EOS R6 II",
    "Sony A7C II", "Nikon Zf", "Hasselblad 907X", "Phase One XT",
    "RED Komodo", "ARRI Alexa Mini", "Blackmagic Pocket 6K",
    "iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "Google Pixel 8 Pro"
]

LENSES = [
    "85mm f/1.4 portrait lens", "50mm f/1.2", "35mm f/1.4",
    "70-200mm f/2.8 telephoto", "24-70mm f/2.8", "105mm f/2.8 macro",
    "135mm f/2", "200mm f/2", "28mm f/1.4 wide angle",
    "40mm f/2 pancake", "56mm f/1.2", "90mm f/2.8 macro",
    "100mm f/2.8L macro", "180mm f/2.8", "300mm f/2.8 telephoto",
    "14mm f/1.8 ultra wide", "20mm f/1.4", "24mm f/1.4",
    "58mm f/0.95 Noct", "75mm f/1.25", "110mm f/2",
    "Petzval 85mm swirly bokeh", "Lensbaby velvet 56", "tilt-shift 45mm",
    "anamorphic 50mm", "vintage helios 44-2 58mm", "zeiss otus 55mm f/1.4",
    "sigma art 105mm f/1.4", "sony gm 135mm f/1.8"
]

STYLES = [
    "photorealistic", "editorial fashion", "commercial advertising",
    "lifestyle candid", "corporate professional", "artistic portrait",
    "documentary", "glamour", "natural authentic", "cinematic",
    "film noir", "vintage 1950s", "retro 1970s", "1980s neon aesthetic",
    "1990s grunge", "y2k aesthetic", "cottagecore", "dark academia",
    "cyberpunk", "steampunk", "vaporwave", "synthwave",
    "minimalist scandinavian", "maximalist baroque", "art deco",
    "pop art warhol style", "impressionist painterly", "surrealist dreamlike",
    "high fashion vogue", "street style hypebeast", "bohemian free spirit",
    "preppy classic", "punk rock edgy", "goth dark romantic",
    "ethereal angelic", "fierce warrior", "soft romantic",
    "bold graphic", "muted film stock", "cross processed",
    "infrared photography", "long exposure motion"
]

CLOTHING = [
    "business suit", "casual t-shirt and jeans", "formal dress",
    "athletic wear", "smart casual", "streetwear", "traditional cultural attire",
    "summer dress", "winter coat", "professional uniform", "elegant evening wear",
    "leather jacket and jeans", "hoodie and joggers", "blazer and chinos",
    "cocktail dress", "ball gown", "tuxedo", "wedding dress",
    "swimwear bikini", "swimwear trunks", "yoga outfit", "running gear",
    "hiking outfit", "ski wear", "beach coverup", "linen summer outfit",
    "knit sweater cozy", "denim jacket", "trench coat", "puffer jacket",
    "vintage dress", "bohemian maxi dress", "punk outfit with patches",
    "goth all black outfit", "preppy polo and khakis", "hipster flannel",
    "minimalist neutral tones", "bold colorful pattern", "monochrome outfit",
    "workwear overalls", "chef uniform", "doctor scrubs", "military uniform",
    "police uniform", "firefighter gear", "construction worker",
    "traditional kimono", "indian saree", "african dashiki",
    "scottish kilt", "mexican embroidered dress", "chinese qipao"
]

QUALITIES = {
    "draft": {"steps": 15, "cfg": 6},
    "quick": {"steps": 20, "cfg": 6.5},
    "normal": {"steps": 30, "cfg": 7},
    "good": {"steps": 40, "cfg": 7.5},
    "high": {"steps": 50, "cfg": 8},
    "very_high": {"steps": 60, "cfg": 8},
    "ultra": {"steps": 75, "cfg": 8.5},
    "extreme": {"steps": 100, "cfg": 9},
    "maximum": {"steps": 150, "cfg": 9}
}

RESOLUTIONS = {
    # Standard
    "sd": (512, 512),
    "hd": (768, 768),
    "full_hd": (1024, 1024),
    "2k": (1280, 1280),
    "4k": (2048, 2048),
    # iPhone resolutions
    "iphone_12mp": (3024, 4032),
    "iphone_48mp": (5712, 4284),
    "iphone_main": (4032, 3024),
    "iphone_wide": (4032, 3024),
    # Samsung resolutions
    "samsung_200mp": (16320, 12240),
    "samsung_108mp": (12000, 9000),
    "samsung_50mp": (8160, 6120),
    "samsung_12mp": (4032, 3024),
    # Google Pixel
    "pixel_50mp": (8160, 6144),
    "pixel_main": (4080, 3072),
    # Common phone ratios
    "phone_12mp": (4000, 3000),
    "phone_16mp": (4608, 3456),
    "phone_48mp": (8000, 6000),
    "phone_64mp": (9216, 6912),
    "phone_108mp": (12000, 9000),
    # Social media optimized
    "instagram_square": (1080, 1080),
    "instagram_portrait": (1080, 1350),
    "instagram_story": (1080, 1920),
    "tiktok": (1080, 1920),
    "twitter": (1200, 675),
    "facebook": (1200, 630),
    "linkedin": (1200, 627)
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

# Natural imperfections for realistic photos
IMPERFECTIONS = [
    # Framing issues
    "slightly off-center framing", "asymmetrical composition", "subject not perfectly centered",
    "headroom slightly too much", "headroom slightly too little", "tilted horizon",
    "awkward crop at edge of frame",
    
    # Focus/technical
    "slightly soft focus", "focus on wrong eye", "shallow depth of field with background person in focus",
    "slight motion blur", "minor camera shake", "slightly overexposed highlights",
    "slightly underexposed shadows", "lens flare in corner", "chromatic aberration on edges",
    
    # Subject behavior
    "caught mid-blink", "looking slightly past camera", "looking at something off-frame",
    "distracted expression", "forced awkward smile", "genuine candid expression",
    "mid-sentence mouth open", "eyes looking different directions", "squinting slightly",
    "one eye more closed than other", "asymmetrical smile", "raised eyebrow",
    
    # Body/pose issues  
    "awkward hand placement", "stiff unnatural pose", "slouching slightly",
    "leaning too far", "arms at awkward angle", "fingers in weird position",
    "caught mid-gesture", "weight on one foot unbalanced",
    
    # Environmental
    "photobomber in background", "distracting background element", "shadow across face",
    "harsh unflattering lighting", "red eye from flash", "shiny forehead from flash",
    "hair in face", "wind-blown messy hair", "clothing wrinkled",
    "collar popped wrong", "tag sticking out", "stain on clothing",
    
    # Amateur photographer vibes
    "snapshot aesthetic", "point and shoot camera quality", "phone camera selfie style",
    "bad flash photography", "fluorescent lighting color cast", "mixed lighting white balance off",
    "jpeg compression artifacts", "slightly grainy high iso noise"
]

IMPERFECTION_LEVELS = {
    "subtle": 1,      # 1 imperfection
    "natural": 2,     # 2 imperfections  
    "candid": 3,      # 3 imperfections
    "amateur": 4,     # 4 imperfections
    "chaotic": 6      # 6 imperfections
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
    additional: str = None,
    imperfect: str = None
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
    
    # Build base quality descriptor based on imperfection level
    if imperfect:
        num_imperfections = IMPERFECTION_LEVELS.get(imperfect, 2)
        selected_imperfections = random.sample(IMPERFECTIONS, min(num_imperfections, len(IMPERFECTIONS)))
        imperfection_text = ", ".join(selected_imperfections)
        quality_text = "candid amateur photograph, authentic real photo"
    else:
        imperfection_text = None
        quality_text = "8k uhd, highly detailed, sharp focus, professional photography"
    
    parts = [
        f"{'candid amateur' if imperfect else 'professional'} {style} photograph",
        f"{framing_desc}",
        f"of a {age} {ethnicity} {gender}",
        f"{pose}",
        clothing and f"wearing {clothing}",
        f"{lighting}",
        f"during {time_of_day}",
        f"at {location}",
        background and f"with {background} background",
        f"shot on {camera} with {lens}",
        quality_text,
        imperfection_text,
        additional
    ]
    
    prompt = ", ".join(p for p in parts if p)
    return prompt


def get_workflow(prompt: str, negative: str, width: int, height: int, 
                 steps: int, cfg: float, seed: int = None, upscale: bool = False) -> dict:
    """Generate ComfyUI workflow JSON.
    
    If upscale=True, uses RealESRGAN_x4plus for 4x AI upscaling and saves both versions.
    """
    
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
    
    # Add AI upscaling nodes if requested
    if upscale:
        # Load RealESRGAN upscale model
        workflow["10"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": "RealESRGAN_x4plus.safetensors"
            }
        }
        # Upscale the decoded image (4x)
        workflow["11"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["10", 0],
                "image": ["8", 0]
            }
        }
        # Save upscaled version with different prefix
        workflow["12"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "portrait_upscaled",
                "images": ["11", 0]
            }
        }
    
    return {"prompt": workflow}


def queue_prompt(workflow: dict) -> str:
    """Queue a prompt in ComfyUI and return prompt_id."""
    resp = requests.post(
        urljoin(COMFYUI_HOST, "/prompt"),
        json=workflow,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    if not resp.ok:
        print(f"   API Error: {resp.text}")
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
    imperfect: str = None,
    
    # Technical
    quality: str = "normal",
    resolution: str = "hd",
    aspect: str = "portrait",
    seed: int = None,
    
    # Output
    negative: str = None
) -> Path:
    """Generate a portrait photo."""
    
    # Resolve random values for metadata tracking
    actual_gender = gender or random.choice(GENDERS)
    actual_age = age or random.choice(AGES)
    actual_ethnicity = ethnicity or random.choice(ETHNICITIES)
    actual_pose = pose or random.choice(POSES)
    actual_lighting = lighting or random.choice(LIGHTING)
    actual_time = time_of_day or random.choice(TIME_OF_DAY)
    actual_location = location or random.choice(LOCATIONS)
    actual_camera = camera or random.choice(CAMERAS)
    actual_lens = lens or random.choice(LENSES)
    actual_clothing = clothing or random.choice(CLOTHING)
    actual_background = background or random.choice(BACKGROUNDS)
    actual_seed = seed or random.randint(1, 2**32-1)
    
    # Build prompt with actual values
    prompt = build_prompt(
        gender=actual_gender, age=actual_age, ethnicity=actual_ethnicity, clothing=actual_clothing,
        framing=framing, pose=actual_pose, lighting=actual_lighting, time_of_day=actual_time,
        location=actual_location, background=actual_background, camera=actual_camera, lens=actual_lens,
        style=style, additional=additional, imperfect=imperfect
    )
    
    # Store metadata for saving later
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "negative_prompt": negative,
        "person": {
            "gender": actual_gender,
            "age": actual_age,
            "ethnicity": actual_ethnicity,
            "clothing": actual_clothing
        },
        "shot": {
            "framing": framing,
            "pose": actual_pose
        },
        "environment": {
            "lighting": actual_lighting,
            "time_of_day": actual_time,
            "location": actual_location,
            "background": actual_background
        },
        "camera": {
            "camera": actual_camera,
            "lens": actual_lens
        },
        "style": {
            "style": style,
            "imperfect": imperfect,
            "additional": additional
        },
        "technical": {
            "quality": quality,
            "resolution": resolution,
            "aspect": aspect,
            "seed": actual_seed
        }
    }
    
    # Default negative prompt
    if negative is None:
        negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, missing fingers, extra fingers, mutated, disfigured, watermark, text, signature, cartoon, anime, illustration, painting, drawing, cgi, 3d render"
    
    # Calculate target dimensions
    base_res = RESOLUTIONS.get(resolution, RESOLUTIONS["hd"])
    ratio = ASPECT_RATIOS.get(aspect, ASPECT_RATIOS["portrait"])
    
    if ratio[0] > ratio[1]:  # Landscape
        target_width = base_res[0]
        target_height = int(base_res[0] * ratio[1] / ratio[0])
    else:  # Portrait or square
        target_height = base_res[1]
        target_width = int(base_res[1] * ratio[0] / ratio[1])
    
    # Cap generation at 1024x1024 max to avoid duplicates/clones
    MAX_GEN_SIZE = 1024
    needs_upscale = target_width > MAX_GEN_SIZE or target_height > MAX_GEN_SIZE
    
    if needs_upscale:
        # Scale down proportionally for generation
        scale_factor = MAX_GEN_SIZE / max(target_width, target_height)
        gen_width = int(target_width * scale_factor)
        gen_height = int(target_height * scale_factor)
    else:
        gen_width = target_width
        gen_height = target_height
    
    # Round to nearest 8 (required by most models)
    gen_width = (gen_width // 8) * 8
    gen_height = (gen_height // 8) * 8
    target_width = (target_width // 8) * 8
    target_height = (target_height // 8) * 8
    
    qual = QUALITIES.get(quality, QUALITIES["normal"])
    
    print(f"📸 Generating portrait...")
    print(f"   Prompt: {prompt[:100]}...")
    if needs_upscale:
        print(f"   Generate: {gen_width}x{gen_height} → Upscale to: {target_width}x{target_height}")
    else:
        print(f"   Size: {gen_width}x{gen_height}")
    print(f"   Steps: {qual['steps']}")
    
    # Generate workflow (with server-side AI upscaling if needed)
    workflow = get_workflow(
        prompt=prompt,
        negative=negative,
        width=gen_width,
        height=gen_height,
        steps=qual["steps"],
        cfg=qual["cfg"],
        seed=actual_seed,
        upscale=needs_upscale
    )
    
    # Add generation dimensions to metadata
    metadata["technical"]["gen_width"] = gen_width
    metadata["technical"]["gen_height"] = gen_height
    metadata["technical"]["target_width"] = target_width
    metadata["technical"]["target_height"] = target_height
    metadata["technical"]["upscaled"] = needs_upscale
    metadata["technical"]["steps"] = qual["steps"]
    metadata["technical"]["cfg"] = qual["cfg"]
    
    # Queue and wait
    prompt_id = queue_prompt(workflow)
    print(f"   Queued: {prompt_id}")
    
    result = wait_for_completion(prompt_id)
    
    # Download images
    outputs = result.get("outputs", {})
    original_path = None
    upscaled_path = None
    
    # Collect all generated images
    for node_id, output in outputs.items():
        if "images" in output:
            for img in output["images"]:
                filename = img["filename"]
                path = download_image(filename, img.get("subfolder", ""))
                
                # Identify by prefix
                if filename.startswith("portrait_upscaled"):
                    upscaled_path = path
                else:
                    original_path = path
    
    if original_path is None:
        raise RuntimeError("No image generated")
    
    # Save metadata
    if needs_upscale and upscaled_path:
        # Both versions - metadata with upscaled as primary
        json_path = upscaled_path.with_suffix(".json")
        metadata["image_file"] = upscaled_path.name
        metadata["original_file"] = original_path.name
        metadata["upscale_model"] = "RealESRGAN_x4plus"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Original: {original_path}")
        print(f"   ✅ Upscaled (RealESRGAN): {upscaled_path}")
        print(f"   📄 Metadata: {json_path}")
        return upscaled_path
    else:
        # Single version
        json_path = original_path.with_suffix(".json")
        metadata["image_file"] = original_path.name
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved: {original_path}")
        print(f"   📄 Metadata: {json_path}")
        return original_path


def main():
    parser = argparse.ArgumentParser(description="Generate portrait photos via ComfyUI")
    
    # Batch options
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--random", action="store_true", help="Randomize all options for each image")
    
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
    parser.add_argument("--imperfect", choices=list(IMPERFECTION_LEVELS.keys()),
                       help="Add natural imperfections: subtle, natural, candid, amateur, chaotic")
    
    # Technical
    parser.add_argument("--quality", choices=list(QUALITIES.keys()), default="normal")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS.keys()), default="hd")
    parser.add_argument("--aspect", choices=list(ASPECT_RATIOS.keys()), default="portrait")
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    
    print(f"🎯 Generating {args.count} portrait(s)...")
    if args.random:
        print("   Mode: RANDOM (all parameters randomized per image)")
    else:
        print("   Mode: FIXED (using provided parameters)")
    print()
    
    generated = []
    failed = 0
    
    for i in range(args.count):
        print(f"[{i+1}/{args.count}] ", end="")
        
        try:
            if args.random:
                # Randomize everything for each image
                rand_quality = random.choice(list(QUALITIES.keys()))
                rand_resolution = random.choice(list(RESOLUTIONS.keys()))
                rand_aspect = random.choice(list(ASPECT_RATIOS.keys()))
                # Randomly decide if this image has imperfections (50% chance if --imperfect not specified)
                rand_imperfect = args.imperfect or (random.choice(list(IMPERFECTION_LEVELS.keys())) if random.random() > 0.5 else None)
                path = generate(
                    gender=None,
                    age=None,
                    ethnicity=None,
                    clothing=random.choice(CLOTHING),
                    framing=random.choice(list(FRAMING.keys())),
                    pose=random.choice(POSES),
                    lighting=random.choice(LIGHTING),
                    time_of_day=random.choice(TIME_OF_DAY),
                    location=random.choice(LOCATIONS),
                    background=random.choice(BACKGROUNDS),
                    camera=random.choice(CAMERAS),
                    lens=random.choice(LENSES),
                    style=random.choice(STYLES),
                    additional=None,
                    imperfect=rand_imperfect,
                    quality=rand_quality,
                    resolution=rand_resolution,
                    aspect=rand_aspect,
                    seed=None  # Random seed each time
                )
            else:
                # Use provided parameters (or defaults) for all images
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
                    imperfect=args.imperfect,
                    quality=args.quality,
                    resolution=args.resolution,
                    aspect=args.aspect,
                    seed=None  # Still random seed for variety
                )
            generated.append(path)
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
            continue
    
    print(f"\n{'='*50}")
    print(f"✨ Complete! Generated: {len(generated)}, Failed: {failed}")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
