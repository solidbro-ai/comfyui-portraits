# ComfyUI Portrait Generator

Generate realistic photos of people via ComfyUI API.

## Setup

1. Set your ComfyUI host:
   ```bash
   export COMFYUI_HOST="http://your-server:8188"
   ```

2. Install RealVisXL_V4.0 model in ComfyUI

3. Run:
   ```bash
   python generate.py --random
   ```

## Options

### Person
- `--gender` male/female
- `--age` young adult, 20s, 30s, middle-aged, etc.
- `--ethnicity` caucasian, african, asian, hispanic, etc.
- `--clothing` business suit, casual, formal dress, etc.

### Shot
- `--framing` profile, headshot, shoulders, torso, three_quarter, full_body
- `--pose` looking at camera, arms crossed, sitting, etc.

### Environment
- `--lighting` natural, studio, golden hour, dramatic, etc.
- `--time` sunrise, morning, golden hour, night, etc.
- `--location` studio, office, street, park, etc.
- `--background` solid gray, bokeh, brick wall, etc.

### Camera
- `--camera` Canon EOS R5, Sony A7R IV, etc.
- `--lens` 85mm f/1.4, 50mm f/1.2, etc.

### Technical
- `--quality` draft, normal, high, ultra
- `--resolution` sd, hd, full_hd, 2k, 4k
- `--aspect` square, portrait, landscape, instagram, story
- `--seed` specific seed for reproducibility

## Examples

```bash
# Random portrait
python generate.py --random

# Specific person
python generate.py --gender female --age 30s --framing headshot --lighting "golden hour" --quality high

# Professional headshot
python generate.py --framing headshot --lighting "studio lighting" --background "solid neutral gray" --style "corporate professional"

# Full body fashion
python generate.py --framing full_body --style "editorial fashion" --location "urban city street" --quality ultra
```

## API Usage

```python
from generate import generate

path = generate(
    gender="female",
    age="30s",
    framing="headshot",
    lighting="golden hour sunlight",
    quality="high"
)
print(f"Generated: {path}")
```
