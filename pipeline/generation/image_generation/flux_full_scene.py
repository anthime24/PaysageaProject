import os
import requests
from pathlib import Path

BFL_API_KEY = os.getenv("BFL_API_KEY")


def generate_full_garden(
    image_path: str,
    prompt: str,
    output_path: str
):
    url = "https://api.bfl.ai/v1/flux-pro"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    files = {
        "image": image_bytes
    }

    data = {
        "prompt": prompt,
        "strength": 0.85
    }

    headers = {
        "Authorization": f"Bearer {BFL_API_KEY}"
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(response.content)
