"""
Module image_generation - Génération de jardin via BFL FLUX.1 Fill PRO.

- config : BFL_API_KEY
- prompt_builder : build_prompt(plant, global_style)
- mask_manager : MaskManager, create_mask
- bfl_provider : inpaint(image_path, mask_path, prompt, out_path)
- mock_provider : inpaint_mock (mode sans API)
- scene_generator : generate_scene(image_path, rag_json_path)
- editor : remove_plant, replace_plant, add_plant
- demo : python -m garden_ai.image_generation.demo
"""
from .config import get_api_key
from .prompt_builder import build_prompt
from .mask_manager import MaskManager, MaskResult
from .scene_generator import generate_scene
from .editor import remove_plant, replace_plant, add_plant
from .mock_provider import inpaint_mock, create_preview_boxes
from .utils_rag import load_rag, load_rag_output

__all__ = [
    "get_api_key",
    "build_prompt",
    "MaskManager",
    "MaskResult",
    "generate_scene",
    "load_rag",
    "load_rag_output",
    "remove_plant",
    "replace_plant",
    "add_plant",
    "inpaint_mock",
    "create_preview_boxes",
]
