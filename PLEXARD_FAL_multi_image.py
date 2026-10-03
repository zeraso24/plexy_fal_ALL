"""
PLEXARD FAL - Multi-model image node (fal.ai)

One ComfyUI node that can call several fal.ai image models.
- Text-to-image when no image is connected
- Edit / reference / remix when images are connected
- Inpaint when a MASK is connected (models that support it)

Every model is described once in MODELS (endpoints, supported parameters,
allowed values). The node only sends the parameters a model understands,
and the same registry is exposed to the frontend (/plexard_fal/models) so
the JS extension can show/hide widgets when you change the model.

To add a model: add one entry to MODELS. Nothing else needs to change.
"""

import os
import io
import json
import base64
import tempfile
import configparser

import torch
import numpy as np
import requests
from PIL import Image
from fal_client.client import SyncClient

LOG = "[PLEXARD_FAL]"
KEY_PLACEHOLDERS = {"", "<your_fal_api_key_here>", "YOUR_FAL_KEY"}


# ==================== MODEL REGISTRY ====================

IMG_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"]
GPT_SIZES = ["auto"] + IMG_SIZES + ["custom"]
SEEDREAM_SIZES = ["auto_2K"] + IMG_SIZES + ["custom"]
IDEOGRAM_SIZES = IMG_SIZES + ["custom"]
SIZE_UNION = ["auto", "auto_2K"] + IMG_SIZES + ["custom"]

ASPECT_RATIOS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8"]
ASPECT_RATIOS_STD = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]
RESOLUTIONS = ["0.5K", "1K", "2K", "4K"]
QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
BACKGROUNDS = ["auto", "transparent", "opaque"]
OUTPUT_FORMATS = ["png", "jpeg", "webp"]
SAFETY_LEVELS = ["1", "2", "3", "4", "5", "6"]
THINKING_LEVELS = ["off", "minimal", "high"]
ENHANCE_MODES = ["standard", "fast"]
RENDERING_SPEEDS = ["TURBO", "BALANCED", "QUALITY"]
IDEOGRAM_STYLES = ["AUTO", "GENERAL", "REALISTIC", "DESIGN"]

# Parameter "keys" a model can support (see _build_arguments for what each one sends)
GPT_COMMON = {"size", "quality", "background", "output_format", "num_images", "sync_mode"}
GPT25_COMMON = GPT_COMMON | {"output_compression"}
NB2_COMMON = {"aspect_ratio", "resolution", "output_format", "num_images", "seed", "safety_tolerance",
              "sync_mode", "limit_generations", "enable_web_search", "thinking_level", "system_prompt"}
NBPRO_COMMON = NB2_COMMON - {"thinking_level"}
NB_COMMON = {"aspect_ratio", "output_format", "num_images", "safety_tolerance", "sync_mode", "limit_generations"}
SD_PRO = {"size", "num_images", "output_format", "sync_mode", "enable_safety_checker", "enhance_prompt_mode"}
SD_LITE = {"size", "num_images", "max_images", "sync_mode", "enable_safety_checker"}
SD_FLASH = {"size", "num_images", "sync_mode", "enable_safety_checker"}
IDEO_T2I = {"size", "rendering_speed", "style", "expand_prompt", "negative_prompt", "num_images", "seed", "sync_mode"}
IDEO_REMIX = IDEO_T2I | {"strength"}
IDEO_INPAINT = {"rendering_speed", "expand_prompt", "num_images", "seed", "sync_mode", "mask"}


def _spec(endpoints, params, options=None, max_refs=10, max_num=4, pixel_limits=None, max_ratio=None, single_image=False):
    return {
        "endpoints": endpoints,        # mode -> fal endpoint id (t2i / edit / inpaint)
        "params": params,              # mode -> set of supported parameter keys
        "options": options or {},      # widget -> allowed values for this model
        "max_refs": max_refs,          # max reference images
        "max_num": max_num,            # max num_images
        "pixel_limits": pixel_limits,  # (min_px, max_px) for custom sizes (warning only)
        "max_ratio": max_ratio,
        "single_image": single_image,  # first image = image_url, the rest = style references
    }


def _both(common, edit_extra=()):
    return {"t2i": set(common), "edit": set(common) | set(edit_extra)}


MODELS = {
    "GPT Image 2.5 Flare": _spec(
        {"t2i": "openai/gpt-image-2.5/flare/text-to-image", "edit": "openai/gpt-image-2.5/flare/edit"},
        _both(GPT25_COMMON, {"mask"}),
        {"size_preset": GPT_SIZES, "quality": QUALITIES, "background": BACKGROUNDS, "output_format": OUTPUT_FORMATS},
        max_refs=16, max_num=10, pixel_limits=(655_360, 8_294_400), max_ratio=3,
    ),
    "GPT Image 2.5 Sunburst": _spec(
        {"t2i": "openai/gpt-image-2.5/sunburst/text-to-image", "edit": "openai/gpt-image-2.5/sunburst/edit"},
        _both(GPT25_COMMON, {"mask"}),
        {"size_preset": GPT_SIZES, "quality": QUALITIES, "background": BACKGROUNDS, "output_format": OUTPUT_FORMATS},
        max_refs=16, max_num=10, pixel_limits=(655_360, 8_294_400), max_ratio=3,
    ),
    "GPT Image 2": _spec(
        {"t2i": "openai/gpt-image-2", "edit": "openai/gpt-image-2/edit"},
        _both(GPT_COMMON, {"mask"}),
        {"size_preset": GPT_SIZES, "quality": ["auto", "low", "medium", "high"], "background": BACKGROUNDS,
         "output_format": OUTPUT_FORMATS},
        max_refs=16, max_num=4, pixel_limits=(655_360, 8_294_400), max_ratio=3,
    ),
    "Nano Banana 2": _spec(
        {"t2i": "fal-ai/nano-banana-2", "edit": "fal-ai/nano-banana-2/edit"},
        _both(NB2_COMMON),
        {"aspect_ratio": ASPECT_RATIOS, "resolution": RESOLUTIONS, "output_format": OUTPUT_FORMATS,
         "safety_tolerance": SAFETY_LEVELS, "thinking_level": THINKING_LEVELS},
        max_refs=14, max_num=4,
    ),
    "Nano Banana Pro": _spec(
        {"t2i": "fal-ai/nano-banana-pro", "edit": "fal-ai/nano-banana-pro/edit"},
        _both(NBPRO_COMMON),
        {"aspect_ratio": ASPECT_RATIOS_STD, "resolution": ["1K", "2K", "4K"], "output_format": OUTPUT_FORMATS,
         "safety_tolerance": SAFETY_LEVELS},
        max_refs=14, max_num=4,
    ),
    "Nano Banana": _spec(
        {"t2i": "fal-ai/nano-banana", "edit": "fal-ai/nano-banana/edit"},
        _both(NB_COMMON),
        {"aspect_ratio": ASPECT_RATIOS_STD, "output_format": OUTPUT_FORMATS, "safety_tolerance": SAFETY_LEVELS},
        max_refs=10, max_num=4,
    ),
    "Seedream 5.0 Pro": _spec(
        {"t2i": "bytedance/seedream/v5/pro/text-to-image", "edit": "bytedance/seedream/v5/pro/edit"},
        _both(SD_PRO),
        {"size_preset": SEEDREAM_SIZES, "output_format": ["jpeg", "png"], "enhance_prompt_mode": ENHANCE_MODES},
        max_refs=10, max_num=6, pixel_limits=(1_048_576, 4_194_304), max_ratio=16,
    ),
    "Seedream 5.0 Lite": _spec(
        {"t2i": "fal-ai/bytedance/seedream/v5/lite/text-to-image", "edit": "fal-ai/bytedance/seedream/v5/lite/edit"},
        _both(SD_LITE),
        {"size_preset": SEEDREAM_SIZES},
        max_refs=10, max_num=6, pixel_limits=(3_686_400, 16_777_216), max_ratio=16,
    ),
    "Seedream 5.0 Flash": _spec(
        {"t2i": "bytedance/seedream/v5/flash/text-to-image", "edit": "bytedance/seedream/v5/flash/edit"},
        _both(SD_FLASH),
        {"size_preset": SEEDREAM_SIZES},
        max_refs=10, max_num=6,
    ),
    "Ideogram V3": _spec(
        {"t2i": "fal-ai/ideogram/v3", "edit": "fal-ai/ideogram/v3/remix", "inpaint": "fal-ai/ideogram/v3/edit"},
        {"t2i": set(IDEO_T2I), "edit": set(IDEO_REMIX), "inpaint": set(IDEO_INPAINT)},
        {"size_preset": IDEOGRAM_SIZES, "rendering_speed": RENDERING_SPEEDS, "style": IDEOGRAM_STYLES},
        max_refs=10, max_num=8, single_image=True,
    ),
}

DEFAULT_MODEL = "GPT Image 2.5 Sunburst"
WIDGET_DEFAULTS = {"quality": "high", "size_preset": "auto", "aspect_ratio": "auto", "resolution": "1K", "output_format": "png"}


# ==================== UI SCHEMA (for the JS extension) ====================

ALWAYS_VISIBLE = ["model", "prompt", "api_key", "extra_args_json"]
KEY_TO_WIDGETS = {
    "size": ["size_preset", "custom_width", "custom_height"],
    "aspect_ratio": ["aspect_ratio"],
    "mask": ["invert_mask"],
}


def _widgets_for(keys):
    out = []
    for k in sorted(keys):
        out.extend(KEY_TO_WIDGETS.get(k, [k]))
    return out


def _build_ui_schema():
    managed = set()
    models = {}
    for name, spec in MODELS.items():
        widgets = set()
        for keys in spec["params"].values():
            widgets.update(_widgets_for(keys))
        managed.update(widgets)
        models[name] = {
            "widgets": sorted(widgets),
            "options": spec["options"],
            "ranges": {"num_images": [1, spec["max_num"]], "max_images": [1, 6]},
        }
    return {
        "always": ALWAYS_VISIBLE,
        "managed": sorted(managed),
        "defaults": WIDGET_DEFAULTS,
        "models": models,
    }


UI_SCHEMA = _build_ui_schema()

try:  # only available inside ComfyUI
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/plexard_fal/models")
    async def _plexard_fal_models(request):
        return web.json_response(UI_SCHEMA)
except Exception as _e:  # noqa: BLE001
    print(f"{LOG} Could not register UI schema route (widgets will not auto-hide): {_e}")


# ==================== API KEY ====================

def resolve_api_key(node_key=""):
    """Priority: node field > FAL_KEY env var > config.ini (next to this file or in the parent folder)."""
    if node_key and node_key.strip() not in KEY_PLACEHOLDERS:
        return node_key.strip()

    env_key = os.environ.get("FAL_KEY", "").strip()
    if env_key not in KEY_PLACEHOLDERS:
        return env_key

    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        path = os.path.join(folder, "config.ini")
        if os.path.isfile(path):
            cfg = configparser.ConfigParser()
            cfg.read(path)
            cfg_key = cfg.get("API", "FAL_KEY", fallback="").strip()
            if cfg_key not in KEY_PLACEHOLDERS:
                return cfg_key

    raise ValueError(
        "FAL API key not found. Paste it in the 'api_key' field, set the FAL_KEY environment variable, "
        "or put it in config.ini. Get one at https://fal.ai/dashboard/keys"
    )


# ==================== IMAGE UTILS ====================

def tensor_to_pil_list(image):
    """ComfyUI IMAGE tensor [B,H,W,C] -> list of PIL images."""
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
    if arr.ndim == 3:
        arr = arr[None]
    frames = []
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        frames.append(Image.fromarray(frame))
    return frames


def mask_to_pil(mask, size, invert=False):
    """ComfyUI MASK -> grayscale PIL image (white = area to edit), resized to `size`."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
    while arr.ndim > 2:
        arr = arr[0]
    if invert:
        arr = 1.0 - arr
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def upload_pil(client, pil_image):
    """Upload a PIL image to fal storage and return its URL."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp, format="PNG")
            tmp_path = tmp.name
        return client.upload_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_image(url):
    """Download an image (http URL or data URI) as a PIL image."""
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def pil_to_comfy(pil_img):
    """Returns (rgb float32 0-1, mask float32 0-1). mask = 1 - alpha (LoadImage convention)."""
    has_alpha = pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info)
    if has_alpha:
        rgba = np.array(pil_img.convert("RGBA")).astype(np.float32) / 255.0
        return rgba[..., :3], 1.0 - rgba[..., 3]
    rgb = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return rgb, np.zeros(rgb.shape[:2], dtype=np.float32)


# ==================== ARGUMENT BUILDER ====================

def resolve_mode(spec, has_images, has_mask):
    if not has_images:
        return "t2i"
    if has_mask and "inpaint" in spec["endpoints"]:
        return "inpaint"
    return "edit"


def _build_arguments(model_name, spec, mode, prompt, p):
    """Builds the request body using ONLY the parameters this model/mode supports."""
    allowed = spec["params"][mode]
    opts = spec["options"]
    a = {"prompt": prompt}

    def valid(widget, value):
        vals = opts.get(widget)
        if vals is None or value in vals:
            return True
        print(f"{LOG} '{value}' is not valid for {model_name} ({widget}); using the API default instead.")
        return False

    # ---- image size ----
    if "size" in allowed:
        preset = p["size_preset"]
        if preset == "custom":
            w = max(256, min(4096, (int(p["custom_width"]) // 16) * 16))
            h = max(256, min(4096, (int(p["custom_height"]) // 16) * 16))
            limits = spec["pixel_limits"]
            ratio_limit = spec["max_ratio"]
            if limits and not (limits[0] <= w * h <= limits[1]):
                print(f"{LOG} Warning: {w}x{h} is outside the documented pixel range "
                      f"({limits[0]:,}-{limits[1]:,}) for {model_name}. The API may reject or rescale it.")
            if ratio_limit and max(w / h, h / w) > ratio_limit:
                print(f"{LOG} Warning: aspect ratio above {ratio_limit}:1 for {model_name}.")
            a["image_size"] = {"width": w, "height": h}
        elif preset == "auto" and (mode == "t2i" or "auto" not in opts.get("size_preset", ["auto"])):
            pass  # let the API pick its default
        elif valid("size_preset", preset):
            a["image_size"] = preset

    # ---- simple enums ----
    for key in ("aspect_ratio", "quality", "background", "output_format", "resolution",
                "safety_tolerance", "enhance_prompt_mode", "rendering_speed", "style"):
        if key in allowed and valid(key, p[key]):
            a[key] = p[key]
    if "thinking_level" in allowed and p["thinking_level"] != "off" and valid("thinking_level", p["thinking_level"]):
        a["thinking_level"] = p["thinking_level"]

    # ---- numbers ----
    if "num_images" in allowed:
        n = max(1, min(int(p["num_images"]), spec["max_num"]))
        if n != int(p["num_images"]):
            print(f"{LOG} {model_name} supports at most {spec['max_num']} images per request; using {n}.")
        a["num_images"] = n
    if "max_images" in allowed:
        a["max_images"] = max(1, min(6, int(p["max_images"])))
    if "seed" in allowed and int(p["seed"]) >= 0:
        a["seed"] = int(p["seed"])
    if "strength" in allowed:
        a["strength"] = float(p["strength"])
    if "output_compression" in allowed and int(p["output_compression"]) >= 0 \
            and a.get("output_format") in ("jpeg", "webp"):
        a["output_compression"] = int(p["output_compression"])

    # ---- booleans / strings ----
    if "sync_mode" in allowed and p["sync_mode"]:
        a["sync_mode"] = True
    if "limit_generations" in allowed:
        a["limit_generations"] = bool(p["limit_generations"])
    if "enable_web_search" in allowed and p["enable_web_search"]:
        a["enable_web_search"] = True
    if "enable_safety_checker" in allowed:
        a["enable_safety_checker"] = bool(p["enable_safety_checker"])
    if "expand_prompt" in allowed:
        a["expand_prompt"] = bool(p["expand_prompt"])
    if "system_prompt" in allowed and p["system_prompt"].strip():
        a["system_prompt"] = p["system_prompt"]
    if "negative_prompt" in allowed and p["negative_prompt"].strip():
        a["negative_prompt"] = p["negative_prompt"]

    return a


NODE_DEFAULTS = {
    "size_preset": "auto", "custom_width": 1280, "custom_height": 720, "aspect_ratio": "auto",
    "resolution": "1K", "quality": "high", "background": "auto", "output_format": "png",
    "output_compression": -1, "num_images": 1, "max_images": 1, "seed": -1, "safety_tolerance": "4",
    "enable_web_search": False, "thinking_level": "off", "limit_generations": True, "system_prompt": "",
    "enable_safety_checker": True, "enhance_prompt_mode": "standard", "rendering_speed": "BALANCED",
    "style": "AUTO", "expand_prompt": True, "negative_prompt": "", "strength": 0.8, "sync_mode": False,
    "invert_mask": False, "extra_args_json": "",
}


# ==================== NODE ====================

class PlexardFalMultiImage:
    """Multi-model fal.ai image node: text-to-image, edit/reference, remix and inpaint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODELS.keys()), {"default": DEFAULT_MODEL,
                                                "tooltip": "Which fal.ai model to call. Unsupported options are hidden/ignored."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "FAL API key. Empty = FAL_KEY env var or config.ini. "
                               "WARNING: whatever you type here is saved inside the workflow .json.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional. No image = text-to-image. With image = edit / remix / reference."}),
                "image_2": ("IMAGE", {"tooltip": "Extra reference image"}),
                "image_3": ("IMAGE", {"tooltip": "Extra reference image"}),
                "image_4": ("IMAGE", {"tooltip": "Extra reference image"}),
                "mask": ("MASK", {"tooltip": "Optional mask (white = area to edit). Used by GPT Image models and Ideogram V3 inpaint."}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "size_preset": (SIZE_UNION, {"default": "auto", "tooltip": "'auto' = infer from input image / API default"}),
                "custom_width": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 720, "min": 256, "max": 4096, "step": 16}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "quality": (QUALITIES, {"default": "high", "tooltip": "More quality = more detail, latency and cost"}),
                "background": (BACKGROUNDS, {"default": "auto"}),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
                "output_compression": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "0-100, jpeg/webp only. -1 = don't send"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6, "tooltip": "Seedream Lite: images per generation"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF, "tooltip": "-1 = random / don't send"}),
                "safety_tolerance": (SAFETY_LEVELS, {"default": "4", "tooltip": "1 = strictest, 6 = most permissive"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "thinking_level": (THINKING_LEVELS, {"default": "off"}),
                "limit_generations": ("BOOLEAN", {"default": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enhance_prompt_mode": (ENHANCE_MODES, {"default": "standard"}),
                "rendering_speed": (RENDERING_SPEEDS, {"default": "BALANCED"}),
                "style": (IDEOGRAM_STYLES, {"default": "AUTO"}),
                "expand_prompt": ("BOOLEAN", {"default": True, "tooltip": "Ideogram MagicPrompt"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 1.0, "step": 0.01,
                                       "tooltip": "Ideogram remix: how strongly the input image is kept"}),
                "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "Returns data URIs; not stored in fal history"}),
                "extra_args_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Advanced: JSON object merged into the request body (overrides everything else). "
                               "e.g. {\"style_preset\": \"WATERCOLOR\"}",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "alpha_mask", "info")
    FUNCTION = "generate"
    CATEGORY = "plexard_FAL"
    DESCRIPTION = "Multi-model fal.ai image generation/editing (GPT Image, Nano Banana, Seedream, Ideogram)"

    def generate(self, model, prompt, api_key="", image=None, image_2=None, image_3=None, image_4=None,
                 mask=None, **kwargs):
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        if not prompt or not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        spec = MODELS[model]
        p = {**NODE_DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}

        client = SyncClient(key=resolve_api_key(api_key))

        # ---- collect images ----
        pil_images = []
        for tensor in (image, image_2, image_3, image_4):
            if tensor is not None:
                pil_images.extend(tensor_to_pil_list(tensor))

        mode = resolve_mode(spec, bool(pil_images), mask is not None)
        endpoint = spec["endpoints"][mode]
        allowed = spec["params"][mode]

        if mask is not None and not pil_images:
            print(f"{LOG} A mask was connected but there is no image; ignoring the mask.")
        if mask is not None and pil_images and "mask" not in allowed:
            print(f"{LOG} {model} does not support masks; ignoring the mask.")

        if len(pil_images) > spec["max_refs"]:
            print(f"{LOG} {model} accepts at most {spec['max_refs']} images; ignoring {len(pil_images) - spec['max_refs']}.")
            pil_images = pil_images[: spec["max_refs"]]

        # ---- arguments ----
        arguments = _build_arguments(model, spec, mode, prompt, p)

        if pil_images:
            urls = [upload_pil(client, im) for im in pil_images]
            if not urls or not all(urls):
                raise RuntimeError("Could not upload the images to fal.ai.")
            if spec["single_image"]:
                arguments["image_url"] = urls[0]
                if len(urls) > 1:
                    arguments["image_urls"] = urls[1:]  # style references
            else:
                arguments["image_urls"] = urls

            if mask is not None and "mask" in allowed:
                arguments["mask_url"] = upload_pil(
                    client, mask_to_pil(mask, pil_images[0].size, invert=bool(p["invert_mask"]))
                )

        # ---- advanced override ----
        extra = (p.get("extra_args_json") or "").strip()
        if extra:
            try:
                extra_dict = json.loads(extra)
            except json.JSONDecodeError as e:
                raise ValueError(f"extra_args_json is not valid JSON: {e}") from e
            if not isinstance(extra_dict, dict):
                raise ValueError("extra_args_json must be a JSON object, e.g. {\"seed\": 42}")
            arguments.update(extra_dict)

        print(f"{LOG} {model} -> {endpoint} ({mode}, {len(pil_images)} input image(s))")

        def on_update(update):
            logs = getattr(update, "logs", None)
            if logs:
                for log in logs:
                    msg = log.get("message") if isinstance(log, dict) else str(log)
                    if msg:
                        print(f"{LOG} {msg}")

        try:
            result = client.subscribe(endpoint, arguments=arguments, with_logs=True, on_queue_update=on_update)
        except Exception as e:
            raise RuntimeError(f"fal.ai error ({endpoint}): {e}") from e

        # ---- results ----
        images_info = result.get("images") or []
        if not images_info:
            raise RuntimeError(f"The API returned no images. Response: {result}")

        rgb_list, mask_list = [], []
        for info in images_info:
            rgb, alpha_mask = pil_to_comfy(download_image(info["url"]))
            rgb_list.append(rgb)
            mask_list.append(alpha_mask)

        h0, w0 = rgb_list[0].shape[:2]
        for i, (rgb, am) in enumerate(zip(rgb_list, mask_list)):
            if rgb.shape[:2] != (h0, w0):
                rgb_list[i] = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize((w0, h0), Image.LANCZOS)).astype(np.float32) / 255.0
                mask_list[i] = np.array(Image.fromarray((am * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR)).astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(np.stack(rgb_list, axis=0))
        mask_tensor = torch.from_numpy(np.stack(mask_list, axis=0))

        sent = {k: v for k, v in arguments.items() if k not in ("image_url", "image_urls", "mask_url")}
        info_text = json.dumps({
            "model": model,
            "endpoint": endpoint,
            "mode": mode,
            "sent_arguments": sent,
            "response_extras": {k: v for k, v in result.items() if k != "images"},
        }, ensure_ascii=False, indent=2, default=str)

        return (image_tensor, mask_tensor, info_text)


NODE_CLASS_MAPPINGS = {
    "PlexardFalMultiImage": PlexardFalMultiImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlexardFalMultiImage": "FAL ALL V1",
}
