from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings


class MultimodalUnavailableError(Exception):
    pass


_clip_model: Any | None = None
_clip_preprocess: Any | None = None
_clip_tokenizer: Any | None = None


def _load_clip_runtime() -> tuple[Any, Any, Any]:
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None and _clip_preprocess is not None and _clip_tokenizer is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer

    try:
        import open_clip  # type: ignore
        import torch  # type: ignore
    except Exception as exc:
        raise MultimodalUnavailableError(
            "CLIP runtime not installed. Install optional deps: open-clip-torch, torch, pillow."
        ) from exc

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )
        model.eval()
        model.to("cpu")
        tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
    except Exception as exc:
        raise MultimodalUnavailableError(f"Unable to initialize CLIP model: {exc}") from exc

    _clip_model = model
    _clip_preprocess = preprocess
    _clip_tokenizer = tokenizer
    return model, preprocess, tokenizer


def embed_images_clip(image_paths: list[str]) -> list[list[float]]:
    if not image_paths:
        return []

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise MultimodalUnavailableError(
            "CLIP runtime not installed. Install optional deps: open-clip-torch, torch, pillow."
        ) from exc

    model, preprocess, _ = _load_clip_runtime()

    vectors: list[list[float]] = []
    with torch.no_grad():
        for path in image_paths:
            image_path = Path(path)
            if not image_path.exists():
                raise MultimodalUnavailableError(f"Image path does not exist: {image_path}")
            image = Image.open(str(image_path)).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append([float(value) for value in features[0].tolist()])
    return vectors


def embed_text_clip(text: str) -> list[float]:
    if not text.strip():
        raise MultimodalUnavailableError("Empty query for CLIP text embedding.")
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise MultimodalUnavailableError(
            "CLIP runtime not installed. Install optional deps: open-clip-torch, torch, pillow."
        ) from exc

    model, _, tokenizer = _load_clip_runtime()
    with torch.no_grad():
        tokens = tokenizer([text])
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features[0].tolist()]


async def generate_vlm_caption(image_path: str, *, source_file: str, page: int) -> str | None:
    if not settings.enable_vlm_captions:
        return None
    if not settings.openai_api_key:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"

    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{image_b64}"

    endpoint = settings.openai_base_url.rstrip("/") + "/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.openai_api_key}",
    }
    payload = {
        "model": settings.vlm_caption_model or settings.openai_model,
        "max_output_tokens": int(settings.vlm_caption_max_tokens),
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Write a concise factual caption for this document image. "
                            "Include visible chart/topic keywords only. 1 sentence."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, content=json.dumps(payload))
            response.raise_for_status()
            response_payload = response.json()
    except Exception:
        return None

    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text", "")).strip()
                if text:
                    parts.append(text)
    combined = " ".join(parts).strip()
    if combined:
        return combined

    return f"Extracted image from {source_file}, page {page}."
