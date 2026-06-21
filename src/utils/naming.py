from __future__ import annotations
import re


def model_name_to_slug(model_name: str) -> str:
    """Return a filesystem-friendly identifier for a Hugging Face model name."""
    model_id = model_name.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^0-9A-Za-z]+", "_", model_id).strip("_").lower()
    if not slug:
        raise ValueError(f"Cannot derive a slug from model name: {model_name!r}")
    return slug
