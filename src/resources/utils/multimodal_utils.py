"""
This module contains all our multimodal utility functions
"""

import base64
import requests


def url_to_base64_data_url(image_url: str, mime_type: str):
    response = requests.get(image_url)
    response.raise_for_status()
    # Try to guess mime type from URL if not provided
    if not mime_type:
        mime_type = response.headers.get("Content-Type", "").split(";")[0]
        if not mime_type.startswith("image/"):
            raise ValueError(f"Invalid MIME type from server: {mime_type}")
    b64 = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def create_openai_style_img_message(
    text: str, image_data: str, image_type: str = "url", mime_type: str = ""
) -> dict:
    """
    Create an OpenAI-style multimodal message with text and image.

    Args:
        text (str): The text prompt.
        image_data (str): The image data. If image_type is "url", this should be a public URL.
                          If image_type is "base64", this should be a base64-encoded string.
        image_type (str): "url" for public URL (OpenAI/Anthropic), "base64" for local models.
        mime_type (str): The image MIME type (used only for base64).

    Returns:
        dict: OpenAI-style message.
    """
    if image_type == "url":
        data_uri = url_to_base64_data_url(image_data, mime_type)
        image_block = {"type": "image_url", "image_url": {"url": data_uri}}
    elif image_type == "base64":
        data_uri = f"data:{mime_type};base64,{image_data}"
        image_block = {"type": "image_url", "image_url": {"url": data_uri}}
    else:
        raise ValueError("image_type must be 'url' or 'base64'")

    return {"role": "user", "content": [{"type": "text", "text": text}, image_block]}
