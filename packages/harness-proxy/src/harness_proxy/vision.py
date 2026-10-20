from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import threading
from collections.abc import Sequence
from typing import Any

from harness_proxy.chat import Refused, text_part

#: Most pages of a document passed to the model, one image each. A longer document is
#: cut, and a text part says how many pages are shown.
MAX_PAGES = 16
#: Longest side of an image, in pixels. Larger images are scaled down, because a vision
#: renderer charges by area.
MAX_SIDE = 1536
#: Scale at which PDF pages are rasterised. 2 is about 144 dpi, enough for small type. A page
#: whose longest side would pass `MAX_SIDE` is rasterised smaller instead.
PAGE_SCALE = 2.0
#: Most pixels an image may decode to, checked from its header before it is decoded. Larger
#: images are refused, because decoding is what costs the memory.
MAX_PIXELS = 50_000_000

PDF = "application/pdf"
NOT_FETCHED = "an image given by URL is not fetched by this proxy; send its bytes as base64"
#: What an image in the system prompt becomes: a system prompt is rendered as text.
IN_SYSTEM = "[an image was here; images in the system prompt are not shown to the model]"
_DATA_URI = re.compile(r"^data:([^;,]+)(;base64)?,(.*)$", re.DOTALL)
#: pdfium is not thread-safe, and requests are read on worker threads.
_PDFIUM = threading.Lock()


def image_part(image: Any) -> dict[str, Any]:
    return {"type": "image", "image": image}


def anthropic_media(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parts of an Anthropic `image` or `document` block."""
    if block.get("type") == "image":
        return [image_part(_anthropic_image(block))]
    return _anthropic_document(block)


def openai_media(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the parts of an OpenAI `image_url` or `file` part."""
    if block.get("type") == "image_url":
        return [image_part(_openai_image(block))]
    return _openai_file(block)


def as_text(parts: Sequence[dict[str, Any]]) -> str:
    """Return parts as one text, each image named rather than dropped in silence."""
    return "".join(
        str(p.get("text") or "") if p.get("type") == "text" else IN_SYSTEM for p in parts
    )


def _anthropic_image(block: dict[str, Any]) -> Any:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return decode_image(source.get("data"), str(source.get("media_type") or ""))
    if source.get("type") == "url":
        raise Refused(NOT_FETCHED)
    raise Refused(f"unsupported image source {source.get('type')!r}")


def _anthropic_document(block: dict[str, Any]) -> list[dict[str, Any]]:
    source = block.get("source") or {}
    media = str(source.get("media_type") or "")
    if source.get("type") == "text":
        return [text_part(source.get("data") or "")]
    if source.get("type") == "base64" and media == PDF:
        return paged(decode_bytes(source.get("data")))
    if source.get("type") == "base64" and media.startswith("image/"):
        return [image_part(decode_image(source.get("data"), media))]
    if source.get("type") == "url":
        raise Refused(NOT_FETCHED)
    raise Refused(f"unsupported document {media or source.get('type')!r}: PDF, image or text")


def _openai_image(block: dict[str, Any]) -> Any:
    url = block.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str):
        raise Refused("'image_url' must carry a url")
    match = _DATA_URI.match(url)
    if match is None:
        raise Refused(NOT_FETCHED)
    media, is_b64, payload = match.groups()
    if not is_b64:
        raise Refused("an image data URI must be base64")
    return decode_image(payload, media)


def _openai_file(block: dict[str, Any]) -> list[dict[str, Any]]:
    spec = block.get("file") or {}
    data = spec.get("file_data")
    if not isinstance(data, str):
        raise Refused("a 'file' part must carry file_data; file ids are not resolved here")
    match = _DATA_URI.match(data)
    media, payload = (match.group(1), match.group(3)) if match else (PDF, data)
    if media == PDF:
        return paged(decode_bytes(payload))
    if media.startswith("image/"):
        return [image_part(decode_image(payload, media))]
    raise Refused(f"unsupported file {media!r}: PDF or image")


def paged(pdf: bytes) -> list[dict[str, Any]]:
    """Return one image part per page, at most `MAX_PAGES`, and a note when the document was cut."""
    images, total = pages(pdf, MAX_PAGES)
    parts = [image_part(image) for image in images]
    if total > len(images):
        parts.append(text_part(f"\n[document cut: {len(images)} of {total} pages shown]\n"))
    return parts


# ------------------------------------------------------------------------ decoding


def decode_bytes(data: Any) -> bytes:
    if not isinstance(data, str) or not data:
        raise Refused("base64 data missing")
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as bad:
        raise Refused(f"base64 data unreadable: {bad}") from bad


def decode_image(data: Any, media: str) -> Any:
    """Decode an image, refusing one over `MAX_PIXELS` before its pixels are read."""
    from PIL import Image, UnidentifiedImageError

    raw = decode_bytes(data)
    try:
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise Refused(
                f"image ({media or 'unknown type'}) is {width}x{height}, over the "
                f"{MAX_PIXELS:,} pixels this proxy decodes; send it smaller"
            )
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as bad:
        raise Refused(f"image data ({media or 'unknown type'}) unreadable: {bad}") from bad
    return fit(image)


def fit(image: Any) -> Any:
    """Convert the image to RGB and shrink it so its longest side is at most `MAX_SIDE`."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        image = image.copy()
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def pages(pdf: bytes, most: int) -> tuple[list[Any], int]:
    """Render a PDF's first `most` pages as images, scaled to fit, with its page count."""
    import pypdfium2

    rendered: list[Any] = []
    with _PDFIUM:
        try:
            document = pypdfium2.PdfDocument(pdf)
        except Exception as bad:  # noqa: BLE001 - pdfium's own error type is not an API
            raise Refused(f"PDF unreadable: {bad}") from bad
        try:
            total = len(document)
            for index in range(min(total, most)):
                page = document[index]
                try:
                    longest = max(page.get_size()) or 1.0
                    bitmap = page.render(scale=min(PAGE_SCALE, MAX_SIDE / longest))
                    try:
                        rendered.append(fit(bitmap.to_pil()))
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            document.close()
    return rendered, total


# -------------------------------------------------------------------- reading back


def has_images(messages: Sequence[Any]) -> bool:
    found = [m.get("content") for m in messages if isinstance(m, dict)]
    parts = [p for content in found if isinstance(content, list) for p in content]
    return any(isinstance(p, dict) and p.get("type") == "image" for p in parts)


def takes_images(renderer: Any) -> bool:
    """Return whether the renderer was built with an image processor."""
    return bool(getattr(renderer, "_has_image_processor", False))


def image_key(part: dict[str, Any]) -> str:
    """Return a digest of an image's pixels, or of the string the image was given as."""
    image = part.get("image")
    if isinstance(image, str):
        return hashlib.sha256(image.encode("utf-8")).hexdigest()[:16]
    try:
        digest = hashlib.sha256()
        digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
        digest.update(image.tobytes())
        return digest.hexdigest()[:16]
    except Exception:  # noqa: BLE001 - not an image this can read; still one entry
        return hashlib.sha256(repr(image).encode("utf-8")).hexdigest()[:16]
