"""Images and documents on the wire, carried to the model as image parts: the cookbook's
parsers take text alone, so every image is lifted out as a sentinel before they run and
put back as a part after, a PDF as one image per page."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
from collections.abc import Sequence
from typing import Any

from shipyard.proxy import cookbook

#: A page of a document becomes one image; a longer document is cut, and the cut is
#: said in a text part so the model knows what it did not see.
MAX_PAGES = 16
#: The longest side an image is scaled down to: a vision renderer charges by area.
MAX_SIDE = 1536
#: The scale a PDF page is rasterised at: 2 is about 144 dpi, enough for small type.
PAGE_SCALE = 2.0

TAG = "vision"
PDF = "application/pdf"
NOT_FETCHED = "an image given by URL is not fetched by this proxy; send its bytes as base64"
_OPEN = "\x00shipyard-image:"
_CLOSE = "\x00"
_SENTINEL = re.compile(re.escape(_OPEN) + r"(\d+)" + re.escape(_CLOSE))
_DATA_URI = re.compile(r"^data:([^;,]+)(;base64)?,(.*)$", re.DOTALL)


class Bank:
    """The images lifted out of one request, in order, each standing in as a sentinel."""

    def __init__(self) -> None:
        self.images: list[Any] = []

    def add(self, image: Any) -> str:
        self.images.append(image)
        return f"{_OPEN}{len(self.images) - 1}{_CLOSE}"


def install() -> None:
    """Wrap the cookbook's two parsers, once per process: a second install, in whatever
    order thinking's came, finds its tag and leaves the wrap it finds."""
    if cookbook.wrapped_by("_parse_openai", TAG):
        return
    anthropic, openai = cookbook.private("_parse_anthropic"), cookbook.private("_parse_openai")

    def parse_anthropic(body: dict[str, Any]) -> Any:
        bank = Bank()
        return restore(anthropic(lift_anthropic(body, bank)), bank)

    def parse_openai(body: dict[str, Any]) -> Any:
        bank = Bank()
        return restore(openai(lift_openai(body, bank)), bank)

    cookbook.patch(
        _parse_anthropic=cookbook.tagged(parse_anthropic, anthropic, TAG),
        _parse_openai=cookbook.tagged(parse_openai, openai, TAG),
    )


# ------------------------------------------------------------------------- lifting


def lift_anthropic(body: dict[str, Any], bank: Bank) -> dict[str, Any]:
    """The body with every image and document block replaced by a sentinel text block."""
    out = dict(body)
    if "system" in out:
        out["system"] = _lift(out["system"], bank, wire="anthropic")
    return _lifted(out, bank, wire="anthropic")


def lift_openai(body: dict[str, Any], bank: Bank) -> dict[str, Any]:
    return _lifted(dict(body), bank, wire="openai")


def _lifted(body: dict[str, Any], bank: Bank, *, wire: str) -> dict[str, Any]:
    if isinstance(body.get("messages"), list):
        body["messages"] = [
            {**m, "content": _lift(m.get("content"), bank, wire=wire)} if isinstance(m, dict) else m
            for m in body["messages"]
        ]
    return body


def _lift(content: Any, bank: Bank, *, wire: str) -> Any:
    if not isinstance(content, list):
        return content
    lifted: list[Any] = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if wire == "anthropic" and kind == "image":
            lifted.append(_text(bank.add(_anthropic_image(block))))
        elif wire == "anthropic" and kind == "document":
            lifted.extend(_text(t) for t in _anthropic_document(block, bank))
        elif wire == "anthropic" and kind == "tool_result":
            lifted.append({**block, "content": _lift(block.get("content"), bank, wire=wire)})
        elif wire == "openai" and kind == "image_url":
            lifted.append(_text(bank.add(_openai_image(block))))
        elif wire == "openai" and kind == "file":
            lifted.extend(_text(t) for t in _openai_file(block, bank))
        else:
            lifted.append(block)
    return lifted


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _anthropic_image(block: dict[str, Any]) -> Any:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return decode_image(source.get("data"), str(source.get("media_type") or ""))
    if source.get("type") == "url":
        raise cookbook.refused(NOT_FETCHED)
    raise cookbook.refused(f"unsupported image source {source.get('type')!r}")


def _anthropic_document(block: dict[str, Any], bank: Bank) -> list[str]:
    source = block.get("source") or {}
    media = str(source.get("media_type") or "")
    if source.get("type") == "text":
        return [str(source.get("data") or "")]
    if source.get("type") == "base64" and media == PDF:
        return _placed(pages(decode_bytes(source.get("data"))), bank)
    if source.get("type") == "base64" and media.startswith("image/"):
        return [bank.add(decode_image(source.get("data"), media))]
    if source.get("type") == "url":
        raise cookbook.refused(NOT_FETCHED)
    raise cookbook.refused(
        f"unsupported document {media or source.get('type')!r}: PDF, image or text"
    )


def _openai_image(block: dict[str, Any]) -> Any:
    url = block.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    if not isinstance(url, str):
        raise cookbook.refused("'image_url' must carry a url")
    match = _DATA_URI.match(url)
    if match is None:
        raise cookbook.refused(NOT_FETCHED)
    media, is_b64, payload = match.groups()
    if not is_b64:
        raise cookbook.refused("an image data URI must be base64")
    return decode_image(payload, media)


def _openai_file(block: dict[str, Any], bank: Bank) -> list[str]:
    spec = block.get("file") or {}
    data = spec.get("file_data")
    if not isinstance(data, str):
        raise cookbook.refused("a 'file' part must carry file_data; file ids are not resolved here")
    match = _DATA_URI.match(data)
    media, payload = (match.group(1), match.group(3)) if match else (PDF, data)
    if media == PDF:
        return _placed(pages(decode_bytes(payload)), bank)
    if media.startswith("image/"):
        return [bank.add(decode_image(payload, media))]
    raise cookbook.refused(f"unsupported file {media!r}: PDF or image")


def _placed(images: Sequence[Any], bank: Bank) -> list[str]:
    """One sentinel per page, and a note when the document was cut."""
    placed = [bank.add(image) for image in images[:MAX_PAGES]]
    if len(images) > MAX_PAGES:
        placed.append(f"\n[document cut: {MAX_PAGES} of {len(images)} pages shown]\n")
    return placed


# ------------------------------------------------------------------------ decoding


def decode_bytes(data: Any) -> bytes:
    if not isinstance(data, str) or not data:
        raise cookbook.refused("base64 data missing")
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as bad:
        raise cookbook.refused(f"base64 data unreadable: {bad}") from bad


def decode_image(data: Any, media: str) -> Any:
    from PIL import Image, UnidentifiedImageError

    raw = decode_bytes(data)
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as bad:
        raise cookbook.refused(f"image data ({media or 'unknown type'}) unreadable: {bad}") from bad
    return fit(image)


def fit(image: Any) -> Any:
    """RGB, no larger than `MAX_SIDE` on its longest side."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        image = image.copy()
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def pages(pdf: bytes) -> list[Any]:
    """Every page of a PDF as an image, in order. The count is the caller's to cap."""
    import pypdfium2

    try:
        document = pypdfium2.PdfDocument(pdf)
    except Exception as bad:  # noqa: BLE001 - pdfium's own error type is not an API
        raise cookbook.refused(f"PDF unreadable: {bad}") from bad
    rendered: list[Any] = []
    for index in range(len(document)):
        page = document[index]
        rendered.append(fit(page.render(scale=PAGE_SCALE).to_pil()))
        page.close()
    document.close()
    return rendered


# ----------------------------------------------------------------------- restoring


def restore(parsed: Any, bank: Bank) -> Any:
    """The parsed chat with every sentinel-bearing message split back into parts, whether
    its content came back as text or as parts another wrap already split."""
    if not bank.images:
        return parsed
    for message in parsed.messages:
        content = message.get("content")
        if isinstance(content, str) and _OPEN in content:
            message["content"] = split(content, bank)
        elif isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.extend(split(str(part.get("text") or ""), bank))
                else:
                    parts.append(part)
            message["content"] = parts
    system = getattr(parsed, "system_text", None)
    if isinstance(system, str) and _OPEN in system:
        # A system prompt is rendered as text by every renderer; an image there has
        # nowhere to go, so it is named rather than dropped in silence.
        parsed.system_text = _SENTINEL.sub(
            "[an image was here; images in the system prompt are not shown to the model]", system
        )
    return parsed


def split(text: str, bank: Bank) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    at = 0
    for match in _SENTINEL.finditer(text):
        if match.start() > at:
            parts.append({"type": "text", "text": text[at : match.start()]})
        parts.append({"type": "image", "image": bank.images[int(match.group(1))]})
        at = match.end()
    if at < len(text):
        parts.append({"type": "text", "text": text[at:]})
    return parts


# -------------------------------------------------------------------- reading back


def has_images(messages: Sequence[Any]) -> bool:
    found = [m.get("content") for m in messages if isinstance(m, dict)]
    parts = [p for content in found if isinstance(content, list) for p in content]
    return any(isinstance(p, dict) and p.get("type") == "image" for p in parts)


def takes_images(renderer: Any) -> bool:
    """Whether a renderer, through any wrappers, was built with an image processor."""
    for _ in range(8):
        flag = getattr(renderer, "_has_image_processor", None)
        if flag is not None or renderer is None:
            return bool(flag)
        renderer = getattr(renderer, "_inner", None) or getattr(renderer, "inner", None)
    return False


def image_key(part: dict[str, Any]) -> str:
    """What identifies an image: a digest of its pixels, or of the string it was given as."""
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
