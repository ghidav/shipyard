"""Tests for images and PDF pages on both wires, passed to the model as image parts."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from harness_proxy import vision
from harness_proxy.chat import Refused
from harness_proxy.dialects import anthropic, openai
from proxies import FakeSampler, ask, ask_anthropic, recorded, server


def _png(width: int = 40, height: int = 30, color: str = "red") -> str:
    image = Image.new("RGB", (width, height), color)
    sink = io.BytesIO()
    image.save(sink, format="PNG")
    return base64.b64encode(sink.getvalue()).decode()


def _pdf(pages: int = 2) -> str:
    import pypdfium2

    document = pypdfium2.PdfDocument.new()
    for _ in range(pages):
        document.new_page(200, 100)
    sink = io.BytesIO()
    document.save(sink)
    return base64.b64encode(sink.getvalue()).decode()


@pytest.fixture
def parsers():
    """The two wires' readers."""
    return SimpleNamespace(anthropic=anthropic.read, openai=openai.read)


def _parts(message: Any) -> list[str]:
    content = message["content"]
    if isinstance(content, str):
        return ["text"]
    return [part["type"] for part in content]


def _image(data: str, media: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _document(data: str, media: str) -> dict[str, Any]:
    return {"type": "document", "source": {"type": "base64", "media_type": media, "data": data}}


def test_an_anthropic_image_block_reaches_the_renderer_as_an_image_part(parsers) -> None:
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    _image(_png()),
                    {"type": "text", "text": "Answer briefly."},
                ],
            }
        ],
    }
    (message,) = parsers.anthropic(body).messages
    assert _parts(message) == ["text", "image", "text"]
    assert message["content"][0]["text"] == "What is this?"
    assert message["content"][1]["image"].size == (40, 30)
    assert message["content"][2]["text"] == "Answer briefly."
    assert vision.has_images([message])


def test_a_pdf_in_a_tool_result_becomes_one_image_per_page(parsers) -> None:
    """Claude Code's `Read` of a PDF arrives inside a `tool_result` block."""
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "The file:"},
                            _document(_pdf(2), "application/pdf"),
                        ],
                    }
                ],
            },
        ],
    }
    result = parsers.anthropic(body).messages[-1]
    assert result["role"] == "tool" and result.get("name") == "Read"
    assert _parts(result) == ["text", "image", "image"]
    assert all(part["image"].size == (400, 200) for part in result["content"][1:])


def test_a_long_document_is_cut_and_says_so(parsers, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vision, "MAX_PAGES", 2)
    body = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": [_document(_pdf(3), "application/pdf")]}],
    }
    (message,) = parsers.anthropic(body).messages
    assert _parts(message) == ["image", "image", "text"]
    assert "2 of 3 pages" in message["content"][-1]["text"]


def test_an_openai_image_url_and_file_part_reach_the_renderer(parsers) -> None:
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_png(8, 8, 'blue')}"},
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": "a.pdf",
                            "file_data": f"data:application/pdf;base64,{_pdf(1)}",
                        },
                    },
                ],
            }
        ],
    }
    (message,) = parsers.openai(body).messages
    assert _parts(message) == ["text", "image", "image"]
    assert message["content"][1]["image"].size == (8, 8)


def test_a_text_only_body_is_read_as_plain_text(parsers) -> None:
    body = {
        "model": "m",
        "max_tokens": 10,
        "system": "be brief",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {"p": "."}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a b"}],
            },
        ],
    }
    parsed = parsers.anthropic(body)
    assert [m["content"] for m in parsed.messages] == ["hi", "", "a b"]
    assert parsed.system == "be brief"
    assert not vision.has_images(parsed.messages)


def test_urls_and_unknown_kinds_are_refused_as_bad_requests(parsers) -> None:
    def body(block: dict[str, Any]) -> dict[str, Any]:
        return {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": [block]}]}

    with pytest.raises(Refused, match="not fetched"):
        parsers.anthropic(
            body({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}})
        )
    with pytest.raises(Refused, match="unreadable"):
        parsers.anthropic(body(_image(base64.b64encode(b"nope").decode())))
    with pytest.raises(Refused, match="PDF, image or text"):
        parsers.anthropic(body(_document("AA==", "audio/wav")))
    with pytest.raises(Refused, match="not fetched"):
        parsers.openai(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                    }
                ],
            }
        )


def test_an_image_in_the_system_prompt_is_named_not_dropped_in_silence(parsers) -> None:
    body = {
        "model": "m",
        "max_tokens": 1,
        "system": [{"type": "text", "text": "You see:"}, _image(_png())],
        "messages": [{"role": "user", "content": "hi"}],
    }
    parsed = parsers.anthropic(body)
    assert parsed.system == f"You see:{vision.IN_SYSTEM}"


def test_a_big_image_is_scaled_down_and_rgb() -> None:
    image = Image.new("RGBA", (4000, 1000), (1, 2, 3, 4))
    fitted = vision.fit(image)
    assert fitted.mode == "RGB" and max(fitted.size) == vision.MAX_SIDE


def test_thinking_text_and_images_keep_their_order_in_one_message(parsers) -> None:
    body = {
        "model": "m",
        "max_tokens": 1,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t"},
                    {"type": "text", "text": "a"},
                    _image(_png()),
                    {"type": "text", "text": "b"},
                    {"type": "text", "text": "c"},
                ],
            }
        ],
    }
    (message,) = parsers.anthropic(body).messages
    assert _parts(message) == ["thinking", "text", "image", "text"]
    assert message["content"][3]["text"] == "bc", "adjacent text is one part"


def test_an_image_key_is_the_pixels_or_the_string() -> None:
    red = {"type": "image", "image": Image.new("RGB", (4, 4), "red")}
    assert vision.image_key(red) == vision.image_key(
        {"type": "image", "image": Image.new("RGB", (4, 4), "red")}
    )
    assert vision.image_key(red) != vision.image_key(
        {"type": "image", "image": Image.new("RGB", (4, 4), "blue")}
    )
    assert vision.image_key({"image": "data:x"}) == vision.image_key({"image": "data:x"})
    assert vision.image_key({"image": object()})  # still one entry


def test_takes_images_reads_the_renderers_flag() -> None:
    assert vision.takes_images(SimpleNamespace(_has_image_processor=True))
    assert not vision.takes_images(SimpleNamespace(_has_image_processor=False))
    assert not vision.takes_images(SimpleNamespace())


async def test_a_text_only_model_refuses_an_image_with_a_400_on_both_wires() -> None:
    """The fake renderer has no image processor, as with a text-only model."""
    sampler = FakeSampler("ok")
    async with server(sampler) as started:
        openai = await ask(
            started,
            "t",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_png()}"},
                        },
                    ],
                }
            ],
        )
        anthropic = await ask_anthropic(
            started, "t", messages=[{"role": "user", "content": [_image(_png())]}]
        )
        plain = await ask(started, "t")
        assert len(recorded(started, "t")) == 1, "a refused image costs no sample"
    assert openai.status_code == 400 and "takes none" in openai.json()["error"]["message"]
    assert anthropic.status_code == 400 and "takes none" in anthropic.json()["error"]["message"]
    assert plain.status_code == 200 and sampler.asked and len(sampler.asked) == 1


def test_only_the_pages_shown_are_rendered_and_a_big_page_is_rendered_small() -> None:
    import pypdfium2

    images, total = vision.pages(base64.b64decode(_pdf(3)), 2)
    assert len(images) == 2 and total == 3
    document = pypdfium2.PdfDocument.new()
    document.new_page(6000, 6000)
    sink = io.BytesIO()
    document.save(sink)
    (page,), _ = vision.pages(sink.getvalue(), 16)
    assert max(page.size) <= vision.MAX_SIDE, "rasterised small, not shrunk after"


def test_an_image_over_the_pixel_ceiling_is_refused_before_it_is_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision, "MAX_PIXELS", 100)
    with pytest.raises(Refused, match="20x20, over the 100 pixels"):
        vision.decode_image(_png(20, 20), "image/png")
    assert vision.decode_image(_png(10, 10), "image/png").size == (10, 10)
