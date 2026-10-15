"""Each frozen case replayed through the trainer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden import CASES, differences, read_frozen, trained

THROUGH_THE_TRAINER = CASES


@pytest.mark.parametrize("name", THROUGH_THE_TRAINER)
async def test_the_trainer_reproduces_the_frozen_case(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = read_frozen(name)
    assert frozen is not None
    found = await trained(name, tmp_path, monkeypatch)
    expected = {key: frozen[key] for key in found if key != "groups"}
    if "groups" in found:
        expected["groups"] = [
            {"task": one["task"], "members": one["members"]} for one in frozen["groups"]
        ]
    found_differences = differences(found, expected)
    assert not found_differences, "\n".join(found_differences)
