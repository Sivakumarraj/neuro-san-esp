"""The beginner's guide is built from the code, so its examples are checked."""

from __future__ import annotations

import re

import pytest

pypdf = pytest.importorskip("pypdf")

from esp.eval.tasks import TASKS  # noqa: E402
from esp.report.guide import Guide  # noqa: E402


@pytest.fixture(scope="module")
def text(tmp_path_factory):
    out = tmp_path_factory.mktemp("guide") / "guide.pdf"
    Guide().build(out)
    reader = pypdf.PdfReader(str(out))
    assert len(reader.pages) >= 5
    return re.sub(r"\s+", " ", " ".join(page.extract_text() for page in reader.pages))


def test_the_worked_question_is_a_real_one_with_its_real_answer(text):
    task = TASKS[1]
    assert f"Question {task.task_id}" in text and f"Answer: {task.answer}" in text


def test_the_worked_score_is_the_committed_one(text):
    assert "0.7761" in text and "0.8941" in text


def test_every_command_it_tells_a_beginner_to_run_exists(text):
    from pathlib import Path
    makefile = (Path(__file__).resolve().parent.parent / "Makefile").read_text()
    targets = set(re.findall(r"^([a-z][\w-]*):", makefile, re.M))
    for target in set(re.findall(r"make ([a-z][\w-]*)", text)):
        assert target in targets, f"the guide tells a beginner to run make {target}"
