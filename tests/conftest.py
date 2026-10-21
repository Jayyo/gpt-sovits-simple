import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simple_tts  # noqa: E402


@pytest.fixture
def fake_root(tmp_path):
    """A directory that looks like a GPT-SoVITS checkout, with empty model files."""
    root = tmp_path / "GPT-SoVITS"
    (root / simple_tts._MARKER).parent.mkdir(parents=True)
    (root / simple_tts._MARKER).write_text("")
    for path in simple_tts.DEFAULT_MODELS["v2"]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"")
    for path in simple_tts.SHARED_MODELS:
        (root / path).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def make_wav(tmp_path):
    def make(seconds, name="ref.wav", rate=16000):
        path = tmp_path / name
        t = np.arange(int(seconds * rate)) / rate
        sf.write(path, 0.3 * np.sin(2 * np.pi * 220 * t), rate)
        return path
    return make


class FakeEngine:
    """Stands in for GPT-SoVITS's TTS: records the inputs and the cwd while running."""

    def __init__(self, results):
        self.results = results
        self.calls = []
        self.prompt_cache = {"prompt_text": None, "prompt_lang": None}

    def run(self, inputs):
        self.calls.append((dict(inputs), Path.cwd()))
        self.prompt_cache["prompt_lang"] = inputs["prompt_lang"]
        for item in self.results:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.fixture
def fake_engine():
    return FakeEngine


FAKE_UPSTREAM = """
import os

calls = {"configs": [], "engines": 0, "saved": 0}


class TTS_Config:
    def __init__(self, configs):
        calls["configs"].append(configs)
        self.cwd = os.getcwd()

    def save_configs(self, configs_path=None):
        calls["saved"] += 1


class TTS:
    def __init__(self, config):
        calls["engines"] += 1
        self.configs = config
        config.save_configs()  # upstream does this while loading weights
        self.prompt_cache = {"prompt_text": None, "prompt_lang": None}
"""


@pytest.fixture
def fake_upstream(fake_root):
    """fake_root plus a TTS_infer_pack.TTS module that records how it is used."""
    (fake_root / simple_tts._MARKER).write_text(FAKE_UPSTREAM)
    (fake_root / simple_tts._MARKER).with_name("__init__.py").write_text("")
    return fake_root


@pytest.fixture(autouse=True)
def forget_upstream_modules():
    """Each test gets its own fake checkout, so drop whatever one imported."""
    yield
    for name in ("TTS_infer_pack.TTS", "TTS_infer_pack"):
        sys.modules.pop(name, None)
