import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import simple_tts
from simple_tts import SetupError, SimpleTTS


def test_find_checkout(fake_root, tmp_path, monkeypatch):
    monkeypatch.delenv("GPT_SOVITS_DIR", raising=False)
    assert simple_tts.find_gpt_sovits_dir(fake_root) == fake_root

    monkeypatch.setenv("GPT_SOVITS_DIR", str(fake_root))
    assert simple_tts.find_gpt_sovits_dir() == fake_root

    # A wrong explicit setting is an error, not a reason to look elsewhere
    monkeypatch.setenv("GPT_SOVITS_DIR", str(tmp_path))
    monkeypatch.chdir(fake_root)
    with pytest.raises(SetupError, match="GPT_SOVITS_DIR"):
        simple_tts.find_gpt_sovits_dir()

    monkeypatch.delenv("GPT_SOVITS_DIR")
    assert simple_tts.find_gpt_sovits_dir() == fake_root


@pytest.mark.parametrize("seconds, ok", [(2.0, False), (3.0, True), (6.5, True), (10.0, True), (10.5, False)])
def test_reference_length(make_wav, seconds, ok):
    path = make_wav(seconds)
    if ok:
        assert simple_tts.check_reference(path) == pytest.approx(seconds)
    else:
        with pytest.raises(SetupError, match="3-10 seconds"):
            simple_tts.check_reference(path)


def test_reference_missing_or_unreadable(tmp_path):
    with pytest.raises(SetupError, match="not found"):
        simple_tts.check_reference(tmp_path / "nope.wav")
    junk = tmp_path / "junk.wav"
    junk.write_text("not audio")
    with pytest.raises(SetupError, match="Cannot read"):
        simple_tts.check_reference(junk)


def test_paths_are_resolved_against_the_callers_directory(fake_root, tmp_path, monkeypatch, make_wav, fake_engine):
    ref = make_wav(4)
    monkeypatch.chdir(tmp_path)
    tts = SimpleTTS(ref_audio="ref.wav", ref_text="hello", gpt_sovits_dir=fake_root)
    engine = fake_engine([(32000, (np.ones(32000) * 1000).astype(np.int16))])
    tts._tts = engine

    path = tts.generate("Hello there", "out/hello.wav", language="en")

    assert Path(path) == tmp_path / "out" / "hello.wav"
    assert sf.info(path).samplerate == 32000
    inputs, cwd_during_run = engine.calls[0]
    assert cwd_during_run == fake_root
    assert os.getcwd() == str(tmp_path)
    assert inputs["ref_audio_path"] == str(ref)
    assert inputs["prompt_lang"] == "en" and inputs["return_fragment"] is False
    assert not list((tmp_path / "out").glob(".*.part"))


def test_generation_failures_write_nothing(fake_root, tmp_path, make_wav, fake_engine):
    tts = SimpleTTS(ref_audio=make_wav(4), ref_text="hi", gpt_sovits_dir=fake_root)
    silence = (16000, np.zeros(16000, dtype=np.int16))
    sound = (32000, np.ones(32000, dtype=np.int16))
    cases = [
        ([silence, RuntimeError("CUDA out of memory")], RuntimeError, "out of memory"),
        ([], RuntimeError, "0 results"),
        ([sound, sound], RuntimeError, "2 results"),
        ([silence], RuntimeError, "only silence"),
    ]
    for results, error, message in cases:
        tts._tts = fake_engine(results)
        out = tmp_path / "out.wav"
        with pytest.raises(error, match=message):
            tts.generate("Hello", out)
        assert not out.exists()
        assert os.getcwd() != str(fake_root)


def test_ref_language_change_resets_prompt_cache(fake_root, make_wav, tmp_path, fake_engine):
    tts = SimpleTTS(ref_audio=make_wav(4), ref_text="안녕하세요 반갑습니다", gpt_sovits_dir=fake_root)
    engine = fake_engine([(32000, np.ones(32000, dtype=np.int16))])
    tts._tts = engine
    engine.prompt_cache.update(prompt_text="안녕하세요 반갑습니다.", prompt_lang="ko")
    tts.generate("Hello", tmp_path / "a.wav", language="en", ref_language="ko")
    assert engine.prompt_cache["prompt_text"] == "안녕하세요 반갑습니다."
    tts.generate("Hello", tmp_path / "b.wav", language="en", ref_language="all_ko")
    assert engine.prompt_cache["prompt_text"] is None


def test_input_validation_happens_before_loading(fake_root, make_wav, tmp_path):
    tts = SimpleTTS(gpt_sovits_dir=fake_root)
    with pytest.raises(ValueError, match="reference clip"):
        tts.generate("Hello")
    with pytest.raises(ValueError, match="empty"):
        tts.generate("   ", ref_audio=make_wav(4))
    with pytest.raises(ValueError, match="speed"):
        tts.generate("Hello", ref_audio=make_wav(4), speed=5)
    with pytest.raises(ValueError, match=".mp4"):
        tts.generate("Hello", tmp_path / "x.mp4", ref_audio=make_wav(4))
    ref = make_wav(4)
    with pytest.raises(ValueError, match="reference clip"):
        tts.generate("Hello", ref, ref_audio=ref)
    v4 = SimpleTTS(gpt_sovits_dir=fake_root, model_version="v4")
    with pytest.raises(ValueError, match="transcript"):
        v4.generate("Hello", ref_audio=ref)
    assert tts._tts is None and v4._tts is None


def test_empty_ref_text_override(fake_root, make_wav, tmp_path, fake_engine):
    tts = SimpleTTS(ref_audio=make_wav(4), ref_text="default transcript", gpt_sovits_dir=fake_root)
    engine = fake_engine([(32000, np.ones(32000, dtype=np.int16))])
    tts._tts = engine
    tts.generate("Hello", tmp_path / "a.wav", ref_text="")
    assert engine.calls[0][0]["prompt_text"] == ""


def test_shared_instance_refuses_conflicting_settings(fake_root, monkeypatch):
    monkeypatch.setattr(simple_tts, "_default_tts", None)
    first = simple_tts.get_tts(gpt_sovits_dir=fake_root)
    assert simple_tts.get_tts() is first
    with pytest.raises(ValueError, match="different settings"):
        simple_tts.get_tts(gpt_sovits_dir=fake_root, device="cpu")


def test_load_model_passes_settings_under_custom(fake_upstream, monkeypatch, make_wav):
    monkeypatch.setattr(simple_tts, "_pick_device", lambda device: "cpu")
    path_before = list(sys.path)
    tts = SimpleTTS(ref_audio=make_wav(4), gpt_sovits_dir=fake_upstream, is_half=True)
    engine = tts._load_model()
    assert tts._load_model() is engine

    upstream = sys.modules["TTS_infer_pack.TTS"].calls
    assert upstream["engines"] == 1
    assert upstream["saved"] == 0  # our subclass swallows save_configs()
    custom = upstream["configs"][0]["custom"]
    assert custom["device"] == "cpu" and custom["is_half"] is False and custom["version"] == "v2"
    assert custom["t2s_weights_path"] == tts.gpt_model and custom["vits_weights_path"] == tts.sovits_model
    assert engine.configs.cwd == str(fake_upstream)
    assert sys.path == path_before


def test_second_checkout_in_one_process_is_refused(fake_upstream, tmp_path, monkeypatch):
    monkeypatch.setattr(simple_tts, "_pick_device", lambda device: "cpu")
    SimpleTTS(gpt_sovits_dir=fake_upstream)._load_model()
    other = tmp_path / "other"
    (other / simple_tts._MARKER).parent.mkdir(parents=True)
    (other / simple_tts._MARKER).write_text("")
    with pytest.raises(SetupError, match="already loaded"):
        with simple_tts._inside(other):
            pass


def test_leftover_package_from_a_failed_import_is_noticed(fake_root, tmp_path, monkeypatch):
    # A failed "import TTS_infer_pack.TTS" can leave the package behind
    package = SimpleNamespace(__file__=None, __path__=[str(fake_root / "GPT_SoVITS" / "TTS_infer_pack")])
    monkeypatch.setitem(sys.modules, "TTS_infer_pack", package)
    other = tmp_path / "other"
    (other / simple_tts._MARKER).parent.mkdir(parents=True)
    (other / simple_tts._MARKER).write_text("")
    with pytest.raises(SetupError, match="already loaded"):
        with simple_tts._inside(other):
            pass
    with simple_tts._inside(fake_root):
        pass


def test_soundfile_patch_is_scoped(monkeypatch, make_wav, tmp_path):
    original_calls = []
    fake_torchaudio = SimpleNamespace(load=lambda *a, **k: original_calls.append((a, k)) or "original")
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda array: array))
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((1600, 2), dtype=np.float32), 16000)

    with simple_tts._soundfile_load():
        audio, rate = fake_torchaudio.load(str(make_wav(1)))
        assert audio.shape == (1, 16000) and audio.dtype == np.float32 and rate == 16000
        assert fake_torchaudio.load(str(stereo))[0].shape == (2, 1600)
        assert fake_torchaudio.load("x.wav", frame_offset=10) == "original"
    assert original_calls == [(("x.wav",), {"frame_offset": 10})]

    with pytest.raises(RuntimeError):
        with simple_tts._soundfile_load():
            raise RuntimeError("boom")
    assert fake_torchaudio.load("y.wav") == "original"


def test_missing_models_are_reported_before_touching_torch(fake_root, make_wav, tmp_path, monkeypatch):
    def no_torch(device):
        raise AssertionError("torch must not be needed to report missing models")

    monkeypatch.setattr(simple_tts, "_pick_device", no_torch)
    tts = SimpleTTS(sovits_model=tmp_path / "typo.pth", ref_audio=make_wav(4), gpt_sovits_dir=fake_root)
    with pytest.raises(SetupError, match="typo.pth"):
        tts.generate("Hello", tmp_path / "out.wav")


def test_failed_write_keeps_the_old_file(fake_root, make_wav, tmp_path, fake_engine, monkeypatch):
    tts = SimpleTTS(ref_audio=make_wav(4), ref_text="hi", gpt_sovits_dir=fake_root)
    tts._tts = fake_engine([(32000, np.ones(32000, dtype=np.int16))])
    out = tmp_path / "out.wav"
    out.write_bytes(b"previous take")

    def broken_write(file, *args, **kwargs):
        Path(file).write_bytes(b"half")
        raise RuntimeError("disk full")

    monkeypatch.setattr(sf, "write", broken_write)
    with pytest.raises(RuntimeError, match="disk full"):
        tts.generate("Hello", out)
    assert out.read_bytes() == b"previous take"
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".part") == []


def test_api_checks_sampling_settings(fake_root, make_wav):
    tts = SimpleTTS(ref_audio=make_wav(4), gpt_sovits_dir=fake_root)
    for kwargs in ({"top_k": 0}, {"top_k": 1.5}, {"top_p": 0}, {"temperature": float("nan")}, {"seed": -2}):
        with pytest.raises(ValueError):
            tts.generate("Hello", **kwargs)
    v4 = SimpleTTS(ref_audio=make_wav(4), gpt_sovits_dir=fake_root, model_version="v4")
    with pytest.raises(ValueError, match="transcript"):
        v4.generate("Hello", ref_text="   ")
