import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_tts


def run(argv, capsys):
    code = simple_tts.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_import_has_no_side_effects(tmp_path):
    code = (
        "import os, sys; cwd, path = os.getcwd(), list(sys.path); import simple_tts; "
        "assert os.getcwd() == cwd and sys.path == path; "
        "assert not {'torch', 'torchaudio', 'TTS_infer_pack'} & set(sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(simple_tts.__file__).parent)}
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


def test_help_and_version(capsys):
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as exit_info:
            simple_tts.main([flag])
        assert exit_info.value.code == 0
    assert "gpt-sovits-simple" in capsys.readouterr().out


@pytest.mark.parametrize("argv, message", [
    (["-r", "ref.wav"], "either as an argument or with --text-file"),
    (["hi", "-f", "x.txt", "-r", "ref.wav"], "either as an argument or with --text-file"),
    (["hi"], "reference clip is required"),
    (["hi", "a.wav", "-o", "b.wav", "-r", "ref.wav"], "output path once"),
    (["hi", "-r", "ref.wav", "--speed", "nan"], "outside"),
    (["hi", "-r", "ref.wav", "--speed", "3"], "outside"),
    (["hi", "-r", "ref.wav", "--top-p", "0"], "outside"),
    (["hi", "-r", "ref.wav", "-l", "korean"], "invalid choice"),
])
def test_bad_arguments(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        simple_tts.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_checkout_is_a_clear_error(tmp_path, monkeypatch, make_wav, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GPT_SOVITS_DIR", raising=False)
    monkeypatch.setattr(simple_tts, "__file__", str(tmp_path / "simple_tts.py"))
    code, _, err = run(["hi", "-r", str(make_wav(4))], capsys)
    assert code == 2
    assert "Could not find GPT-SoVITS" in err


def test_legacy_positional_output_and_cpu_flag(fake_root, make_wav, monkeypatch, capsys):
    seen = {}

    def fake_generate(self, text, output_path=None, **kwargs):
        seen.update(text=text, output=output_path, device=self.device, **kwargs)
        return output_path

    monkeypatch.setattr(simple_tts.SimpleTTS, "generate", fake_generate)
    code, out, _ = run(["Hello", "old.wav", "-r", str(make_wav(4)), "--cpu",
                        "--gpt-sovits-dir", str(fake_root), "-l", "en"], capsys)
    assert code == 0 and out.strip() == "old.wav"
    assert seen["output"] == "old.wav" and seen["device"] == "cpu" and seen["language"] == "en"
    assert seen["ref_language"] is None and seen["text_split_method"] == "cut1"


def test_text_file(fake_root, make_wav, tmp_path, monkeypatch, capsys):
    script = tmp_path / "script.txt"
    script.write_text("첫 문장입니다.\n두 번째 문장입니다.\n", encoding="utf-8")
    texts = []
    monkeypatch.setattr(simple_tts.SimpleTTS, "generate", lambda self, text, *a, **k: texts.append(text) or "x.wav")
    code, _, _ = run(["-f", str(script), "-r", str(make_wav(4)), "--gpt-sovits-dir", str(fake_root)], capsys)
    assert code == 0 and texts == ["첫 문장입니다.\n두 번째 문장입니다.\n"]


def test_check_reports_missing_models(fake_root, monkeypatch, capsys):
    (fake_root / simple_tts.DEFAULT_MODELS["v2"][1]).unlink()
    code, out, _ = run(["--check", "--gpt-sovits-dir", str(fake_root)], capsys)
    assert code == 1
    assert "FAIL  model files" in out and "s2G2333k.pth" in out


def test_check_passes_with_a_complete_setup(fake_upstream, make_wav, monkeypatch, capsys):
    fake_cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    fake_torch = SimpleNamespace(__version__="2.11", cuda=fake_cuda, device=lambda name: SimpleNamespace(type=name))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(__version__="2.11", load=None))
    code, out, _ = run(["--check", "--gpt-sovits-dir", str(fake_upstream), "-r", str(make_wav(4))], capsys)
    assert code == 0, out
    assert "FAIL" not in out and "cpu (CPU)" in out and "reference clip" in out

    code, out, _ = run(["--check", "--gpt-sovits-dir", str(fake_upstream), "--device", "cuda"], capsys)
    assert code == 1 and "FAIL  device" in out
