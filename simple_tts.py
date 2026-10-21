#!/usr/bin/env python
"""Text-to-speech with an existing GPT-SoVITS checkout.

Clones the voice of a 3-10 second reference clip using GPT-SoVITS's own
inference code (GPT_SoVITS/TTS_infer_pack). This file only adds a command
line, input checks and a small Python API on top.

    python simple_tts.py "Hello there." hello.wav -r ref.wav -t "Words spoken in ref.wav"
"""
import argparse
import contextlib
import math
import os
import sys
import tempfile
from pathlib import Path

__version__ = "0.2.0"

LANGUAGES = ("auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko")
SPLIT_METHODS = ("cut0", "cut1", "cut2", "cut3", "cut4", "cut5")
REF_MIN_SECONDS, REF_MAX_SECONDS = 3.0, 10.0
DEFAULT_OUTPUT = "output/tts_output.wav"

# Pretrained files per model version, relative to the checkout. Mirrors
# TTS_Config.default_configs in GPT-SoVITS (checked against commit d523079).
_PRETRAINED = "GPT_SoVITS/pretrained_models"
DEFAULT_MODELS = {
    "v1": (f"{_PRETRAINED}/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", f"{_PRETRAINED}/s2G488k.pth"),
    "v2": (f"{_PRETRAINED}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
           f"{_PRETRAINED}/gsv-v2final-pretrained/s2G2333k.pth"),
    "v2Pro": (f"{_PRETRAINED}/s1v3.ckpt", f"{_PRETRAINED}/v2Pro/s2Gv2Pro.pth"),
    "v2ProPlus": (f"{_PRETRAINED}/s1v3.ckpt", f"{_PRETRAINED}/v2Pro/s2Gv2ProPlus.pth"),
    "v3": (f"{_PRETRAINED}/s1v3.ckpt", f"{_PRETRAINED}/s2Gv3.pth"),
    "v4": (f"{_PRETRAINED}/s1v3.ckpt", f"{_PRETRAINED}/gsv-v4-pretrained/s2Gv4.pth"),
}
SHARED_MODELS = (f"{_PRETRAINED}/chinese-hubert-base", f"{_PRETRAINED}/chinese-roberta-wwm-ext-large")
NEEDS_REF_TEXT = ("v3", "v4")
_MARKER = Path("GPT_SoVITS") / "TTS_infer_pack" / "TTS.py"


class SetupError(Exception):
    """GPT-SoVITS, its models or an input file is not usable. The message says what to fix."""


def find_gpt_sovits_dir(explicit=None):
    """Locate the GPT-SoVITS checkout.

    An explicit path or $GPT_SOVITS_DIR has to be right; otherwise the current
    directory and the directory of this file are tried, so copying this file
    into the checkout keeps working.
    """
    for source, value in (("--gpt-sovits-dir", explicit), ("GPT_SOVITS_DIR", os.environ.get("GPT_SOVITS_DIR"))):
        if value:
            root = Path(value).expanduser().resolve()
            if not (root / _MARKER).is_file():
                raise SetupError(f"{source} is {root}, which is not a GPT-SoVITS checkout (no {_MARKER}).")
            return root
    for candidate in (Path.cwd(), Path(__file__).resolve().parent):
        if (candidate / _MARKER).is_file():
            return candidate
    raise SetupError(
        "Could not find GPT-SoVITS. Pass --gpt-sovits-dir, set GPT_SOVITS_DIR, "
        "or run this from inside the GPT-SoVITS checkout."
    )


def check_reference(path):
    """Return the length of a reference clip, which upstream only accepts at 3-10 seconds."""
    import soundfile as sf

    path = Path(path)
    if not path.is_file():
        raise SetupError(f"Reference audio not found: {path}")
    try:
        info = sf.info(str(path))
    except Exception as e:
        raise SetupError(f"Cannot read reference audio {path} ({e}). WAV or FLAC works best.") from e
    seconds = info.frames / info.samplerate
    if not REF_MIN_SECONDS <= seconds <= REF_MAX_SECONDS:
        raise SetupError(f"Reference audio must be 3-10 seconds long; {path.name} is {seconds:.1f} s.")
    return seconds


def _absolute(path):
    """Resolve a user path against the caller's working directory, before we chdir anywhere."""
    return str(Path(path).expanduser().resolve()) if path else None


def _read_audio(path):
    """torchaudio.load replacement: (channels, frames) float32 tensor and sample rate."""
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T.copy()), sample_rate


@contextlib.contextmanager
def _soundfile_load():
    """Let GPT-SoVITS read the reference clip with soundfile.

    torchaudio 2.9 moved torchaudio.load onto TorchCodec, which GPT-SoVITS
    environments often don't have. Only the plain load(path) call that
    GPT-SoVITS makes is redirected, and only while we are running it.
    """
    try:
        import torchaudio
    except Exception:  # GPT-SoVITS's own import will report the real problem
        yield
        return
    original = torchaudio.load

    def load(uri, *args, **kwargs):
        if args or kwargs:
            return original(uri, *args, **kwargs)
        return _read_audio(uri)

    torchaudio.load = load
    try:
        yield
    finally:
        torchaudio.load = original


@contextlib.contextmanager
def _inside(root):
    """Run GPT-SoVITS code from its checkout.

    It imports its modules from there and opens files relative to it. Its
    progress output goes to stderr so stdout only carries our result.
    """
    for name in ("TTS_infer_pack.TTS", "TTS_infer_pack"):
        location = _module_location(sys.modules.get(name))
        if location is not None and root not in location.parents:
            raise SetupError(
                f"GPT-SoVITS from {location.parent} is already loaded in this process; "
                f"using {root} as well needs a new process."
            )
    added = [p for p in (str(root), str(root / "GPT_SoVITS")) if p not in sys.path]
    sys.path[:0] = added
    previous = os.getcwd()
    os.chdir(root)
    try:
        with contextlib.redirect_stdout(sys.stderr), _soundfile_load():
            yield
    finally:
        os.chdir(previous)
        for path in added:
            if path in sys.path:
                sys.path.remove(path)


def _module_location(module):
    """File (or package folder) a loaded module came from, or None."""
    if module is None:
        return None
    if getattr(module, "__file__", None):
        return Path(module.__file__).resolve()
    paths = list(getattr(module, "__path__", []))
    return Path(paths[0]).resolve() if paths else None


def _pick_device(device):
    """Turn "auto", "cpu", "cuda" or "cuda:N" into a device PyTorch can actually use."""
    try:
        import torch
    except Exception as e:  # missing, or a broken CUDA install that fails to load
        raise SetupError(f"PyTorch is not usable here ({e}). Use the Python environment of GPT-SoVITS.") from e

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as e:
        raise SetupError(f"Unknown device {device!r}; use auto, cpu, cuda or cuda:N.") from e
    if parsed.type == "cpu":
        return "cpu"
    if parsed.type != "cuda":
        raise SetupError(f"Device {device!r} is not supported; use auto, cpu, cuda or cuda:N.")
    if not torch.cuda.is_available():
        raise SetupError("CUDA was requested, but this PyTorch build cannot see a GPU. Use --device cpu.")
    if parsed.index is not None and parsed.index >= torch.cuda.device_count():
        raise SetupError(f"There is no {device}; this machine has {torch.cuda.device_count()} GPU(s).")
    return str(parsed)


class SimpleTTS:
    """Small wrapper around GPT-SoVITS's TTS class.

    The model is loaded on the first generate() call. GPT-SoVITS is not
    thread-safe and imports from a single checkout, so use one instance at a
    time per process.
    """

    def __init__(
        self,
        gpt_model=None,
        sovits_model=None,
        ref_audio=None,
        ref_text=None,
        device="auto",
        is_half=True,
        *,
        model_version="v2",
        gpt_sovits_dir=None,
    ):
        """
        Args:
            gpt_model: GPT (s1) checkpoint; defaults to the pretrained one for model_version
            sovits_model: SoVITS (s2) checkpoint; same default rule
            ref_audio: default reference clip (3-10 s)
            ref_text: what is said in the reference clip
            device: "auto", "cpu", "cuda" or "cuda:N"
            is_half: use fp16 on GPU (ignored on CPU)
            model_version: which pretrained defaults to use (v1, v2, v2Pro, v2ProPlus, v3, v4)
            gpt_sovits_dir: GPT-SoVITS checkout; see find_gpt_sovits_dir()
        """
        if model_version not in DEFAULT_MODELS:
            raise ValueError(f"model_version must be one of {', '.join(DEFAULT_MODELS)}")
        self.root = find_gpt_sovits_dir(gpt_sovits_dir)
        self.model_version = model_version
        default_gpt, default_sovits = DEFAULT_MODELS[model_version]
        self.gpt_model = _absolute(gpt_model) or str(self.root / default_gpt)
        self.sovits_model = _absolute(sovits_model) or str(self.root / default_sovits)
        self.ref_audio = _absolute(ref_audio)
        self.ref_text = ref_text
        self.device = device
        self.is_half = is_half
        self._tts = None

    def missing_models(self):
        """Checkpoints and pretrained folders that are not on disk."""
        missing = [p for p in (self.gpt_model, self.sovits_model) if not Path(p).is_file()]
        missing += [str(self.root / p) for p in SHARED_MODELS if not (self.root / p).is_dir()]
        return missing

    def _load_model(self):
        """Load GPT-SoVITS on first use."""
        if self._tts is not None:
            return self._tts

        missing = self.missing_models()
        if missing:
            raise SetupError(
                "Missing model files (download the pretrained models into GPT_SoVITS/pretrained_models):\n  "
                + "\n  ".join(missing)
            )
        device = _pick_device(self.device)
        is_half = bool(self.is_half) and device.startswith("cuda")

        print(f"Loading GPT-SoVITS {self.model_version} on {device}{' (fp16)' if is_half else ''}...", file=sys.stderr)
        with _inside(self.root):
            try:
                from TTS_infer_pack.TTS import TTS, TTS_Config
            except (ImportError, OSError) as e:
                raise SetupError(
                    f"Could not import GPT-SoVITS from {self.root} ({e}). "
                    "Run this in the Python environment GPT-SoVITS is installed in."
                ) from e

            class _Config(TTS_Config):
                def save_configs(self, configs_path=None):
                    # Upstream rewrites GPT_SoVITS/configs/tts_infer.yaml on every
                    # load; don't touch the user's WebUI settings.
                    pass

            # Settings must go under "custom"; a flat dict is merged into the
            # defaults and silently ignored.
            config = _Config({"custom": {
                "device": device,
                "is_half": is_half,
                "version": self.model_version,
                "t2s_weights_path": self.gpt_model,
                "vits_weights_path": self.sovits_model,
                "cnhuhbert_base_path": SHARED_MODELS[0],
                "bert_base_path": SHARED_MODELS[1],
            }})
            self._tts = TTS(config)
        return self._tts

    def generate(
        self,
        text,
        output_path=None,
        ref_audio=None,
        ref_text=None,
        language="auto",
        speed=1.0,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        *,
        ref_language=None,
        seed=-1,
        text_split_method="cut1",
    ):
        """
        Synthesize `text` in the voice of the reference clip and write it to output_path.

        Args:
            text: text to speak
            output_path: .wav, .flac or .ogg file (default: output/tts_output.wav)
            ref_audio / ref_text: override the instance defaults ("" clears ref_text)
            language: language of `text`, one of LANGUAGES
            speed: speed factor, 0.5-2.0
            top_k, top_p, temperature: sampling settings of the GPT model
            ref_language: language of ref_text (default: same as language)
            seed: random seed, -1 for a random one
            text_split_method: how GPT-SoVITS splits long text, cut0-cut5

        Returns:
            Absolute path of the written file.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        ref_audio = _absolute(ref_audio) if ref_audio is not None else self.ref_audio
        ref_text = ref_text if ref_text is not None else self.ref_text
        ref_language = ref_language or language
        if not ref_audio:
            raise ValueError("A reference clip is required (ref_audio).")
        for name, value in (("language", language), ("ref_language", ref_language)):
            if value not in LANGUAGES:
                raise ValueError(f"{name} must be one of {', '.join(LANGUAGES)}")
        if text_split_method not in SPLIT_METHODS:
            raise ValueError(f"text_split_method must be one of {', '.join(SPLIT_METHODS)}")
        _check_number("speed", speed, 0.5, 2.0)
        _check_number("top_k", top_k, 1, 100, integer=True)
        _check_number("top_p", top_p, 0, 1, low_inclusive=False)
        _check_number("temperature", temperature, 0, 2, low_inclusive=False)
        _check_number("seed", seed, -1, 2**32 - 1, integer=True)
        if not (ref_text or "").strip() and self.model_version in NEEDS_REF_TEXT:
            raise ValueError(f"{self.model_version} models need the reference transcript (ref_text).")

        output = Path(output_path or DEFAULT_OUTPUT).expanduser().resolve()
        if output == Path(ref_audio):
            raise ValueError("The output path is the reference clip.")
        _check_output_format(output)
        check_reference(ref_audio)

        tts = self._load_model()
        inputs = {
            "text": text,
            "text_lang": language,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text or "",
            "prompt_lang": ref_language,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed,
            "seed": seed,
            "text_split_method": text_split_method,
            "return_fragment": False,
            "streaming_mode": False,
        }
        preview = text if len(text) <= 50 else text[:50] + "..."
        print(f"Generating: {preview}", file=sys.stderr)
        with _inside(self.root):
            # Upstream caches the reference transcript by text only, so a new
            # language for the same transcript would reuse stale phonemes.
            if tts.prompt_cache.get("prompt_lang") != ref_language:
                tts.prompt_cache["prompt_text"] = None
            # Run the generator to the end: on errors upstream yields a second of
            # silence first and only raises when resumed.
            results = list(tts.run(inputs))

        if len(results) != 1:
            raise RuntimeError(f"GPT-SoVITS returned {len(results)} results instead of one audio clip.")
        sample_rate, audio = results[0]
        if not audio.any():
            raise RuntimeError("GPT-SoVITS produced only silence. Check that the language matches the text.")

        _write_audio(output, audio, sample_rate)
        print(f"Saved {output} ({len(audio) / sample_rate:.2f} s)", file=sys.stderr)
        return str(output)

    def batch_generate(self, texts, output_dir="output", prefix="tts", **kwargs):
        """Generate one file per text (prefix_001.wav, ...), stopping at the first failure."""
        output_dir = Path(output_dir).expanduser().resolve()
        return [
            self.generate(text, str(output_dir / f"{prefix}_{i:03d}.wav"), **kwargs)
            for i, text in enumerate(texts, 1)
        ]


def _check_number(name, value, low, high, low_inclusive=True, integer=False):
    """Same limits as the command line, for callers of the Python API."""
    if integer and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    if (value < low if low_inclusive else value <= low) or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _check_output_format(path):
    import soundfile as sf

    if path.suffix[1:].upper() not in sf.available_formats():
        raise ValueError(f"Cannot write {path.suffix or 'files without an extension'}; use .wav, .flac or .ogg.")


def _write_audio(path, audio, sample_rate):
    """Write through a new temporary file so a failed write never damages an existing output."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        sf.write(tmp, audio, sample_rate, format=path.suffix[1:].upper())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# Shortcuts that share one SimpleTTS instance.
_default_tts = None
_default_settings = None


def get_tts(**kwargs):
    """Return the shared SimpleTTS, creating it with these settings on the first call."""
    global _default_tts, _default_settings
    if _default_tts is None:
        _default_tts = SimpleTTS(**kwargs)
        _default_settings = kwargs
    elif kwargs and kwargs != _default_settings:
        raise ValueError("get_tts() was already called with different settings; create a SimpleTTS instead.")
    return _default_tts


def generate_tts(text, output_path=None, **kwargs):
    """Generate one file with the shared instance (pass ref_audio and ref_text here)."""
    return get_tts().generate(text, output_path, **kwargs)


def batch_generate(texts, output_dir="output", prefix="tts", **kwargs):
    """Generate several files with the shared instance."""
    return get_tts().batch_generate(texts, output_dir, prefix, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Simple TTS for GPT-SoVITS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simple_tts.py "Hello world"
    python simple_tts.py "Hello world" output.wav
    python simple_tts.py "Hello world" -r ref.wav -t "Reference text"
    python simple_tts.py "Hello world" --language en --speed 1.2
        """
    )

    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("output", nargs="?", default=None, help="Output wav file path")
    parser.add_argument("-r", "--ref-audio", help="Reference audio file")
    parser.add_argument("-t", "--ref-text", help="Reference audio text content")
    parser.add_argument("-l", "--language", default="auto", help="Language (ko, en, zh, ja, auto)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speed factor (default: 1.0)")
    parser.add_argument("--gpt-model", help="Path to GPT model")
    parser.add_argument("--sovits-model", help="Path to SoVITS model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    if not args.ref_audio or not args.ref_text:
        print("Error: Reference audio (-r) and reference text (-t) are required.")
        print("Example: python simple_tts.py \"Hello\" -r reference.wav -t \"Reference text\"")
        sys.exit(1)

    tts = SimpleTTS(
        gpt_model=args.gpt_model,
        sovits_model=args.sovits_model,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        device="cpu" if args.cpu else "auto",
    )

    tts.generate(
        text=args.text,
        output_path=args.output,
        language=args.language,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
