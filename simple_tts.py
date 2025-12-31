#!/usr/bin/env python
"""
Simple TTS Script for GPT-SoVITS
A lightweight wrapper for easy text-to-speech generation.

Usage:
    python simple_tts.py "Text to synthesize" [output.wav]

Examples:
    python simple_tts.py "Hello, this is a test."
    python simple_tts.py "Hello, this is a test." my_audio.wav
    python simple_tts.py "Hello, this is a test." /path/to/output.wav
"""
import os
import sys
import argparse

# Change to script directory for relative imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Monkey patch torchaudio.load to use soundfile instead of torchcodec
# This fixes compatibility issues with certain PyTorch/torchaudio versions
import soundfile as sf
import torch
import torchaudio

def _patched_load(filepath, *args, **kwargs):
    """Load audio using soundfile instead of torchaudio's default backend."""
    audio, sr = sf.read(filepath, dtype='float32')
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    else:
        audio = audio.T
    return torch.from_numpy(audio), sr

torchaudio.load = _patched_load

# Add GPT-SoVITS to path
sys.path.insert(0, "GPT_SoVITS")

from TTS_infer_pack.TTS import TTS, TTS_Config


class SimpleTTS:
    """Simple TTS wrapper for GPT-SoVITS."""

    def __init__(
        self,
        gpt_model: str = None,
        sovits_model: str = None,
        ref_audio: str = None,
        ref_text: str = None,
        device: str = "cuda",
        is_half: bool = True,
    ):
        """
        Initialize TTS model.

        Args:
            gpt_model: Path to GPT (S1) model checkpoint
            sovits_model: Path to SoVITS (S2) model checkpoint
            ref_audio: Path to reference audio file
            ref_text: Text content of the reference audio
            device: Device to use ("cuda" or "cpu")
            is_half: Use half precision (saves VRAM)
        """
        self.gpt_model = gpt_model or self._find_default_gpt()
        self.sovits_model = sovits_model or self._find_default_sovits()
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.device = device
        self.is_half = is_half
        self._tts = None

    def _find_default_gpt(self):
        """Find default pretrained GPT model."""
        default_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        if os.path.exists(default_path):
            return default_path
        raise FileNotFoundError(
            "No GPT model found. Please specify gpt_model path or download pretrained models."
        )

    def _find_default_sovits(self):
        """Find default pretrained SoVITS model."""
        default_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        if os.path.exists(default_path):
            return default_path
        raise FileNotFoundError(
            "No SoVITS model found. Please specify sovits_model path or download pretrained models."
        )

    def _load_model(self):
        """Load TTS model (lazy loading)."""
        if self._tts is not None:
            return self._tts

        print("Loading TTS model...")
        config = TTS_Config({
            "device": self.device,
            "is_half": self.is_half,
            "version": "v2",
            "t2s_weights_path": self.gpt_model,
            "vits_weights_path": self.sovits_model,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        })

        self._tts = TTS(config)
        self._tts.init_t2s_weights(self.gpt_model)
        self._tts.init_vits_weights(self.sovits_model)
        print("TTS model loaded!")

        return self._tts

    def generate(
        self,
        text: str,
        output_path: str = None,
        ref_audio: str = None,
        ref_text: str = None,
        language: str = "auto",
        speed: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate TTS audio from text.

        Args:
            text: Text to synthesize
            output_path: Output wav file path
            ref_audio: Reference audio path (overrides instance default)
            ref_text: Reference text (overrides instance default)
            language: Language code ("ko", "en", "zh", "ja", "auto")
            speed: Speed factor (1.0 = normal, 1.2 = faster, 0.8 = slower)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling

        Returns:
            Path to generated audio file
        """
        tts = self._load_model()

        ref_audio = ref_audio or self.ref_audio
        ref_text = ref_text or self.ref_text

        if not ref_audio or not ref_text:
            raise ValueError(
                "Reference audio and text are required. "
                "Set them in __init__ or pass them to generate()."
            )

        if output_path is None:
            os.makedirs("output", exist_ok=True)
            output_path = "output/tts_output.wav"

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Generating: {text[:50]}..." if len(text) > 50 else f"Generating: {text}")

        result = tts.run({
            "text": text,
            "text_lang": language,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_lang": language,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed,
        })

        for item in result:
            sr, audio = item
            sf.write(output_path, audio, sr)
            duration = len(audio) / sr
            print(f"Saved: {output_path} ({duration:.2f}s)")
            return output_path

    def batch_generate(
        self,
        texts: list,
        output_dir: str = "output",
        prefix: str = "tts",
        **kwargs
    ) -> list:
        """
        Generate multiple TTS files.

        Args:
            texts: List of texts to synthesize
            output_dir: Output directory
            prefix: Filename prefix
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        outputs = []
        for i, text in enumerate(texts, 1):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
            self.generate(text, output_path, **kwargs)
            outputs.append(output_path)

        return outputs


# Convenience functions for simple usage
_default_tts = None

def get_tts(**kwargs) -> SimpleTTS:
    """Get or create default TTS instance."""
    global _default_tts
    if _default_tts is None:
        _default_tts = SimpleTTS(**kwargs)
    return _default_tts


def generate_tts(text: str, output_path: str = None, **kwargs) -> str:
    """Quick function to generate TTS audio."""
    tts = get_tts()
    return tts.generate(text, output_path, **kwargs)


def batch_generate(texts: list, output_dir: str = "output", prefix: str = "tts", **kwargs) -> list:
    """Quick function to generate multiple TTS files."""
    tts = get_tts()
    return tts.batch_generate(texts, output_dir, prefix, **kwargs)


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
        device="cpu" if args.cpu else "cuda",
    )

    tts.generate(
        text=args.text,
        output_path=args.output,
        language=args.language,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
