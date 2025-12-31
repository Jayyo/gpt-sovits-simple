# GPT-SoVITS Simple

A lightweight wrapper for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - easy text-to-speech with voice cloning.

## Features

- Simple CLI and Python API
- Automatic torchaudio compatibility fix (soundfile patch)
- Batch generation support
- Minimal configuration required

## Requirements

- Python 3.10+
- NVIDIA GPU with **6GB+ VRAM** (or CPU, but slow)
- CUDA 12.4+ recommended

## Installation

### 1. Clone GPT-SoVITS

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install PyTorch (IMPORTANT: correct CUDA version)

```bash
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install soundfile  # Required for audio loading fix
```

### 5. Download pretrained models

Download from [Hugging Face](https://huggingface.co/lj1995/GPT-SoVITS) and place in:
```
GPT_SoVITS/pretrained_models/
├── chinese-hubert-base/
├── chinese-roberta-wwm-ext-large/
└── gsv-v2final-pretrained/
    ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt  # GPT model
    └── s2G2333k.pth  # SoVITS model
```

### 6. Copy simple_tts.py

```bash
# Copy to GPT-SoVITS root directory
cp simple_tts.py /path/to/GPT-SoVITS/
```

## Usage

### CLI

```bash
cd GPT-SoVITS

# Basic usage (reference audio required)
python simple_tts.py "Hello, this is a test." -r reference.wav -t "Reference text content"

# With output path
python simple_tts.py "Hello world" output.wav -r ref.wav -t "Reference text"

# Specify language and speed
python simple_tts.py "안녕하세요" -r ref.wav -t "참조 텍스트" -l ko -s 1.2

# Use CPU
python simple_tts.py "Hello" -r ref.wav -t "Reference" --cpu
```

### Python API

```python
from simple_tts import SimpleTTS

# Initialize
tts = SimpleTTS(
    ref_audio="reference.wav",
    ref_text="Reference audio text content",
)

# Generate single file
tts.generate("Hello, this is a test.", "output.wav")

# Generate multiple files
texts = ["First sentence.", "Second sentence.", "Third sentence."]
tts.batch_generate(texts, output_dir="output/", prefix="audio")
# -> output/audio_001.wav, audio_002.wav, audio_003.wav

# With custom models
tts = SimpleTTS(
    gpt_model="path/to/gpt.ckpt",
    sovits_model="path/to/sovits.pth",
    ref_audio="reference.wav",
    ref_text="Reference text",
)
```

### Quick functions

```python
from simple_tts import generate_tts, batch_generate

# These use a shared TTS instance
generate_tts("Hello world", "output.wav", ref_audio="ref.wav", ref_text="Ref text")
batch_generate(["Text 1", "Text 2"], output_dir="output/")
```

## Troubleshooting

### CUDA architecture error (sm_80 vs sm_70)

```
FATAL: this function is for sm_80, but was built for sm_70
```

**Solution:** Reinstall PyTorch with correct CUDA version:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### torchaudio loading error

This wrapper already includes a fix (soundfile patch). If you still have issues:

```bash
pip install soundfile
```

### CUDA out of memory

- Use `is_half=True` (default)
- Use shorter text segments
- Close other GPU applications
- Check with `nvidia-smi`

### Model not found

Make sure pretrained models are downloaded to:
```
GPT_SoVITS/pretrained_models/
```

## Supported Languages

| Code | Language |
|------|----------|
| `ko` | Korean |
| `en` | English |
| `zh` | Chinese |
| `ja` | Japanese |
| `auto` | Auto-detect |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `auto` | Language code |
| `speed` | `1.0` | Speed factor (0.5-2.0) |
| `top_k` | `15` | Top-k sampling |
| `top_p` | `1.0` | Top-p sampling |
| `temperature` | `1.0` | Sampling temperature |

## License

MIT License - see [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for original project license.

## Credits

- Original project: [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- This wrapper by: [@Jayyo](https://github.com/Jayyo)
