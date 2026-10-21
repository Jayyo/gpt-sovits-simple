# gpt-sovits-simple

A command line tool and small Python API for generating speech with
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS): give it some text and a
3-10 second reference clip, get a WAV in that voice.

```bash
gpt-sovits-simple "이진 탐색은 정렬된 배열에서만 쓸 수 있어요." \
    -r ref.wav -t "Words spoken in ref.wav" -l ko -o out.wav
```

GPT-SoVITS comes with a WebUI and an HTTP API server. I wanted something I
could call from a shell script or a few lines of Python without starting a
server, so this wraps GPT-SoVITS's own inference code (`TTS_infer_pack`) and
adds the checks I kept needing:

- finds the GPT-SoVITS checkout and fails with a readable message when it,
  the models or the Python packages are missing; `--check` reports the
  imports, the device, the main model files and the reference clip
- checks the reference clip length (GPT-SoVITS only accepts 3-10 seconds)
  before loading any model
- picks CUDA with fp16 when a GPU is available, CPU otherwise
- reads the reference clip with soundfile. Since torchaudio 2.9,
  `torchaudio.load` needs TorchCodec, which many GPT-SoVITS environments
  don't have
- doesn't rewrite `GPT_SoVITS/configs/tts_infer.yaml`, which GPT-SoVITS does
  every time a model is loaded

It is a thin wrapper: the voice quality, languages and speed are whatever
GPT-SoVITS gives you.

## Setup

1. Install GPT-SoVITS following [its README](https://github.com/RVC-Boss/GPT-SoVITS)
   and put the pretrained models in `GPT_SoVITS/pretrained_models/`. For the
   default v2 model that is `chinese-hubert-base/`,
   `chinese-roberta-wwm-ext-large/` and `gsv-v2final-pretrained/`.
2. In the same Python environment, install this:

   ```bash
   pip install git+https://github.com/Jayyo/gpt-sovits-simple
   ```

   It only adds `soundfile`; torch and everything else come from the
   GPT-SoVITS environment. Copying `simple_tts.py` into the GPT-SoVITS folder
   and running `python simple_tts.py ...` also works.
3. Check the setup:

   ```bash
   gpt-sovits-simple --check --gpt-sovits-dir ~/GPT-SoVITS -r ref.wav
   ```

The checkout is taken from `--gpt-sovits-dir`, then `$GPT_SOVITS_DIR`, then
the current directory, then the directory `simple_tts.py` is in.

## Command line

```bash
# basic
gpt-sovits-simple "Hello, this is a test." -r ref.wav -t "Reference transcript" -o hello.wav

# Korean voice reading English text
gpt-sovits-simple "Each step throws away half of the range." \
    -r ref_ko.wav -t "참조 음성에서 말한 문장" -l en --ref-language ko -o en.wav

# longer text from a file, split at punctuation, fixed seed
gpt-sovits-simple -f script.txt -r ref.wav -t "..." --text-split-method cut5 --seed 42 -o script.wav

# your own fine-tuned weights, on CPU
gpt-sovits-simple "Hi" -r ref.wav -t "..." \
    --gpt-model GPT_weights_v2/me-e15.ckpt --sovits-model SoVITS_weights_v2/me_e8_s200.pth --device cpu
```

Only the path of the written file goes to stdout; progress, including
GPT-SoVITS's own log lines, goes to stderr. Exit code 2 means a setup or
input problem and 1 a failure inside GPT-SoVITS; `--check` exits with 0 when
everything is in place and 1 otherwise.

| Option | Default | |
|---|---|---|
| `-r, --ref-audio` | | reference clip, 3-10 s; WAV or FLAC |
| `-t, --ref-text` | | transcript of the clip; optional for v1/v2 but much better with it, required for v3/v4 |
| `-l, --language` | `auto` | language of the text: `auto`, `en`, `ko`, `ja`, `zh`, `yue`, `all_ko`, `all_ja`, `all_zh`, `all_yue`, `auto_yue` |
| `--ref-language` | same as `-l` | language of the transcript |
| `-o, --output` | `output/tts_output.wav` | `.wav`, `.flac` or `.ogg` |
| `-f, --text-file` | | read the text from a UTF-8 file |
| `-s, --speed` | `1.0` | 0.5-2.0 |
| `--top-k`, `--top-p`, `--temperature` | `15`, `1.0`, `1.0` | sampling of the GPT stage |
| `--seed` | `-1` | `-1` is random |
| `--text-split-method` | `cut1` | `cut0` none, `cut1` every four sentences, `cut2` every 50 characters, `cut3` at `。`, `cut4` at `.`, `cut5` at any punctuation |
| `--gpt-model`, `--sovits-model` | pretrained | checkpoints to use |
| `--model-version` | `v2` | which pretrained files are the defaults: `v1`, `v2`, `v2Pro`, `v2ProPlus`, `v3`, `v4` |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, ... |
| `--fp32` | off | full precision on GPU |

`ko` treats the text as Korean mixed with English, `all_ko` as Korean only;
the same goes for `ja`/`all_ja` and `zh`/`all_zh`.

## Python

```python
from simple_tts import SimpleTTS

tts = SimpleTTS(
    ref_audio="ref.wav",
    ref_text="Reference transcript",
    gpt_sovits_dir="~/GPT-SoVITS",
)
tts.generate("Hello, this is a test.", "hello.wav", language="en")
tts.batch_generate(["First line.", "Second line."], output_dir="out", prefix="line")
# -> out/line_001.wav, out/line_002.wav
```

The model is loaded on the first `generate()` call and reused afterwards.
Relative paths are resolved against your working directory; GPT-SoVITS runs
from its own folder, and the working directory is restored afterwards.
GPT-SoVITS isn't thread-safe, so use one instance at a time.

## Tested with

GPT-SoVITS at commit `d523079` (July 2026) with the v2 pretrained models,
Python 3.11, torch and torchaudio 2.11 (CUDA 12.8, no TorchCodec), on an
RTX 3090 in fp16 and on the CPU, Korean and English text. The other model
versions go through the same GPT-SoVITS code but I haven't run them with
this wrapper; v2Pro, v3 and v4 also need their extra pretrained files.

The unit tests don't need GPT-SoVITS, torch or any model:

```bash
pip install -e ".[test]"
pytest
```

## Troubleshooting

- **`Resource 'cmudict' not found` from NLTK**: GPT-SoVITS loads NLTK data
  for English even when the text is Korean. Recent NLTK versions also refuse
  a `~/nltk_data` folder that other users can write to. Download `cmudict`
  and `averaged_perceptron_tagger_eng` into a private folder and point
  `NLTK_DATA` at it.
- **CUDA out of memory**: shorter text or a split method like `cut5`, or
  `--device cpu`.
- **Reference audio must be 3-10 seconds long**: trim the clip; GPT-SoVITS
  rejects anything outside that range.

## 한국어 요약

GPT-SoVITS의 추론 코드를 명령줄과 파이썬에서 바로 쓰기 위한 작은
래퍼입니다. 3~10초 참조 음성과 그 문장(`-t`)을 주면 같은 목소리로 텍스트를
읽은 WAV를 만듭니다. GPT-SoVITS가 설치된 파이썬 환경에서
`pip install git+https://github.com/Jayyo/gpt-sovits-simple` 후
`gpt-sovits-simple --check --gpt-sovits-dir <GPT-SoVITS 경로>`로 설정을
확인하세요.

## License

MIT for this wrapper (see `LICENSE`). GPT-SoVITS and its pretrained models
have their own licenses.
