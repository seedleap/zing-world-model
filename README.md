<div align="center">

<img src="assets/zing-logo.svg" alt="Zing" height="110">&emsp;&emsp;<img src="assets/loopit-wordmark.svg" alt="Loopit" height="110">

# Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control

<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case0.mp4"><img src="assets/case0.jpg" alt="Play Zing interactive world model case 0" title="Click to play" width="30%"></a>&emsp;<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case1.mp4"><img src="assets/case1.jpg" alt="Play Zing interactive world model case 1" title="Click to play" width="30%"></a>&emsp;<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case2.mp4"><img src="assets/case2.jpg" alt="Play Zing interactive world model case 2" title="Click to play" width="30%"></a>

[English](README.md) | [简体中文](README_zh-CN.md)

<a href="https://zing.loopit.me/"><img src="assets/zing-logo-symbol.svg" alt="" height="16"> Project Page</a> · <a href="https://github.com/seedleap/zing-world-model"><img src="https://github.githubassets.com/favicons/favicon.svg" alt="" height="16"> GitHub</a> · <a href="https://github.com/seedleap/Zing-SGLang"><img src="https://github.githubassets.com/favicons/favicon.svg" alt="" height="16"> SGLang Inference</a> · <a href="https://huggingface.co/seedleap/zing-0.5"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" height="16"> Hugging Face</a> · <a href="https://modelscope.cn/models/seedleap/Zing-0.5"><img src="https://g.alicdn.com/sail-web/maas/2.13.133/favicon/128.ico" alt="" height="16"> ModelScope</a> · <a href="docs/Zing-0.5.pdf"><img src="https://raw.githubusercontent.com/primer/octicons/main/icons/file-text-16.svg" alt="" height="16"> Paper (PDF)</a>

[![arXiv](https://img.shields.io/badge/arXiv-2609.17909-b31b1b.svg)](https://arxiv.org/abs/2609.17909)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)

</div>

Developed by the Seedleap.ai team (涌跃智能), Zing 0.5 is an efficient causal world model built for real-time interaction. It continuously rolls out a visual world from its current state while text prompts and keyboard actions alter the world's semantics, motion, and future state as it unfolds.

Text prompts can initialize the world or change it during a rollout. Continuous W/A/S/D/I/J/K/L keyboard actions steer movement and interaction. A causal KV cache and four-step DMD sampling support responsive, long-horizon world rollouts in a compact single-GPU pipeline.

This inference release represents an interaction session as timestamped prompt and keyboard controls in JSONL. The model consumes those controls causally and records the resulting world rollout as video; interactive applications can map their prompt and keyboard events to the same control schema.

The runtime expects a model directory with this layout:

```text
Zing-0.5/
├── generator/
│   └── model.pt
└── pretrained/
    ├── text_encoder/
    ├── tokenizer/
    └── vae/
```

`generator/model.pt` must directly contain the generator state dict. Parameter names and shapes are matched strictly; wrapped checkpoints, renamed parameter prefixes, adapters, and partial state dicts are not accepted.

## Paper

**[Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control](https://arxiv.org/abs/2609.17909)**

Zing Team · September 15, 2026 · 19 pages

The report describes joint keyboard and online text control, segment-level teacher supervision for block-level generation, and low-cost real-time inference, with quantitative results and a discussion of persistent world state.

## News

- [Sep 16, 2026]: Our Zing-0.5 technical report is now available on [arXiv](https://arxiv.org/abs/2609.17909) ([PDF](docs/Zing-0.5.pdf)), presenting real-time joint action and text control for playable worlds.
- [Sep 14, 2026]: [Zing-SGLang](https://github.com/seedleap/Zing-SGLang) is released, providing SGLang-based inference and real-time WebSocket serving for Zing-0.5.
- [Aug 26, 2026]: Zing-0.5 ranks No. 2 overall on the [WBench leaderboard](https://meituan-longcat.github.io/WBench/#leaderboard) and No. 1 among real-time world models. 🏆
- [Aug 26, 2026]: Zing-0.5 is released. 🎉

## Requirements

The default installation targets NVIDIA CUDA:

- Linux
- A CUDA-capable NVIDIA GPU
- Python 3.11
- Dependencies matching [requirements.txt](requirements.txt)

ROCm inference is also available on compatible AMD GPUs through PyTorch SDPA.
Install PyTorch for ROCm separately before installing
[requirements-rocm.txt](requirements-rocm.txt); do not install the CUDA-only
`flash-attn` package into the ROCm environment.

## Quick Start

### NVIDIA CUDA

Run commands from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
ZING_PYTHON=/path/to/python \
bash run.sh \
  --pretrained-dir /path/to/Zing-0.5/pretrained \
  --checkpoint /path/to/Zing-0.5/generator/model.pt \
  --messages examples/case3_action_t2v.jsonl \
  --output-dir outputs/case3 \
  --seed 0
```

`ZING_PYTHON` defaults to `python3`. The runtime configuration is loaded from `config/zing.yaml`.

### Serve with SGLang

For realtime WebSocket serving, use
[Zing-SGLang](https://github.com/seedleap/Zing-SGLang). Complete its
[one-time setup](https://github.com/seedleap/Zing-SGLang/blob/main/ZING.md),
then run from the Zing-SGLang repository:

```bash
ZING_MODEL_PATH=./models/Zing-0.5-SGLang \
  examples/zing_0_5/launch_server.sh

# In another terminal:
python examples/zing_0_5/client.py \
  --prompt "A first-person walk through a misty pine forest at sunrise" \
  --action w \
  --chunks 4 \
  --output outputs/forest
```

### AMD ROCm

Install a compatible PyTorch build by following the
[official ROCm documentation](https://rocm.docs.amd.com/en/latest/), then
install the remaining dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# Install a compatible PyTorch ROCm build using the official instructions.
python -m pip install -r requirements-rocm.txt
```

Verify that the environment contains a working HIP build before loading the
model:

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.version.hip
properties = torch.cuda.get_device_properties(0)
print(torch.__version__, torch.version.hip, properties.gcnArchName)
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
PY
```

Run the five-frame smoke input with the math SDPA backend first:

```bash
ZING_ATTENTION_BACKEND=sdpa-math \
ZING_PYTHON=/path/to/.venv/bin/python \
bash run.sh \
  --pretrained-dir /path/to/Zing-0.5/pretrained \
  --checkpoint /path/to/Zing-0.5/generator/model.pt \
  --messages examples/rocm_smoke.jsonl \
  --output-dir outputs/rocm-smoke \
  --local-attn-size 33 \
  --sink-size 5 \
  --seed 0
```

The default `auto` selection requires external FlashAttention on CUDA and
fails clearly if that package is absent. On ROCm, `auto` uses the portable SDPA
math backend and chunks long query sequences to bound each estimated FP32
attention-score allocation to 512 MiB. `ZING_ATTENTION_BACKEND=sdpa` opts into
PyTorch's automatic fused-backend selection; validate its output on the exact
GPU, PyTorch build, and production sequence length before relying on it.
The 512 MiB limit covers the score estimate, not other backend intermediates or
model memory, and chunked math is a compatibility path rather than a
performance-equivalent replacement for fused attention.
PyTorch uses the `torch.cuda` API and `cuda` device strings for both CUDA and
ROCm.

ROCm kernel availability and numerical behavior vary by GPU and PyTorch build.
Inspect the generated smoke video; successful package import, device
enumeration, or non-empty output alone is not a correctness result. The
five-frame smoke also does not validate memory or performance at long-rollout
sequence lengths. Performance and numerical results can differ from CUDA
FlashAttention.

## Ready-to-Run World Rollouts

Every JSONL file under `examples/` defines a complete world rollout and can be passed directly to `run.sh`. Image-initialized examples use their bundled reference images under `examples/assets/`.

| Example | Input | Command |
| --- | --- | --- |
| Long action T2V rollout | [case3_action_t2v.jsonl](examples/case3_action_t2v.jsonl) | `bash run.sh ... --messages examples/case3_action_t2v.jsonl --output-dir outputs/case3` |
| Action TI2V rollout | [case4_action_ti2v.jsonl](examples/case4_action_ti2v.jsonl) | `bash run.sh ... --messages examples/case4_action_ti2v.jsonl --output-dir outputs/case4` |
| Prompt-switched T2V rollout | [case6_prompt_switch_t2v.jsonl](examples/case6_prompt_switch_t2v.jsonl) | `bash run.sh ... --messages examples/case6_prompt_switch_t2v.jsonl --output-dir outputs/case6` |
| Combined action + prompt-switch T2V rollout | [case7_action_prompt_switch_t2v.jsonl](examples/case7_action_prompt_switch_t2v.jsonl) | `bash run.sh ... --messages examples/case7_action_prompt_switch_t2v.jsonl --output-dir outputs/case7` |

Replace `...` with the same `--pretrained-dir` and `--checkpoint` arguments shown in Quick Start.

## Input Format

The input is a JSONL interaction timeline with one independent world rollout per line. The `user/text` message defines the initial world, while the `target/video` message defines the rollout and its controls.

```json
{"sample_id":"demo","messages":[{"role":"user","type":"text","content":"A quiet lake at sunrise"},{"role":"target","type":"video","reference_frame_count":0,"output":{"frames":31,"height":480,"width":832},"controls":[]}]}
```

### Text-Prompt Rollout

Set `reference_frame_count` to `0` and omit `uri`. The text prompt initializes the visual world.

### Image-Initialized Rollout

Set `reference_frame_count` to `1` and provide a local image path in `uri`. The image initializes the world state. `output.frames` excludes the reference image, while the recorded rollout includes it as its first frame.

### Keyboard Action Control

Add one `keyboard_direction_frame_interval` control to steer the world as it runs. `actions` must have shape `[N, 8]`, with values in `[0, 1]` ordered as W/A/S/D/I/J/K/L. For a text-initialized rollout, `N = output.frames - 1`; for an image-initialized rollout, `N = output.frames`.

### Prompt-Driven State Changes

Add one `text_prompt_interval` control to change the world's semantics and future evolution during a rollout. It uses `[start, end)` pixel-frame intervals. Intervals must be ordered, gap-free, non-overlapping, and valid after conversion to the latent timeline.

Height and width must be divisible by 32. The reference frame count plus generated frame count must follow the `1 + 4N` temporal structure.

## Sliding-Window Inference

The default sliding-window configuration is `97/9`. It is recommended for offline inference on an NVIDIA H100 or a GPU with at least 80 GB of memory. Both values follow the VAE's `4N+1` temporal structure:

```bash
bash run.sh ... --local-attn-size 97 --sink-size 9
```

For GPUs with less memory or for online real-time inference, use the `33/5` configuration:

```bash
bash run.sh ... --local-attn-size 33 --sink-size 5
```

Use full-history attention with:

```bash
bash run.sh ... --local-attn-size -1 --sink-size 0
```

The sink remains visible throughout the rollout. After a prompt-driven state change, the first block of the new prompt is pinned when it leaves the regular tail window. These options affect only runtime cache behavior and do not change checkpoint parameter names.

## License

Zing is released under the [Apache License 2.0](LICENSE).

---

<div align="center">

<a href="https://loopit.me/"><img src="assets/loopit-wordmark.svg" alt="Loopit" height="72"></a>

### Make Everything Playable

<a href="https://loopit.me/get"><img src="assets/loopit-download-qr.png" alt="Download the Loopit app" width="160"></a>

Scan to download the Loopit app<br>
[Official Website](https://loopit.me/) · [Download Loopit](https://loopit.me/get)

</div>

## Acknowledgements

We thank the teams behind [Wan2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2), [minWM](https://github.com/shengshu-ai/minWM), [WBench](https://github.com/meituan-longcat/WBench), and [LongLive](https://github.com/NVlabs/LongLive) for their open-source work.
