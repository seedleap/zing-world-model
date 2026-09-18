<div align="center">

<img src="assets/zing-logo.svg" alt="Zing" height="110">&emsp;&emsp;<img src="assets/loopit-wordmark.svg" alt="Loopit" height="110">

# Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control

<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case0.mp4"><img src="assets/case0.jpg" alt="播放 Zing 实时交互世界模型案例 0" title="点击播放" width="30%"></a>&emsp;<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case1.mp4"><img src="assets/case1.jpg" alt="播放 Zing 实时交互世界模型案例 1" title="点击播放" width="30%"></a>&emsp;<a href="https://cdn.jsdelivr.net/gh/seedleap/zing-world-model@main/assets/case2.mp4"><img src="assets/case2.jpg" alt="播放 Zing 实时交互世界模型案例 2" title="点击播放" width="30%"></a>

[English](README.md) | [简体中文](README_zh-CN.md)

<a href="https://zing.loopit.me/"><img src="assets/zing-logo-symbol.svg" alt="" height="16"> 项目主页</a> · <a href="https://github.com/seedleap/zing-world-model"><img src="https://github.githubassets.com/favicons/favicon.svg" alt="" height="16"> GitHub</a> · <a href="https://github.com/seedleap/Zing-SGLang"><img src="https://github.githubassets.com/favicons/favicon.svg" alt="" height="16"> SGLang 推理</a> · <a href="https://huggingface.co/seedleap/zing-0.5"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" height="16"> Hugging Face</a> · <a href="https://modelscope.cn/models/seedleap/Zing-0.5"><img src="https://g.alicdn.com/sail-web/maas/2.13.133/favicon/128.ico" alt="" height="16"> ModelScope</a> · <a href="docs/Zing-0.5.pdf"><img src="https://raw.githubusercontent.com/primer/octicons/main/icons/file-text-16.svg" alt="" height="16"> 论文（PDF）</a>

[![arXiv](https://img.shields.io/badge/arXiv-2609.17909-b31b1b.svg)](https://arxiv.org/abs/2609.17909)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)

</div>

Zing 0.5 是由 Seedleap.ai（涌跃智能）团队研发的高效因果世界模型，专为实时交互而设计。模型能够从当前状态出发，持续预测并生成后续画面；在推演过程中，用户可以随时输入文本提示词或键盘操作，实时改变场景内容、运动方式和后续演化。

文本提示词既可以创建初始世界，也可以在运行过程中切换场景。连续输入 W/A/S/D/I/J/K/L 键可控制移动与交互。借助因果 KV 缓存和四步 DMD 采样，模型能够在单张 GPU 上实现低延迟、长时序的世界推演。

本仓库使用 JSONL 描述交互时间线，其中包含按时间排列的文本提示词和键盘操作。模型会按照因果顺序处理这些输入，并将推演过程保存为视频。接入实时应用时，只需将文本输入和键盘事件转换为相同的控制格式。

模型文件应按以下目录结构放置，其中 `generator/` 与 `pretrained/` 位于同一级目录：

```text
Zing-0.5/
├── generator/
│   └── model.pt
└── pretrained/
    ├── text_encoder/
    ├── tokenizer/
    └── vae/
```

`generator/model.pt` 中应直接存放生成器的参数字典（state dict），文件中的参数名和形状必须与模型完全一致。如果文件还包含训练状态等外层字段、参数名前缀已被修改、带有额外的 Adapter 参数，或缺少模型参数，加载都会失败。

## 论文

**[Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control](https://arxiv.org/abs/2609.17909)**

Zing Team · 2026 年 9 月 15 日 · 19 页

论文介绍了键盘与在线文本联合控制、片段级教师对分块生成的监督，以及低成本实时推理，并报告定量评测结果、讨论世界状态持久性问题。

## 新闻

- [2026 年 9 月 16 日]：Zing-0.5 技术报告已发布于 [arXiv](https://arxiv.org/abs/2609.17909)（[PDF](docs/Zing-0.5.pdf)），介绍面向可玩世界的实时动作与文本联合控制。
- [2026 年 9 月 14 日]：[Zing-SGLang](https://github.com/seedleap/Zing-SGLang) 已发布，提供基于 SGLang 的 Zing-0.5 推理与实时 WebSocket 服务。
- [2026 年 8 月 26 日]：Zing-0.5 位列 [WBench 总榜](https://meituan-longcat.github.io/WBench/#leaderboard)第二，并在实时世界模型中排名第一。🏆
- [2026 年 8 月 26 日]：Zing-0.5 正式发布。🎉

## 环境要求

默认安装方式面向 NVIDIA CUDA：

- Linux
- 支持 CUDA 的 NVIDIA GPU
- Python 3.11
- 与 [requirements.txt](requirements.txt) 一致的依赖

兼容的 AMD GPU 也可以通过 PyTorch SDPA 运行 ROCm 推理。请先单独安装
ROCm 版 PyTorch，再安装 [requirements-rocm.txt](requirements-rocm.txt)；
不要在 ROCm 环境中安装仅支持 CUDA 的 `flash-attn`。

## 快速开始

### NVIDIA CUDA

请在仓库根目录执行：

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

如果未设置 `ZING_PYTHON`，脚本将使用 `python3`。推理配置统一从 `config/zing.yaml` 读取。

### 使用 SGLang 启动

如需通过实时 WebSocket 服务运行模型，请使用
[Zing-SGLang](https://github.com/seedleap/Zing-SGLang)。先完成其中的
[一次性安装与模型下载](https://github.com/seedleap/Zing-SGLang/blob/main/ZING.md)，
再在 Zing-SGLang 仓库中执行：

```bash
ZING_MODEL_PATH=./models/Zing-0.5-SGLang \
  examples/zing_0_5/launch_server.sh

# 在另一个终端中执行：
python examples/zing_0_5/client.py \
  --prompt "A first-person walk through a misty pine forest at sunrise" \
  --action w \
  --chunks 4 \
  --output outputs/forest
```

### AMD ROCm

按照 [ROCm 官方文档](https://rocm.docs.amd.com/en/latest/) 安装兼容的
PyTorch 版本，然后安装其余依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# 按官方说明安装兼容的 ROCm 版 PyTorch。
python -m pip install -r requirements-rocm.txt
```

加载模型前，先确认当前环境包含可以实际执行计算的 HIP build：

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

首次运行时，使用数学 SDPA backend 执行五帧 smoke：

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

默认的 `auto` 模式在 CUDA 上要求 external FlashAttention；缺少该包时会
明确报错。在 ROCm 上，`auto` 使用可移植的 SDPA 数学 backend，并对较长的
query 序列分块，将每次调用估算的 FP32 attention score 控制在 512 MiB
以内。`ZING_ATTENTION_BACKEND=sdpa` 会让 PyTorch 自动选择 fused backend；
依赖它之前，必须在实际 GPU、PyTorch build 和生产序列长度上验证输出。
512 MiB 只限制 score 估算，不包括 backend 的其他中间结果或模型显存；
chunked math 是兼容路径，并不是 fused attention 的等性能替代。
PyTorch 在 CUDA 和 ROCm 上都使用 `torch.cuda` API 和 `cuda` 设备字符串。

ROCm kernel 的可用性和数值行为会随 GPU 与 PyTorch build 变化。请检查实际
生成的 smoke 视频；仅仅成功导入包、枚举设备或生成非空文件不能证明结果
正确。五帧 smoke 也不能验证长 rollout 序列长度下的内存和性能。性能和
数值结果可能与 CUDA FlashAttention 不同。

## 开箱即用的推演示例

`examples/` 目录中的每个 JSONL 文件都描述了一次完整推演，可直接传给 `run.sh`。需要图像输入的示例已经在 `examples/assets/` 中附带对应的参考图。

| 示例 | 输入 | 命令 |
| --- | --- | --- |
| 长时序动作控制（T2V） | [case3_action_t2v.jsonl](examples/case3_action_t2v.jsonl) | `bash run.sh ... --messages examples/case3_action_t2v.jsonl --output-dir outputs/case3` |
| 图像与动作联合控制（TI2V） | [case4_action_ti2v.jsonl](examples/case4_action_ti2v.jsonl) | `bash run.sh ... --messages examples/case4_action_ti2v.jsonl --output-dir outputs/case4` |
| 运行中切换提示词（T2V） | [case6_prompt_switch_t2v.jsonl](examples/case6_prompt_switch_t2v.jsonl) | `bash run.sh ... --messages examples/case6_prompt_switch_t2v.jsonl --output-dir outputs/case6` |
| 按键与提示词切换联合控制（T2V） | [case7_action_prompt_switch_t2v.jsonl](examples/case7_action_prompt_switch_t2v.jsonl) | `bash run.sh ... --messages examples/case7_action_prompt_switch_t2v.jsonl --output-dir outputs/case7` |

请将命令中的 `...` 替换为“快速开始”所示的 `--pretrained-dir` 和 `--checkpoint` 参数。

## 输入格式

输入采用 JSONL 格式，每行描述一次独立的世界推演。`user/text` 消息用于定义初始世界，`target/video` 消息用于指定生成范围和交互控制。

```json
{"sample_id":"demo","messages":[{"role":"user","type":"text","content":"日出时分的宁静湖面"},{"role":"target","type":"video","reference_frame_count":0,"output":{"frames":31,"height":480,"width":832},"controls":[]}]}
```

### 文本初始化

将 `reference_frame_count` 设为 `0`，并省略 `uri`。模型将根据文本提示词生成初始世界。

### 图像初始化

将 `reference_frame_count` 设为 `1`，并通过 `uri` 指定本地图片路径。该图片将作为世界的初始状态。`output.frames` 仅表示待生成的帧数，不包含参考图；最终视频会将参考图作为第一帧。

### 键盘操作控制

添加一个 `keyboard_direction_frame_interval` 控制项，即可在推演过程中持续输入键盘操作。`actions` 的形状必须为 `[N, 8]`，取值范围为 `[0, 1]`，八个维度依次对应 W/A/S/D/I/J/K/L。使用文本初始化时，`N = output.frames - 1`；使用图像初始化时，`N = output.frames`。

### 运行中切换文本提示词

添加一个 `text_prompt_interval` 控制项，可以在推演过程中改变场景内容及后续演化。时间范围使用像素帧区间 `[start, end)` 表示。各区间必须按顺序排列、首尾相接且互不重叠；映射到潜空间时间轴后，每个区间仍须至少包含一个有效时间步。

输出宽高必须是 32 的倍数。参考帧数与生成帧数之和必须满足 `1 + 4N` 的时间结构。

## 滑动窗口推理

默认滑动窗口配置为 `97/9`，推荐用于 NVIDIA H100 或显存不低于 80GB 的 GPU 进行离线推理。两个数值均满足 VAE 的 `4N+1` 时间结构：

```bash
bash run.sh ... --local-attn-size 97 --sink-size 9
```

显存较小或需要在线实时推理时，推荐使用 `33/5` 配置：

```bash
bash run.sh ... --local-attn-size 33 --sink-size 5
```

如需让模型关注完整历史上下文，可使用：

```bash
bash run.sh ... --local-attn-size -1 --sink-size 0
```

`sink_size` 指定的前缀帧在整个推演过程中始终参与注意力计算。切换文本提示词后，新提示词对应的首个分块即使滑出常规尾部窗口，也会继续固定保留。这两个参数只影响运行时的 KV 缓存，不会改变模型权重中的参数名称。

## 开源协议

Zing 基于 [Apache License 2.0](LICENSE) 开源。

---

<div align="center">

<a href="https://loopit.me/"><img src="assets/loopit-wordmark.svg" alt="Loopit" height="72"></a>

### 让一切都可玩

<a href="https://loopit.me/get"><img src="assets/loopit-download-qr.png" alt="下载 Loopit 应用" width="160"></a>

扫描二维码下载 Loopit 应用<br>
[官方网站](https://loopit.me/) · [下载 Loopit](https://loopit.me/get)

</div>

## 致谢

感谢 [Wan2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2)、[minWM](https://github.com/shengshu-ai/minWM)、[WBench](https://github.com/meituan-longcat/WBench) 和 [LongLive](https://github.com/NVlabs/LongLive) 团队的开源工作。
