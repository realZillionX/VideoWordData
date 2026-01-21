# VideoWordData

一个将 HuggingFace 上的大量简单语料数据转为视频数据以训练视频模型的文字渲染/推理能力的脚本库。

## 目录结构

```
VideoWordData/
├── inference/           # 推理任务（prompt无答案，静默视频，使用 OpenCV 渲染）
│   ├── gsm8k.py              # GSM8K 数学题 (英文)
│   ├── openmath2_gsm8k.py    # OpenMath-2-GSM8K (英文)
│   ├── belle_school_math.py  # BELLE 中文数学题
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── rendering/           # 渲染任务（prompt含答案，静默视频，使用 OpenCV 渲染）
│   └── ...
├── inference_audio/     # 🆕 推理任务（带离线TTS音频和逐句字幕，使用 FFmpeg 极速合成）
│   ├── gsm8k.py              # GSM8K 数学题 (英文)
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── rendering_audio/     # 🆕 渲染任务（带离线TTS音频和逐句字幕，使用 FFmpeg 极速合成）
│   ├── gsm8k.py              # GSM8K 数学题 (英文)
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── common/              # 共享代码
│   ├── video_utils.py        # 静默视频生成函数 (OpenCV)
│   ├── audio_video_utils.py  # 🆕 音频视频生成函数 (FFmpeg + Piper TTS)
│   └── dataset_utils.py      # 数据集加载工具
├── fonts/               # 字体文件
│   ├── DejaVuSansMono.ttf         # 英文等宽字体
│   └── DroidSansFallbackFull.ttf  # 中文字体
├── server_diagnose_ffmpeg.py # 🆕 服务器环境诊断工具
└── test_audio_all.py         # 🆕 音频视频生成测试套件
```


## 视频特性

| 属性 | 静默视频 (Standard) | 音频视频 (Audio) |
|------|--------------------|------------------|
| **生成引擎** | OpenCV (逐帧绘制) | **FFmpeg 直出 (极速)** |
| **分辨率** | 640 × 360 | 640 × 360 |
| **时长** | 10 秒 | **10 秒 (固定)** |
| **帧率** | 19.3 FPS | 19.3 FPS |
| **总帧数** | 193 帧 | 193 帧 |
| **音频** | 无 | **TTS (Piper, 离线)** |
| **字幕** | 逐字显现 | **逐句同步显现** |
| **背景** | 白色 | 白色 |
| **文字** | 黑色 | 黑色 (Subtitle area) |


## 🚀 核心优化 (Latest Updates)

1.  **离线 TTS**: 完全移除 `edge-tts` (需联网)，改用 **`piper-tts`**。
    *   **优点**: 纯离线运行，无需外网，速度快。
    *   **效果**: 语速已调优 (x0.85)，自然且高效。
2.  **极速生成**: 音频视频不再使用 Python 逐帧渲染，而是直接调用 **FFmpeg** 合成。
    *   **速度**: 提升 **5-10倍** (生成一个视频仅需 ~0.2秒)。
    *   **资源**: 充分利用多核 CPU。
3.  **音画同步**: 重构了对齐逻辑。
    *   **机制**: 逐句生成音频 + 逐句锚定字幕。
    *   **效果**: 彻底解决了长文本的音画漂移问题。
4.  **智能填充**:
    *   **TinyStories**: 自动使用高语速填充，保证 10s 视频内容充实。
    *   **GSM8K**: 针对数学文本自动使用保守策略，防止读音过长导致超时。

---

## 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 或者手动安装
pip install datasets opencv-python numpy pillow tqdm tiktoken onnxruntime-gpu piper-tts imageio-ffmpeg
```
*注意：`moviepy` 已被移除，不再需要。*

### 2. 下载数据集 & 模型

```bash
# 下载文本数据集
python download_datasets.py

# TTS 模型会自动下载到 models/piper/ 目录 (首次运行需联网，之后可离线)
```

### 3. 服务器诊断 (推荐)

在跑大规模任务前，建议先运行诊断脚本，确保 FFmpeg 和 TTS 环境正常：

```bash
python server_diagnose_ffmpeg.py
```

### 4. 运行生成任务

#### A. 音频视频任务 (Audio Video) - **推荐**

```bash
# 生成 100 个 GSM8K 视频 (英文)
python inference_audio/gsm8k.py --num_samples 100

# 生成 100 个 TinyStories 视频 (英文)
python inference_audio/tinystories.py --num_samples 100

# 生成 100 个 TinyStories 视频 (中文)
python inference_audio/tinystories_chinese.py --num_samples 100

# 多进程并行 (推荐使用 CPU 核心数)
python inference_audio/tinystories.py --num_workers 32
```

#### B. 静默视频任务 (Silent Video)

```bash
python inference/gsm8k.py --num_samples 100
python inference/tinystories.py --num_samples 100
```

### 5. 测试套件

快速验证所有音频数据集的生成是否正常：

```bash
python test_audio_all.py --num_samples 3
```

---

## 支持的数据集

| 脚本 | 数据集 | 语言 | 数据量 | 音频支持 |
|------|--------|------|--------|----------|
| `gsm8k.py` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 英文 | ~7.5K | ✅ |
| `tinystories.py` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英文 | ~2.1M | ✅ |
| `tinystories_chinese.py` | [adam89/TinyStoriesChinese](https://huggingface.co/datasets/adam89/TinyStoriesChinese) | 中文 | ~2.1M | ✅ |
| `openmath2_gsm8k.py` | [ai2-adapt-dev/openmath-2-gsm8k](https://huggingface.co/datasets/ai2-adapt-dev/openmath-2-gsm8k) | 英文 | 大规模 | ❌ |
| `belle_school_math.py` | [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 中文 | ~250K | ❌ |

---

## 输出格式

```
[base_dir]/
├── gsm8k_audio/                 # 音频视频输出
│   ├── video/                   # .mp4 文件
│   ├── gsm8k_inference_audio_video_data_0.jsonl
│   └── gsm8k_rendering_audio_video_data_0.jsonl
├── tinystories/                 # 静默视频输出
│   ├── video/
│   └── ...
```

### Inference vs Rendering

| 类型 | 目录 | 提示词 (Prompt) | 视频内容 |
|------|------|----------------|----------|
| **Inference** | `inference*/` | 只有问题/开头 | 只有开头部分 |
| **Rendering** | `rendering*/` | 问题 + 完整答案 | 开头 + 逐步显现的答案 |

