# VideoWordData

一个将 HuggingFace 上的大量简单语料数据转为视频数据以训练视频模型的文字渲染/推理能力的脚本库。

## 目录结构

```
VideoWordData/
├── inference/           # 推理任务（prompt无答案，静默视频）
│   ├── gsm8k.py              # GSM8K 数学题 (英文)
│   ├── openmath2_gsm8k.py    # OpenMath-2-GSM8K (英文)
│   ├── belle_school_math.py  # BELLE 中文数学题
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── rendering/           # 渲染任务（prompt含答案，静默视频）
│   └── ...
├── inference_audio/     # 🆕 推理任务（带TTS音频和逐句字幕）
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── rendering_audio/     # 🆕 渲染任务（带TTS音频和逐句字幕）
│   ├── tinystories.py        # TinyStories 故事续写 (英文)
│   └── tinystories_chinese.py # TinyStories 中文版
├── common/              # 共享代码
│   ├── video_utils.py        # 静默视频生成函数
│   ├── audio_video_utils.py  # 🆕 音频视频生成函数（逐句字幕）
│   └── dataset_utils.py      # 数据集加载工具
└── fonts/               # 字体文件
    ├── DejaVuSansMono.ttf         # 英文等宽字体
    └── DroidSansFallbackFull.ttf  # 中文字体
```


## 视频特性

| 属性 | 值 |
|------|-----|
| 分辨率 | 640 × 360 (360P) |
| 时长 | 10 秒 |
| 总帧数 | 193 帧 |
| 帧率 | 19.3 FPS |
| 背景色 | 白色 (#FFFFFF) |
| 文字颜色 | 黑色 (#000000) |
| 字体 | DejaVuSansMono/DroidSansFallback, 28pt |
| 编码格式 | MP4 (mp4v) |

## 渲染机制

- **逐词显示**: 每一帧新增一个单词，模拟逐字打印效果
- **翻页机制**: 当文本超出当前页面可显示区域时，自动翻到下一页继续显示
- **分割线设计**: 第一页上方显示提示/问题/故事开头，下方显示逐词展开的回答/故事续写
- **句末换行**: 每个句子结束时自动换行，提高可读性
- **长度过滤**: 回答/续写部分超过限定字数的样本会被跳过

## 支持的数据集

| 脚本 | 数据集 | 语言 | 数据量 |
|------|--------|------|--------|
| `gsm8k.py` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 英文 | ~7.5K |
| `openmath2_gsm8k.py` | [ai2-adapt-dev/openmath-2-gsm8k](https://huggingface.co/datasets/ai2-adapt-dev/openmath-2-gsm8k) | 英文 | 大规模 |
| `belle_school_math.py` | [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 中文 | ~250K |
| `tinystories.py` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英文 | ~2.1M |
| `tinystories_chinese.py` | [adam89/TinyStoriesChinese](https://huggingface.co/datasets/adam89/TinyStoriesChinese) | 中文 | ~2.1M |

## Inference vs Rendering

| 类型 | 目录 | JSONL prompt 内容 | 用途 |
|------|------|------------------|------|
| **inference** | `inference/` | 只有问题，不含答案 | 训练推理能力 |
| **rendering** | `rendering/` | 问题 + 答案都包含 | 训练渲染能力 |

两种任务使用相同的数据集和视频，唯一区别是 JSONL 文件中 `prompt` 字段是否包含答案。

## 使用方法

```bash
# 推理任务
python inference/gsm8k.py --num_samples 1000

# 渲染任务
python rendering/gsm8k.py --num_samples 1000

# 指定起始索引（用于分布式处理）
python inference/gsm8k.py --start_idx 5000 --num_samples 1000

# 指定并行工作进程数
python inference/gsm8k.py --num_workers 8

# 指定输出目录 (默认: /inspire/hdd/project/embodied-multimodality/public/textcentric)
python inference/gsm8k.py --base_dir /your/custom/path
```

## 输出文件命名

- **Video**: `[base_dir]/[dataset]/video/[dataset]_[index].mp4`
- **Inference JSONL**: `[base_dir]/[dataset]/[dataset]_inference_video_data_[start_idx].jsonl`
- **Rendering JSONL**: `[base_dir]/[dataset]/[dataset]_rendering_video_data_[start_idx].jsonl`

### Inference JSONL（推理任务）
```json
{
    "video_path": "/path/to/video.mp4",
    "prompt": "Question: ... (不含答案)"
}
```

### Rendering JSONL（渲染任务）
```json
{
    "video_path": "/path/to/video.mp4",
    "prompt": "Question: ... Answer: ... (包含答案)"
}
```

## 快速开始

### 第一步：下载数据集

使用 `download_datasets.py` 脚本将数据集下载到本地，避免每次运行时重复下载。

```bash
# 下载所有数据集到默认目录
python download_datasets.py

# 下载到自定义目录
python download_datasets.py --base_dir /your/custom/path

# 只下载某个数据集
python download_datasets.py --dataset gsm8k
python download_datasets.py --dataset tinystories

# 查看所有可用数据集
python download_datasets.py --list

# 强制重新下载（即使本地已存在）
python download_datasets.py --force
```

**下载目录结构**：
```
/inspire/hdd/project/embodied-multimodality/public/
├── gsm8k/dataset/           # GSM8K 数据
├── openmath2_gsm8k/dataset/ # OpenMath-2-GSM8K 数据
├── belle_school_math/dataset/ # BELLE 中文数学
├── gsm8k_chinese/dataset/   # GSM8K 中文版
└── tinystories/dataset/     # TinyStories 故事
```

### 第二步：生成视频

下载完成后，运行脚本会自动从本地加载数据集：

```bash
# 单机运行
python inference/gsm8k.py --num_samples 1000
python rendering/tinystories.py --num_samples 5000
```

---

---
## 🚀 服务器完整部署工作流 (Server Workflow)

这是在服务器集群上从零开始部署并运行大规模生成任务的标准流程。

### 第一步：数据准备 (Data Preparation)

在任何计算节点运行之前，先在主节点（或拥有公网权限的节点）将所需数据集下载到共享存储中。

```bash
# 1. 确认已安装所有依赖
pip install -r requirements.txt  # 如果有

# 2. 下载所有数据集到默认共享路径
# 默认路径: /inspire/hdd/project/embodied-multimodality/public/
python download_datasets.py

# 或者下载到你指定的共享目录
python download_datasets.py --base_dir /path/to/shared/storage
```

### 第二步：单机并行调试 (Local Debugging)

在提交大规模集群作业前，先在单个节点上进行测试，确保代码和环境正常，且并行生成没有问题。

1. **小样本单进程测试**（验证代码逻辑）：
    ```bash
    # 生成 5 个样本，检查 output/video 目录下是否有视频，是否能播放
    python inference/belle_school_math.py --num_samples 5 --base_dir ./debug_output
    ```

2. **单机多核并行测试**（验证 CPU 跑满和多进程稳定性）：
    ```bash
    # 使用 16 个进程生成 1000 个样本
    python inference/belle_school_math.py --num_samples 1000 --num_workers 16 --base_dir ./debug_output
    ```
    * 观察 `htop`，确认 16 个 CPU 核心都被占用。
    * 检查生成的 JSONL 文件是否包含 1000 条数据。

### 第三步：集群作业模拟 (Slurm Dry-Run)

在正式提交任务前，预览 Slurm 脚本生成的作业数组配置，防止参数配置错误。

```bash
# 预览将要提交的任务（不会真正运行）
./submit_jobs.sh --dry-run all

# 预期输出示例：
# [DRY-RUN] Would submit job for gsm8k (inference)
# Nodes: 1, Samples per node: 10000
# Node 0 handles indices: 0 to 10000
```

### 第四步：大规模正式运行 (Production Run)

确认无误后，正式提交作业到 Slurm 集群。建议优先处理较小的数据集，最后处理 TinyStories。

```bash
# 1. 先提交一个小数据集（如 gsm8k）试水
./submit_jobs.sh gsm8k

# 2. 检查作业状态
squeue -u $USER

# 3. 确认日志正常（日志通常在 logs/ 目录下）
tail -f logs/gsm8k_inference_*.out

# 4. 如果一切顺利，提交所有任务
./submit_jobs.sh
```

---

## 依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
# 核心依赖
pip install datasets opencv-python numpy pillow tqdm tiktoken

# 音频视频功能 (可选)
pip install edge-tts moviepy
```

## 字体

需要在 `fonts/` 目录下放置 `DejaVuSansMono.ttf` 字体文件。

---

## 🆕 音频视频功能 (Audio Video Feature)

带 TTS 语音朗读和**逐句同步字幕**的视频生成功能，使用 Microsoft Edge TTS (免费，高质量)。

### 视频特性

| 属性 | 值 |
|------|-----|
| 分辨率 | 640 × 360 (360P) |
| 时长 | 动态（由 TTS 音频决定，通常 5-30 秒）|
| 帧率 | 24 FPS |
| 上方区域 | 白色背景 + 黑色 prompt 文字 |
| 字幕区域 | **视频底部**，半透明黑色背景 + 白色大字体 |
| 字幕同步 | **逐句显示**（每句话与音频同步出现）|
| 字幕字体 | 36pt（清晰可读）|

### 使用方法

```bash
# 推理任务（prompt 不含续写）
python inference_audio/tinystories.py --num_samples 100
python inference_audio/tinystories_chinese.py --num_samples 100

# 渲染任务（prompt 包含完整文本）
python rendering_audio/tinystories.py --num_samples 100
python rendering_audio/tinystories_chinese.py --num_samples 100

# 指定输出目录
python inference_audio/tinystories.py --base_dir ./output --num_samples 10
```

### Inference vs Rendering (Audio)

| 类型 | 目录 | JSONL prompt 内容 | 用途 |
|------|------|------------------|------|
| **inference_audio** | `inference_audio/` | 只有开头，不含续写 | 训练推理能力 |
| **rendering_audio** | `rendering_audio/` | 开头 + 续写都包含 | 训练渲染能力 |

### 输出格式

```
[base_dir]/
├── tinystories_audio/           # 英文带音频字幕视频
│   ├── video/
│   ├── tinystories_inference_audio_video_data_0.jsonl
│   └── tinystories_rendering_audio_video_data_0.jsonl
└── tinystories_chinese_audio/   # 中文带音频字幕视频
    ├── video/
    ├── tinystories_chinese_inference_audio_video_data_0.jsonl
    └── tinystories_chinese_rendering_audio_video_data_0.jsonl
```

### TTS 支持语言

| 语言 | Voice ID |
|------|----------|
| 英文 | en-US-AriaNeural |
| 中文 | zh-CN-XiaoxiaoNeural |

