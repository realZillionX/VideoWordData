# VideoWordData

一个将 HuggingFace 上的大量简单语料数据转为视频数据以训练视频模型的文字渲染/推理能力的脚本库。

## 目录结构

```
VideoWordData/
├── inference/           # 推理任务（prompt无答案，需要模型推理）
│   ├── gsm8k.py              # GSM8K 数学题 (英文)
│   ├── openmath2_gsm8k.py    # OpenMath-2-GSM8K (英文)
│   ├── belle_school_math.py  # BELLE 中文数学题
│   ├── gsm8k_chinese.py      # GSM8K 中文版
│   └── tinystories.py        # TinyStories 故事续写
├── common/              # 共享代码
│   └── video_utils.py        # 视频生成函数
└── fonts/               # 字体文件
    └── DejaVuSansMono.ttf
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
| 字体 | DejaVuSansMono, 28pt |
| 编码格式 | MP4 (mp4v) |

## 渲染机制

- **逐词显示**: 每一帧新增一个单词，模拟逐字打印效果
- **翻页机制**: 当文本超出当前页面可显示区域时，自动翻到下一页继续显示
- **分割线设计**: 第一页上方显示提示/问题/故事开头，下方显示逐词展开的回答/故事续写
- **句末换行**: 每个句子结束时自动换行，提高可读性
- **长度过滤**: 回答/续写部分超过限定字数的样本会被跳过

## 支持的数据集

### 推理任务 (inference/)

JSONL 的 prompt 字段只包含问题/开头，不包含答案/续写。用于训练模型的推理能力。

| 脚本 | 数据集 | 语言 | 数据量 |
|------|--------|------|--------|
| `gsm8k.py` | [gsm8k](https://huggingface.co/datasets/gsm8k) | 英文 | ~7.5K |
| `openmath2_gsm8k.py` | [ai2-adapt-dev/openmath-2-gsm8k](https://huggingface.co/datasets/ai2-adapt-dev/openmath-2-gsm8k) | 英文 | 大规模 |
| `belle_school_math.py` | [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 中文 | ~250K |
| `gsm8k_chinese.py` | [swulling/gsm8k_chinese](https://huggingface.co/datasets/swulling/gsm8k_chinese) | 中文 | ~8.8K |
| `tinystories.py` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英文 | ~2.1M |

## 使用方法

```bash
# 推理任务 - GSM8K
python inference/gsm8k.py --num_samples 1000

# 推理任务 - 中文数学
python inference/belle_school_math.py --num_samples 1000
python inference/gsm8k_chinese.py --num_samples 1000

# 渲染任务 - TinyStories
python rendering/tinystories.py --num_samples 1000

# 指定起始索引（用于分布式处理）
python inference/gsm8k.py --start_idx 5000 --num_samples 1000

# 指定并行工作进程数
python inference/gsm8k.py --num_workers 8
```

## 输出格式

每个脚本会生成：
1. **视频文件**: 保存在 `VIDEO_DIR` 目录下
2. **JSONL 文件**: 包含视频元数据

### Inference JSONL 格式（推理任务）
```json
{
    "video_path": "/path/to/video.mp4",
    "visual_description": "...",
    "speech_description": "",
    "audio_description": "...",
    "prompt": "问题内容（不含答案）"
}
```

### Rendering JSONL 格式（渲染任务）
```json
{
    "video_path": "/path/to/video.mp4",
    "visual_description": "...",
    "speech_description": "",
    "audio_description": "...",
    "prompt": "问题和答案的完整内容"
}
```

## 依赖

```bash
pip install datasets opencv-python numpy pillow tqdm tiktoken
```

## 字体

需要在 `fonts/` 目录下放置 `DejaVuSansMono.ttf` 字体文件。如果字体缺失，将使用系统默认字体。
