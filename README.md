# VideoWordData

一个将 HuggingFace 上的大量简单语料数据转为视频数据以训练视频模型的文字渲染/推理能力的脚本库。

## 视频特性

| 属性 | 值 |
|------|-----|
| 分辨率 | 640 × 360 (360P) |
| 时长 | 10 秒 |
| 总帧数 | 193 帧 |
| 帧率 | 19.3 FPS |
| 背景色 | 白色 (#FFFFFF) |
| 文字颜色 | 黑色 (#000000) |
| 字体 | DejaVuSans, 20pt |
| 编码格式 | MP4 (mp4v) |

## 渲染机制

- **逐词显示**: 每一帧新增一个单词，模拟逐字打印效果
- **翻页机制**: 当文本超出当前页面可显示区域时，自动翻到下一页继续显示
- **分割线设计**: 第一页上方显示提示/问题/故事开头，下方显示逐词展开的回答/故事续写
- **长度过滤**: 回答/续写部分超过 192 词的样本会被跳过（因为只有 192 帧用于显示回答）

## 支持的数据集

| 脚本 | 数据集 | 格式说明 |
|------|--------|----------|
| `gsm8k.py` | [gsm8k](https://huggingface.co/datasets/gsm8k) | 数学题问答格式 (question → answer) |
| `openmath2_gsm8k.py` | [ai2-adapt-dev/openmath-2-gsm8k](https://huggingface.co/datasets/ai2-adapt-dev/openmath-2-gsm8k) | 消息格式 (user → assistant) |
| `tinystories.py` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 故事格式 (前2句 → 后续) |
| `cosmopedia_stories.py` | [khairi/cosmopedia-stories-young-children](https://huggingface.co/datasets/khairi/cosmopedia-stories-young-children) | 故事格式 (前2句 → 后续) |

## 使用方法

```bash
# 基本用法
python gsm8k.py

# 指定处理样本数量
python gsm8k.py --num_samples 1000

# 指定起始索引（用于分布式处理）
python gsm8k.py --start_idx 5000 --num_samples 1000

# 指定并行工作进程数
python gsm8k.py --num_workers 8
```

## 输出格式

每个脚本会生成：
1. **视频文件**: 保存在 `VIDEO_DIR` 目录下，命名格式为 `{dataset}_{index:06d}.mp4`
2. **JSONL 文件**: 包含视频元数据，格式如下：

```json
{
    "video_path": "/path/to/video.mp4",
    "visual_description": "视频的视觉描述",
    "speech_description": "",
    "audio_description": "Silent video with no audio content.",
    "prompt": "用于训练的提示文本"
}
```

## 依赖

```bash
pip install datasets opencv-python numpy pillow tqdm tiktoken
```

## 字体

需要在 `fonts/` 目录下放置 `DejaVuSans.ttf` 字体文件。如果字体缺失，将使用系统默认字体。
