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
├── rendering/           # 渲染任务（prompt含答案，训练渲染能力）
│   ├── gsm8k.py
│   ├── openmath2_gsm8k.py
│   ├── belle_school_math.py
│   ├── gsm8k_chinese.py
│   └── tinystories.py
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

| 脚本 | 数据集 | 语言 | 数据量 |
|------|--------|------|--------|
| `gsm8k.py` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 英文 | ~7.5K |
| `openmath2_gsm8k.py` | [ai2-adapt-dev/openmath-2-gsm8k](https://huggingface.co/datasets/ai2-adapt-dev/openmath-2-gsm8k) | 英文 | 大规模 |
| `belle_school_math.py` | [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | 中文 | ~250K |
| `gsm8k_chinese.py` | [swulling/gsm8k_chinese](https://huggingface.co/datasets/swulling/gsm8k_chinese) | 中文 | ~8.8K |
| `tinystories.py` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 英文 | ~2.1M |

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

## Slurm 集群使用

提供两个脚本用于在 Slurm 集群上进行多节点并行处理。

### 脚本说明

| 脚本 | 用途 |
|------|------|
| `submit_jobs.sh` | 主控脚本，用于提交作业 |
| `slurm_job.sh` | 单节点作业脚本，被 submit_jobs.sh 调用 |

### 基本用法

```bash
# 提交所有数据集的所有任务（inference + rendering）
./submit_jobs.sh

# 只提交某个数据集
./submit_jobs.sh gsm8k
./submit_jobs.sh tinystories

# 只提交某个数据集的某个任务类型
./submit_jobs.sh gsm8k inference
./submit_jobs.sh tinystories rendering

# 提交所有数据集的 rendering 任务
./submit_jobs.sh all rendering
```

### 预览和管理

```bash
# 预览命令（不实际提交）
./submit_jobs.sh --dry-run
./submit_jobs.sh --dry-run gsm8k

# 查看作业状态
./submit_jobs.sh --status

# 取消所有作业
./submit_jobs.sh --cancel
```

### 自定义配置

通过环境变量自定义运行参数：

```bash
# 使用 40 个节点，每节点处理 20000 个样本
NUM_NODES=40 SAMPLES_PER_NODE=20000 ./submit_jobs.sh tinystories

# 使用不同的 Slurm 分区
PARTITION=gpu ./submit_jobs.sh gsm8k

# 使用 64 核 CPU
CPUS_PER_NODE=64 ./submit_jobs.sh
```

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NUM_NODES` | 20 | 最大节点数 |
| `CPUS_PER_NODE` | 120 | 每节点 CPU 数 |
| `SAMPLES_PER_NODE` | 10000 | 每节点处理样本数 |
| `PARTITION` | cpu | Slurm 分区名 |
| `TIME_LIMIT` | 48:00:00 | 作业时间限制 |
| `BASE_DIR` | `/inspire/.../textcentric` | 输出目录 |

### 工作原理

1. `submit_jobs.sh` 根据数据集大小自动计算需要多少节点
2. 使用 `sbatch --array` 提交作业数组
3. 每个节点处理数据集的不同分片（通过 `--start_idx` 区分）
4. 所有节点并行工作，互不干扰

**示例**：处理 250K 样本的 belle_school_math
- 每节点 10000 样本 → 需要 25 个节点
- 节点 0: `--start_idx 0`
- 节点 1: `--start_idx 10000`
- 节点 2: `--start_idx 20000`
- ...

---

## 依赖

```bash
pip install datasets opencv-python numpy pillow tqdm tiktoken
```

## 字体

需要在 `fonts/` 目录下放置 `DejaVuSansMono.ttf` 字体文件。
