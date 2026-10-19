# 个体船声学重识别协议

**语言：** [English](README.md) | 简体中文

本仓库公开个体船声学重识别实验所使用的固定数据划分、任务构造代码、指标实现、主要模型训练与评估代码，以及可迁移的实验配置。

仓库不包含论文原文、论文表格、实验结果汇总、图片、原始音频、提取后的特征、模型权重或预测文件。

## 目录结构

```text
README.md
README_zh.md
requirements.txt
configs/
  deepship_current7.json
  qiandao_resnet18_multilabel.json
code/
  build_evaluation_tasks.py
  evaluation_metrics.py
  audit_deepship_min5_median_split.py
  build_deepship_25class_random_segment_split.py
  verify_splits.py
  audit_release_privacy.py
  models/
    resnet18/
    ecapa_tdnn/
    ast/
    panns_cnn14/
    qiandao_resnet18/
splits/
  deepship/
  qiandaoear22/
```

## DeepShip 严格评价协议

主实验采用 25 个个体船身份和非重叠 5 秒音频片段。

- 每个个体船固定一条 validation recording，作为 enrollment/gallery。
- 每个个体船固定一条未见过的 test recording，作为 query。
- 其余满足条件的 recordings 全部用于训练。
- train、validation、test 之间不存在源 recording 重叠。
- validation segments 的 embedding 按身份求平均，形成 gallery prototypes。
- test segments 用于闭集分类和 segment-level retrieval。
- 同一 test recording 的 segment 证据按固定规则聚合，用于 recording-level retrieval。
- embedding 匹配使用余弦相似度。
- hard negative 定义为同一船型、不同个体船。

固定划分文件包括：

- `splits/deepship/min5_median_recording_manifest.csv`
- `splits/deepship/min5_median_segment_manifest.csv`
- `splits/deepship/ship_label_map.json`
- `splits/deepship/random_segment_25class_manifest.csv`

random-segment split 只用于 RSS 与 RWP 的受控对比。它允许同一源 recording 的 segments 跨集合出现，不能作为严格 recording-wise benchmark 使用。

## QiandaoEar22 协议

QiandaoEar22 rebuilt recordings 被切成非重叠 5 秒片段，用于六目标 multi-label presence 分类：

- SpeedBoat
- QianDao
- KaiYuan
- No7
- GreenCity
- TheEarl

仓库保留三种背景设置下的 random-segment 与 recording-wise 固定划分：

- `fullDATA`：保留全部 DATAxxxx 背景 recordings。
- `dataxxxx30`：按已经固定的选择保留 30% DATAxxxx 数据。
- `noDATA`：删除 DATAxxxx，但保留其他有效的 all-zero negative 样本。

recording-wise manifest 强制每个 `rebuilt_uid` 只属于一个集合；random-segment manifest 则允许 `rebuilt_uid` 跨集合，用于研究随机短片段划分产生的偏差。

## 最终模型配置

DeepShip 严格 RWP 主对比使用七个配置：

- `RN18-IN`：ResNet18，ImageNet 初始化。
- `RN18-Rnd`：ResNet18，随机初始化。
- `ECAPA-CE`：ECAPA-TDNN，随机初始化。
- `AST-AS`：Audio Spectrogram Transformer，AudioSet 预训练初始化。
- `AST-Rnd`：Audio Spectrogram Transformer，随机初始化。
- `CNN14-AS`：PANNs CNN14，AudioSet 预训练初始化。
- `CNN14-Rnd`：PANNs CNN14，随机初始化。

七个配置均执行 25 类个体船身份分类，并使用交叉熵损失。分类、verification、segment retrieval 和 recording retrieval 共用同一个固定 RWP 与 enrollment/query 定义。历史 oversampling、TypePretrain、ECAPA-AAM 等开发配置不属于最终七模型对比。

`configs/deepship_current7.json` 记录随机种子、训练轮数、优化器、预处理参数和初始化方式。`configs/qiandao_resnet18_multilabel.json` 记录 QiandaoEar22 六标签 BCE 任务配置。配置中不包含本机绝对路径。

## 关键代码

- `code/build_evaluation_tasks.py`：从固定 recording manifest 构造身份集合、gallery/query 以及 hard/easy negative identity pairs。
- `code/evaluation_metrics.py`：classification、AUC、EER、Recall@K、MRR、单正例 gallery mAP、prototype 和 recording aggregation。
- `code/models/`：ResNet18、ECAPA-TDNN、AST、PANNs CNN14 和 Qiandao ResNet18 的关键训练与评估实现。
- `code/audit_deepship_min5_median_split.py`：审计严格 recording-wise split。
- `code/build_deepship_25class_random_segment_split.py`：以固定种子复现 25 类 random-segment split。

模型 checkpoint、第三方预训练权重和数据特征未包含在仓库中，需要用户按原始许可自行获取。

## 使用方法

所有公开 manifest 中的音频路径均为相对路径。使用时将 `segment_relpath` 或 `rel_path` 与本地数据集根目录拼接。复现实验时应直接复用仓库中的固定 manifest，不要重新随机生成划分。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

检查全部 split：

```bash
python code/verify_splits.py
```

构造严格评价任务：

```bash
python code/build_evaluation_tasks.py --output evaluation_tasks.json
```

发布前检查隐私、绝对路径和禁止上传的二进制文件：

```bash
python code/audit_release_privacy.py
```

## 数据说明

本仓库不重新分发 DeepShip 或 QiandaoEar22 音频。用户需要按照原数据集许可自行获取数据。本仓库只公开实验划分、协议代码、指标实现、主要模型代码和实验配置。

## 可重复性原则

`splits/` 中的 manifest 是正式实验定义。后续增加新模型时，应直接复用这些文件，确保不同模型之间的数据协议不发生变化。

