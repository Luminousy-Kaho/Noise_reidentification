# Individual-Vessel Acoustic Re-identification Protocols

**Language:** English | [简体中文](README_zh.md)

This repository publishes the frozen data-split protocols used for individual-vessel acoustic re-identification experiments on DeepShip and the auxiliary split-integrity experiments on QiandaoEar22.

The repository intentionally contains no manuscript, paper tables, result spreadsheets, figures, raw audio, extracted features, model checkpoints, or prediction files.

## Repository contents

```text
README.md
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

## DeepShip protocol

The main benchmark is a 25-identity, recording-wise protocol built from non-overlapping 5 s segments.

- Each vessel identity has one validation recording used as enrollment/gallery data.
- Each vessel identity has one unseen test recording used as query data.
- All remaining eligible recordings are assigned to training.
- Source recordings do not overlap across train, validation, and test.
- Validation segment embeddings are averaged per identity to form gallery prototypes.
- Test segments are used for closed-set classification and segment retrieval.
- Test segments from the same recording are aggregated for recording-level retrieval.
- Cosine similarity is used for embedding matching.
- Hard negatives are different vessel identities belonging to the same vessel type.

The published fixed files are:

- `splits/deepship/min5_median_recording_manifest.csv`
- `splits/deepship/min5_median_segment_manifest.csv`
- `splits/deepship/ship_label_map.json`
- `splits/deepship/random_segment_25class_manifest.csv`

The random-segment manifest is provided only for the controlled RSS-versus-RWP comparison. It permits segments from the same source recording to appear in different subsets and must not be treated as the strict benchmark protocol.

## QiandaoEar22 protocol

QiandaoEar22 rebuilt recordings are divided into non-overlapping 5 s segments for a six-target multi-label presence task:

- SpeedBoat
- QianDao
- KaiYuan
- No7
- GreenCity
- TheEarl

The repository includes frozen random-segment and recording-wise manifests for three background settings:

- `fullDATA`: all DATAxxxx background recordings are retained.
- `dataxxxx30`: 30% of the DATAxxxx segments/recordings are retained according to the frozen selection.
- `noDATA`: DATAxxxx recordings are excluded while other valid all-zero negatives remain.

The recording-wise manifests enforce one split per `rebuilt_uid`. Random-segment manifests deliberately allow a `rebuilt_uid` to cross subsets.

## Model configurations used in the study

The final strict DeepShip comparison used seven configurations:

- `RN18-IN`: ResNet18 initialized from ImageNet.
- `RN18-Rnd`: ResNet18 with random initialization.
- `ECAPA-CE`: ECAPA-TDNN with random initialization.
- `AST-AS`: Audio Spectrogram Transformer initialized from AudioSet pretraining.
- `AST-Rnd`: Audio Spectrogram Transformer with random initialization.
- `CNN14-AS`: PANNs CNN14 initialized from AudioSet pretraining.
- `CNN14-Rnd`: PANNs CNN14 with random initialization.

All configurations perform 25-way vessel-identity classification with cross-entropy loss. The same frozen recording-wise split and the same enrollment/query definition are used for classification, verification, segment retrieval, and recording retrieval. Historical oversampling, TypePretrain, ECAPA-AAM, and development-only configurations are not part of the final seven-model comparison.

Model implementations rely on standard public libraries and architectures (`torchvision`, SpeechBrain-style ECAPA-TDNN components, Hugging Face AST, and PANNs CNN14). The key training and evaluation implementations are retained under `code/models/`. Checkpoints, pretrained-weight files, extracted features, and prediction outputs are omitted.

The machine-readable settings in `configs/` record the frozen protocol, random seed, optimization settings, preprocessing parameters, and initialization source. Dataset and feature locations must be supplied locally; no workstation-specific path is embedded in the release.

## Using the manifests

Audio paths in the manifests are relative. Join `segment_relpath` or `rel_path` with your local dataset root. Do not regenerate the split when reproducing the reported protocol.

Run the included consistency checks from the repository root:

```bash
python code/verify_splits.py
```

The protocol interfaces can be materialized with:

```bash
python code/build_evaluation_tasks.py --output evaluation_tasks.json
```

Reusable implementations of classification, verification, segment retrieval, recording aggregation, and multi-label metrics are provided in `code/evaluation_metrics.py`.

The script uses only the Python standard library. A successful run ends with:

```text
ALL SPLIT CHECKS PASSED
```


## Data availability

DeepShip and QiandaoEar22 audio are not redistributed. Users must obtain the datasets under their original terms. This repository only publishes split assignments and the code needed to audit or reproduce the split construction.

## Reproducibility note

The manifests in `splits/` are the authoritative experiment definitions. For new models, reuse these files directly so that the data protocol remains unchanged across architectures.

