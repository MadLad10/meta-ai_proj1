# Scripts

All results were produced on one Windows 11 PC with an RTX 5090 (32 GB), Python 3.11.9 and CUDA 13.0. Run every script from this folder.

## 1. Environment

```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r ../requirements.txt
```

## 2. Data (not in this repository)

- **DUO.** Get `DUO.zip` from Figshare record 25370527 (CC BY 4.0) and unzip it into `duo_raw/`.
- **RUOD.** Use the download link in the README of github.com/dlut-dimt/RUOD and unzip it into `ruod_raw/`.

## 3. Runs

| Step | Command | What it does |
|---|---|---|
| Job 4 | `python job4_duo.py check`, then `prepare`, `train`, `evaluate` | Fine-tunes YOLO11n/s/m/l, YOLOv8n and RT-DETR-L on DUO, with 10% of the official train split held out to choose the checkpoint |
| Job 7 | `python job7_finetuned_coco.py --dataset duo` | Scores the fine-tuned models with pycocotools and TIDE (score ≥ 0.001, up to 300 boxes per image) |
| Job 5 | `python job5_duo_zeroshot.py` | Zero-shot Grounding DINO, OWLv2, Qwen2.5-VL-3B/7B and Qwen3-VL-8B on the DUO test set |
| Job 6 | `python job6_ruod.py check`, then `prepare`, `train`, `evaluate` | The Job 4 procedure on RUOD (prepared, never trained) |
| Status | `python summarize.py` | Prints every finished result |

**Job 5 options used for the ablations:**

| Option | Ablation |
|---|---|
| `--names latin --only grounding-dino owlv2` | Latin class names (B-A1) |
| `--pixels low` or `--pixels high` with `--only qwen2.5-vl-7b` | VLM image resolution (B-A2) |
| `--prompt noempty --first 100 --only qwen2.5-vl-3b qwen2.5-vl-7b` | Prompt without "Answer [] if there are none" (B-A5) |
| `--gdino text --only grounding-dino` | Hugging Face phrase decoding instead of Grounding DINO's own scoring (B-A6) |
| `--first N` | Run on the first N test images only |
| `--limit 20` | Smoke test |

A VLM run saves each answer as soon as it is produced, to `partial.jsonl`. Running the same command again continues where an interrupted run stopped.

## 4. Outputs

Results go to `outputs/` (not in git), with this layout:
- `outputs/job4/<model>/` for fine-tuning;
- `outputs/job7/duo/<model>/` for COCO scoring of the fine-tuned models;
- `outputs/job5/duo/<run>/` for zero-shot runs: `detections.json`, `metrics.json`, `record.json` and, for VLMs, `replies.jsonl`.

The small files from our runs are in `../results/`. The full outputs, including all detections, are kept as a release zip because they are too large for git.
