# Can vision-language models see underwater?

Workshop paper project for the Embodied AI Meta-alignment workshop at AAAI-27. The submission deadline is 24 Nov 2026, 23:59 AoE.

**The question.** Can an underwater robot rely on an off-the-shelf vision-language model or open-vocabulary detector, without labelled underwater data? Or does it still need a small detector fine-tuned on underwater images?

**What we compared.** Fine-tuned detectors and zero-shot models on the DUO underwater benchmark, all under one protocol: COCO AP via pycocotools, plus the TIDE error breakdown.
- **Fine-tuned:** YOLO11n/s/m/l, YOLOv8n, RT-DETR-L.
- **Zero-shot:** Grounding DINO, OWLv2, Qwen2.5-VL-3B/7B, Qwen3-VL-8B.

**Start with [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).** It covers what we ran, every number, the findings and what is still missing.

## Layout

| Path | Contents |
|---|---|
| `PROJECT_BRIEF.md` | Status, results, findings, open work |
| `EXPERIMENT_PLAN.md` | The plan the GPU runs followed (historical record) |
| `gpu_job/` | Scripts for fine-tuning, zero-shot runs and scoring; see `gpu_job/README.md` |
| `results/` | Small result files from our runs: metrics, timings, VLM replies, the RUOD train/test duplicate list |
| `paper/` | AAAI-27 style files; the paper is written here |
| `requirements.txt` | Package versions used |

**Not in this repository:**
- benchmark data: DUO from Figshare record 25370527; RUOD from its authors;
- model weights;
- the full detection files: about 160 MB, kept as a release zip.

**Rules for this project:**
- established public benchmarks only;
- open models only, no paid APIs;
- the paper is double-blind, so do not link this repository from the submission.
