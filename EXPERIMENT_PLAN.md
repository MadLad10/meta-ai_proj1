# Experiment plan: AAAI-27 workshop paper, A to Z

> **Historical record.** This is the plan the GPU PC followed on 18–19 Nov 2026. GPU access has ended; for the current status and results see `PROJECT_BRIEF.md`.


Written 19 November 2026. **This replaces every earlier version** (see git history).

**The user's rules:**
- established benchmarks only, no custom simulation or own data;
- **nothing paid** (no paid APIs or credits);
- the GPU PC runs **Windows only** (no Linux or WSL).

**The paper:** "Can vision-language models see underwater?" on DUO and RUOD.
- **E3VS-Bench (the former track A) is deferred.** It needs Linux (section H).
- **GPU Jobs 1, 2, 3 and 3b are cancelled.** They used our simulator.

**Who does what:**

| Who | Where | What | Section |
|---|---|---|---|
| **GPU agent** | Windows PC with RTX 5090 | Runs the jobs | D |
| **Laptop agent** | Ubuntu laptop, CPU only | Analysis and paper | E |
| **User** | Both machines | Git, file transfer, submission | — |

Sections A–C are fixed now, before any result. Later additions are labelled exploratory.

---

## A. The question

> **Can vision-language models see underwater? On established underwater detection benchmarks, how far are zero-shot VLMs and open-vocabulary detectors from fine-tuned detectors, why do they fail, and does a detector's COCO rank predict its underwater rank?**

### A1. Benchmarks (established, published)
- **DUO** (Liu et al., arXiv 2106.05681; ICME Workshops 2021):
  - 7,782 real underwater images;
  - 4 classes: holothurian (sea cucumber), echinus (sea urchin), scallop, starfish;
  - official split 6,671 train / 1,111 test.
- **RUOD** (Fu et al., Neurocomputing 2023):
  - 14,000 real underwater images, 10 classes;
  - split 9,800 train / 4,200 test, to be confirmed by `check`.
- **COCO val2017 mAP50-95** (published by Ultralytics): YOLO11n 39.5, YOLO11s 47.0, YOLO11m 51.5, YOLO11l 53.4, YOLOv8n 37.3, RT-DETR-L 53.0.

### A2. Models
| Group | Models | How |
|---|---|---|
| Fine-tuned detectors | YOLO11 n/s/m/l, YOLOv8n, RT-DETR-L | COCO-pretrained, then fine-tuned on each benchmark's train split (640 px, 40 epochs, batch 8, seed 17; checkpoint chosen on 10% held out from train) |
| Open-vocabulary detectors (zero-shot) | Grounding DINO base, OWLv2 base | Class names as text prompts, no training |
| VLMs (zero-shot) | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct | Asked to return every object as JSON boxes with labels, no training |

### A3. Scoring (identical for every model)
- **Protocol:** standard COCO evaluation (pycocotools) on the test split: AP (IoU 0.50–0.95), AP50, AP75, AP_small, AP_medium, AP_large, AR100, and per-class AP.
- **Grounding DINO is scored as in its own COCO evaluation** (positive map): each class's score is the mean token probability over the tokens of its name, and the top 300 (query, class) pairs per image are kept. No phrase is decoded, so no box is lost. (Changed 19 Nov, before any full-run Grounding DINO number was looked at: the Hugging Face phrase decoding left 32% of its boxes without a class on the full DUO run, e.g. 'seachin', 'sea', 'sea urchin starfish'. The phrase-decoding run is kept as a comparison, B-A6.)
- **VLMs give no confidence score,** so all their boxes score 1.0. Their AP therefore reflects a single operating point; AR and AP50 are reported alongside.
- **Error breakdown: TIDE** (Bolya et al., ECCV 2020), an established toolbox. It splits the AP lost into:
  - classification error (`Cls`);
  - localisation error (`Loc`);
  - both (`Both`);
  - duplicates (`Dupe`);
  - background false positives (`Bkg`);
  - missed objects (`Miss`);
  - plus the totals of false positives and false negatives.
- **RUOD:** zero-shot VLMs run on a fixed random subset of 1,000 test images (seed 17), because VLM inference is slow. Every other model on RUOD uses the full test set and is also reported on the same subset, for a paired comparison.

### A4. Research questions and hypotheses
| | Question | Hypothesis | How judged |
|---|---|---|---|
| **RQ1** | Does COCO rank predict underwater rank? | (descriptive) | Spearman ρ between COCO AP and DUO AP over the 6 fine-tuned architectures (and RUOD AP), with a bootstrap interval over test images. With 6 points it is descriptive only |
| **RQ2** | How large is the zero-shot gap? | **H2:** every zero-shot model is below the weakest fine-tuned model on DUO AP | AP difference with a 95% bootstrap interval over test images (500 draws), below 0, for each zero-shot model |
| **RQ3** | Why do zero-shot models fail? | **H3:** missed objects (TIDE `Miss`) are the largest error type for every zero-shot model on DUO | TIDE point estimates (descriptive; TIDE has no interval), plus FalseNeg against FalsePos |
| **RQ4** | Do larger or newer VLMs see better underwater? | **H4:** Qwen2.5-VL-7B > 3B, and Qwen3-VL-8B > Qwen2.5-VL-7B, on DUO AP and AR100 | Bootstrap interval of each difference above 0 |
| **RQ5** | Where is it hardest? | **H5:** AP_small is the lowest size bin, and scallop the lowest class, for every model | Point estimates, with bootstrap intervals for scallop against the other classes |

### A5. Ablations
| # | Ablation | Question |
|---|---|---|
| B-A1 | Class names: common ("sea cucumber") against Latin ("holothurian") for Grounding DINO and OWLv2 | Does vocabulary decide zero-shot underwater detection? |
| B-A2 | VLM image resolution: low / default / high pixel budget for Qwen2.5-VL-7B | Are VLM misses a resolution problem (AP_small, AR_small)? |
| B-A3 | Same metric protocol: Ultralytics' own mAP (Job 4) against pycocotools (Job 7) for the fine-tuned models | Are comparisons robust to the evaluation code? |
| B-A4 | Second benchmark: everything above on RUOD | Do the findings hold beyond DUO? |
| B-A6 | Grounding DINO scoring: its own COCO evaluation (positive map, primary) against Hugging Face phrase decoding matched to class names (32% of boxes unmatched on DUO) | How much of Grounding DINO's result depends on turning phrases into classes? |
| B-A5 | VLM prompt without the sentence "Answer [] if there are none", first 100 DUO test images, all three VLMs | Are the VLMs' empty answers caused by our prompt? (Added 19 Nov after the smoke test showed Qwen2.5-VL answering [] on almost every image; exploratory) |

---

## B. Jobs (all on the Windows GPU PC)
| Job | Script | What | GPU time (est.) | Priority |
|---|---|---|---|---|
| 4 | `job4_duo.py` | Fine-tune 6 architectures on DUO and evaluate (**running**) | ~3–4 h | 1 |
| 7-DUO | `job7_finetuned_coco.py --dataset duo` | Re-score the 6 DUO models with pycocotools and TIDE | ~10 min | 1 |
| 5-DUO | `job5_duo_zeroshot.py` | Grounding DINO, OWLv2, 3 VLMs on the DUO test set | ~6–8 h | 1 |
| 5-abl | `job5_duo_zeroshot.py --names latin --only grounding-dino owlv2`; `--pixels low` and `--pixels high --only qwen2.5-vl-7b` | Ablations B-A1, B-A2 | ~4–6 h | 2 |
| 6 | `job6_ruod.py` | Download, prepare, fine-tune 6 architectures on RUOD | ~6–8 h | 3 |
| 7-RUOD | `job7_finetuned_coco.py --dataset ruod` | Re-score the RUOD models | ~20 min | 3 |
| 5-RUOD | `job5_duo_zeroshot.py --dataset ruod` | Zero-shot models on RUOD (VLMs on 1,000 images) | ~6–8 h | 3 |

**Cut rule if time runs short:** priority 3 first, then priority 2. Priority 1 alone makes the paper.

---

## C. The paper
- **Title (final wording follows the results):** "Can Vision-Language Models See Underwater? Zero-Shot Detection on Established Underwater Benchmarks".
- **Content:**
  - results table: every model on DUO (and RUOD) with AP, AP50, AP_small and AR100;
  - figure: TIDE error breakdown per model;
  - figure: COCO AP against DUO AP for the 6 architectures;
  - per-class table, highlighting scallop;
  - ablation tables;
  - related work: underwater detection benchmarks, open-vocabulary detection, VLM grounding, and VLM perception limits (BLINK, All-Angles Bench).
- **Writing:** fresh text on the AAAI-27 template, every number checked by script, at most 8 pages. Venue: Embodied AI Meta-alignment (due 24 Nov 23:59 AoE). The embodied angle: what perception an underwater robot can rely on.

---

## D. GPU runbook (Windows, for the GPU agent)

### D0. Rules
- **Git:** the user does all git writes. Print the command and wait.
- **No AI attribution** anywhere.
- **Nothing paid.** No Linux or WSL.
- **Nothing large in git:** not data, weights or outputs.
- **Secrets:** never print, log or commit tokens.
- **Downloads:** only those listed here:
  - PyPI packages;
  - Ultralytics starting weights;
  - Hugging Face models: Grounding DINO, OWLv2, Qwen2.5-VL-3B/7B, Qwen3-VL-8B;
  - DUO (Figshare, already done) and RUOD (via its GitHub README link, approved by the user).
- **Stop and ask the user when:** counts don't match, a smoke test looks wrong, or something fails twice.
- **Never change benchmark data, prompts, recipes or thresholds.** Record every deviation.

### D1. Environment (the existing `.venv_gpu`)
```
.\.venv_gpu\Scripts\Activate.ps1
$env:HF_HOME = "D:\hf_cache"
python -m pip install -r requirements.txt   # versions actually used: see requirements.txt
python -c "import torch, transformers, pycocotools, tidecv; from transformers import AutoModelForImageTextToText; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```
Qwen3-VL needs transformers 4.57 or newer. If the PyTorch version changes or CUDA becomes `False`, stop and tell the user.

### D2. Order of work
1. **Job 4 (running):** let it finish (`train`, then `evaluate`).
2. **Job 7-DUO:** `python job7_finetuned_coco.py --dataset duo`. Run it before deleting `outputs\job4\_runs`.
3. **Job 5-DUO.** **Rerun every Job 5 model with the current script.** Results from the earlier script (in `PROGRESS.txt`) are superseded: Grounding DINO's label matching now keeps fragments like "cucumber" (3,609 boxes were dropped before), and the VLM reply parser now accepts any key order and saves every reply. Delete `outputs\job5\` first, or move it to `outputs\job5_old\`.
   
   - **Smoke test:** `python job5_duo_zeroshot.py --limit 20`. Check that each model prints an AP and detections. For VLMs, the printed first reply must be JSON boxes. **`boxes outside image` should be near 0**; if not, the coordinate convention is wrong, so stop and report.
   - **Full run:** `python job5_duo_zeroshot.py`.
   - **Grounding DINO with its own scoring (added 19 Nov, after `git pull`):** the default is now `--gdino posmap` (folder suffix `_posmap`). The earlier folder `grounding-dino_common` is the phrase-decoding run (B-A6) and stays.
     1. Smoke: `python job5_duo_zeroshot.py --limit 20 --only grounding-dino`. It prints `grounding-dino class tokens:`; each class must list the word pieces of its own name (e.g. sea cucumber -> sea, cucumber). `unmatched labels` must be 0. Otherwise stop and report.
     2. Full: `python job5_duo_zeroshot.py --only grounding-dino` (about 3 min).
     3. To save the decoded phrases of B-A6: move `outputs\job5\duo\grounding-dino_common` to `outputs\job5_old\`, then `python job5_duo_zeroshot.py --gdino text --only grounding-dino` (about 3 min; writes `replies.jsonl` with every phrase).
4. **Ablations:**
   - `python job5_duo_zeroshot.py --names latin --only grounding-dino owlv2` (Grounding DINO with its own scoring, after the pull)
   - `python job5_duo_zeroshot.py --pixels low --only qwen2.5-vl-7b`
   - `python job5_duo_zeroshot.py --pixels high --only qwen2.5-vl-7b`
   - `python job5_duo_zeroshot.py --prompt noempty --first 100 --only qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-8b` (B-A5)
5. **RUOD (priority 3):**
   1. Download via the link in github.com/dlut-dimt/RUOD's README into `gpu_job\ruod_raw\`, and write its licence into `ruod_raw\LICENCE_NOTE.txt`.
   2. `python job6_ruod.py check`: it must report 9,800 / 4,200, or stop and report.
   3. `python job6_ruod.py prepare`, `train --smoke`, `train`, `evaluate`.
   4. `python job7_finetuned_coco.py --dataset ruod`
   5. `python job5_duo_zeroshot.py --dataset ruod`
6. **One GPU job at a time.**
7. **Interim reports (so the laptop can start the analysis):** after the full Job 5-DUO run, and again after the ablations:
   - update `PROGRESS.txt` with the `python summarize.py` output and any new findings (reply cut-offs, unmatched labels, boxes outside the image), and ask the user to push it;
   - zip `outputs\job4`, `job5`, `job7` and `gt` (without `_runs`, `_val` and images) into `b_outputs_part1.zip` (after the ablations: `b_outputs_part2.zip`) for the user to copy to the laptop.

**Check progress at any time:** `python summarize.py`.

### D3. Troubleshooting (the only edits allowed)
| Problem | Fix |
|---|---|
| CUDA out of memory, Job 4 or 6 | Batch 8 is already set; report if it still fails |
| CUDA out of memory, a VLM | Close other GPU jobs. Otherwise report; do not change the pixel budget except through `--pixels` |
| A VLM reply is cut off (no closing `]`) on many images | Report the count; do not change `max_new_tokens`. Qwen3-VL-8B's cut-offs (4/20 in the smoke test) are a known finding (user decision): report and continue |
| `tide_error` in `metrics.json` | Report the message. COCO metrics stand without TIDE |
| Many `unmatched labels` (> 10% of boxes) | VLMs: report a few replies from `replies.jsonl`. Grounding DINO `--gdino text`: expected (~32%), report and continue. Grounding DINO `posmap`: must be 0 |

### D4. Send results back (user)
1. `python summarize.py > outputs\summary.txt`.
2. Copy the test annotation files into `outputs\gt\`: DUO's and RUOD's test COCO JSON. The laptop needs them for bootstrap intervals.
   Also write `outputs\gt\ruod_test_duplicates.json`: the RUOD test image IDs whose content duplicates a training image (the 322 found by the check). The laptop reports fine-tuned results on the de-duplicated test set too.
3. **Keep** `outputs\job4\_runs` and `outputs\job6\_runs` on the GPU PC (the `best.pt` weights may be needed for timing runs); leave them and `_val` out of the zip. Delete them only when the user says so.
4. Zip `outputs\job4`, `job5`, `job6`, `job7`, `gt` and `summary.txt` into `b_outputs.zip`. No images.
5. Copy it to the laptop at `gpu_job/outputs/`. Don't commit it.

**Report to the user:** the `summarize.py` output, versions, the RUOD licence, and every deviation.

---

## E. Laptop runbook (for the laptop agent)
1. **Analysis script** (`tools/analysis_b.py`), using `detections.json` and the ground-truth files:
   - all metrics recomputed with pycocotools;
   - bootstrap over test images (500 draws, seed 20261119) for every AP difference in A4;
   - Spearman ρ of COCO against DUO and RUOD AP with an image-bootstrap interval;
   - TIDE tables.

   Writes `tools/analysis_b.json`.
2. **Paper:**
   - fresh text and figures (TIDE bars, COCO-against-DUO scatter);
   - `tools/facts.py` and `check_numbers.py` rebuilt for these sources;
   - references verified three times;
   - at most 8 pages.
3. **Final checks:** build, fonts embedded (no Identity-H in figures), anonymity, PDF metadata. Send the PDF to the user by 23 Nov.

## F. Timeline (user's time, UTC+6)
| Date | GPU PC | Laptop |
|---|---|---|
| 19–20 Nov | Jobs 4, 7-DUO, 5-DUO, then ablations | Analysis script, paper skeleton and related work |
| 20–21 Nov | RUOD (priority 3) if time | Analysis as results arrive |
| 22 Nov | — | Paper written from results |
| 23 Nov | — | Checks; PDF to the user |
| 24–25 Nov | — | User review; submit before 25 Nov 17:59 (24 Nov 23:59 AoE) |

## G. Licences
- **Data:** DUO and RUOD, as recorded in their licence notes.
- **Models:**
  - Ultralytics models: AGPL-3.0.
  - Grounding DINO, OWLv2 and Qwen2.5-VL-7B: Apache-2.0.
  - Qwen2.5-VL-3B: Qwen research licence (non-commercial research).
  - Qwen3-VL-8B: record from its model card.

## H. Deferred: E3VS-Bench (VLM active perception)
The planned failure analysis on E3VS-Bench is the strongest next step. It splits failures into "looked in the wrong place" and "misread a good view", using annotated answerable viewpoints, with local open answerers and judges. It needs Linux (vLLM, gsplat, the benchmark's scripts) and gated ScanNet++ / SceneSplat++ access. It is kept for a later paper, if Linux GPU access becomes available.
