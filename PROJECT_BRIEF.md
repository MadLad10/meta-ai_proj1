# Project brief: Can vision-language models see underwater?

Status on 19 Nov 2026, evening (UTC+6). GPU access has ended for good, so the results below are final unless another GPU turns up. This brief is for a co-author joining now. It says what the paper is about, what we ran, the numbers we have, what they show, and what is still missing.

---

## 1. The paper in one paragraph

**Question.** An underwater robot that picks or inspects objects needs a detector. Can it use an off-the-shelf vision-language model (VLM) or an open-vocabulary detector without labelled underwater data? Or does it still need a small detector fine-tuned on underwater images?

**Approach.** We compare both kinds of model on an established underwater benchmark (DUO), under one scoring protocol (COCO AP via pycocotools), and break their errors down with an established toolbox (TIDE).

**Working title.** *Can Vision-Language Models See Underwater? Zero-Shot Detection on Established Underwater Benchmarks.* The final wording will follow the results.

**Venue.** Embodied AI Meta-alignment workshop at AAAI-27, regular paper, 6–8 pages.
- AAAI-27 template, double-blind, submitted through HotCRP.
- Non-archival.
- In-person presentation in Montréal, 22–23 Feb 2027.
- **Deadline: 24 Nov 2026, 23:59 AoE, which is 25 Nov, 17:59 UTC+6.**

**Ground rules we set:**
- **Public benchmarks only.** No own data and no own simulator. An earlier simulator-based idea was dropped for this reason.
- **Free, open models only.** No paid APIs.
- **No overlap** with our other paper that is under review.

---

## 2. What we ran

All runs were done on a borrowed Windows PC with an RTX 5090 (32 GB). The software was Python 3.11, PyTorch 2.14.1 (CUDA 13.0), Ultralytics 8.4.60 and transformers 4.57.6. That PC is no longer available.

### 2.1 Benchmark: DUO

DUO is the Detecting Underwater Objects dataset (Liu et al., 2021, arXiv 2106.05681). It was built from the underwater robot-picking contests URPC 2017–2020 and the UDD dataset. Licence CC BY 4.0, Figshare record 25370527.

- **Size:** 7,782 images, 4 classes.
- **Classes:** holothurian (sea cucumber), echinus (sea urchin), scallop, starfish.
- **Official split:** 6,671 train and 1,111 test.
- **Our use of the training set:** we held out 10% of it (seed 17) to choose the training checkpoint. So 6,004 images train and 667 select the checkpoint. **The test set is never used for any choice.**
- **Test objects: 10,517, heavily imbalanced:**

| Class | Count | Share |
|---|---|---|
| Sea urchin | 7,201 | 68% |
| Starfish | 2,020 | |
| Sea cucumber | 1,079 | |
| Scallop | 217 | |

  COCO AP averages over classes, so the rare classes count as much as sea urchins.

### 2.2 Second benchmark: RUOD (prepared, not run)

RUOD has 14,000 images and 10 classes (official split 9,800 train / 4,200 test). It was downloaded and prepared, but no model was run on it before GPU access ended.

**Side finding:** 322 distinct images appear in both the official train and test sets. They cover 349 test entries and 1,143 test annotations. The list is in `outputs/gt/ruod_test_duplicates.json`. We have not found this reported anywhere.

### 2.3 Models

**Fine-tuned on DUO** (Ultralytics, starting from COCO weights). Same recipe for all six: 640 px, 40 epochs, batch 8, seed 17.

| Model | Published COCO AP (Ultralytics) | Training time (min) |
|---|---|---|
| YOLO11n | 39.5 | 31.1 |
| YOLO11s | 47.0 | 34.7 |
| YOLO11m | 51.5 | 49.0 |
| YOLO11l | 53.4 | 49.7 |
| YOLOv8n | 37.3 | 27.3 |
| RT-DETR-L | 53.0 | 154.6 |

**Zero-shot** (no underwater training at all; class names given in the prompt):

| Model | Kind | How it is asked |
|---|---|---|
| Grounding DINO base | Open-vocabulary detector | Text prompt `sea cucumber. sea urchin. scallop. starfish.` |
| OWLv2 base | Open-vocabulary detector | One query per class: `a photo of a <class>` |
| Qwen2.5-VL-3B / 7B Instruct | VLM | Prompt below; answer parsed as JSON boxes |
| Qwen3-VL-8B Instruct | VLM | Same prompt, but box coordinates on a 0–1000 grid |

**VLM prompt:**

> Detect every sea cucumber, sea urchin, scallop, starfish in this underwater image. Answer with JSON only: a list of objects, each {"bbox_2d": [x1, y1, x2, y2], "label": one of [sea cucumber, sea urchin, scallop, starfish]}, with coordinates in pixel coordinates of the image. Answer [] if there are none.

- Decoding is greedy, with at most 1,536 new tokens.
- Qwen's default image-size budget is used.

### 2.4 Scoring (identical for every model)

**COCO metrics with pycocotools on the DUO test set:**
- AP: average over IoU 0.50 to 0.95, the main number;
- AP50 and AP75;
- AP by object size (small means under 32×32 pixels);
- AR100: recall with up to 100 detections per image;
- AP per class.

**Score thresholds:**
- Fine-tuned detectors keep boxes with score ≥ 0.001, at most 300 per image.
- Open-vocabulary detectors keep boxes with score ≥ 0.01.
- Grounding DINO is scored as in its own COCO evaluation: each class's score comes from the model's token outputs for that class name, and the top 300 (box, class) pairs per image are kept.

**VLMs return no confidence score,** so every VLM box scores 1.0. Their AP therefore reflects a single operating point. That is a known disadvantage against models that rank their boxes, so we always show AP50 and AR100 next to AP.

**TIDE error breakdown** (Bolya et al., ECCV 2020). It reports how many AP50 points each error type costs:
- wrong class (Cls);
- poor box (Loc);
- both (Both);
- duplicate (Dupe);
- box on background (Bkg);
- missed object (Miss).

**Merged-class AP** (computed on the laptop): every class is relabelled "object", keeping one box per location (the highest-scoring class). This measures whether a model finds the objects regardless of naming.

**Same 100 images:** Qwen3-VL-8B only ran on the first 100 test images (section 5). To compare it fairly, every model is also scored on exactly those 100 images.

### 2.5 Ablations

| ID | What changes | Models | Status |
|---|---|---|---|
| B-A1 | Latin class names (holothurian, echinus) instead of common names | Grounding DINO, OWLv2 | Done |
| B-A2 | VLM image resolution, low and high | Qwen2.5-VL-7B | Low done; **high not run** |
| B-A3 | Ultralytics' own mAP against pycocotools | 6 fine-tuned | Done |
| B-A4 | Everything on RUOD | All | **Not run** |
| B-A5 | Prompt without "Answer [] if there are none" (first 100 images) | Qwen2.5-VL-3B/7B | Done; Qwen3-VL not run |
| B-A6 | Grounding DINO decoded the way the Hugging Face pipeline does it: it writes a phrase per box, which is then matched to a class name | Grounding DINO | Done |

---

## 3. The numbers

**How the numbers were checked:** every AP and AP50 below was recomputed on the laptop from the saved raw detections. All 17 runs match the stored values exactly. The other metrics come from the same pycocotools run on the GPU PC. **No confidence intervals yet** (section 6).

### 3.1 Main table: DUO test set, 1,111 images

| Model | Type | AP | AP50 | AP75 | AP_small | AR100 |
|---|---|---|---|---|---|---|
| YOLO11l | fine-tuned | **0.690** | 0.863 | 0.763 | 0.573 | 0.780 |
| YOLO11m | fine-tuned | 0.685 | 0.857 | 0.761 | 0.543 | 0.780 |
| YOLO11s | fine-tuned | 0.674 | 0.859 | 0.749 | 0.526 | 0.770 |
| RT-DETR-L | fine-tuned | 0.669 | 0.849 | 0.745 | 0.498 | 0.774 |
| YOLO11n | fine-tuned | 0.625 | 0.820 | 0.702 | 0.491 | 0.747 |
| YOLOv8n | fine-tuned | 0.624 | 0.824 | 0.707 | 0.417 | 0.741 |
| OWLv2 | zero-shot | **0.099** | 0.183 | 0.093 | 0.040 | 0.190 |
| Grounding DINO | zero-shot | 0.090 | 0.151 | 0.097 | 0.035 | 0.360 |
| Qwen2.5-VL-7B | zero-shot VLM | 0.013 | 0.024 | 0.012 | 0.000 | 0.021 |
| Qwen2.5-VL-3B | zero-shot VLM | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3-VL-8B | zero-shot VLM | full run not finished; see 3.2 | | | | |

### 3.2 The same first 100 test images, all models

| Model | AP | AP50 |
|---|---|---|
| RT-DETR-L | 0.692 | 0.881 |
| YOLO11l | 0.691 | 0.858 |
| YOLO11m | 0.684 | 0.865 |
| YOLO11s | 0.656 | 0.837 |
| YOLO11n | 0.628 | 0.827 |
| YOLOv8n | 0.606 | 0.809 |
| OWLv2 | 0.107 | 0.186 |
| Grounding DINO | 0.078 | 0.127 |
| **Qwen3-VL-8B** | **0.054** | **0.111** |
| Qwen2.5-VL-7B | 0.018 | 0.028 |
| Qwen2.5-VL-3B | 0.000 | 0.000 |

On 100 images the order of the fine-tuned models already shifts (RT-DETR-L is first here, fourth on the full set). Conclusions from 100 images need intervals.

### 3.3 Per class and per size (AP, full test set; Qwen3-VL on the first 100)

| Model | Sea cucumber | Sea urchin | Scallop | Starfish | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|---|---|
| YOLO11l | 0.690 | 0.745 | 0.556 | 0.768 | 0.573 | 0.708 | 0.674 |
| YOLO11n | 0.606 | 0.718 | 0.461 | 0.716 | 0.491 | 0.653 | 0.603 |
| OWLv2 | **0.001** | 0.182 | 0.042 | 0.170 | 0.040 | 0.093 | 0.123 |
| Grounding DINO | **0.004** | 0.086 | 0.136 | 0.135 | 0.035 | 0.109 | 0.096 |
| Qwen2.5-VL-7B | **0.004** | 0.016 | 0.016 | 0.015 | 0.000 | 0.005 | 0.024 |
| Qwen3-VL-8B (100 images) | **0.003** | 0.147 | 0.020 | 0.045 | 0.009 | 0.044 | 0.140 |

All six fine-tuned models: scallop is the lowest class (0.46–0.56) and sea urchin or starfish the highest.

### 3.4 Do they find the objects? Merged-class AP

| Model | AP, 4 classes | AP, merged | AP50, merged | AR100, merged |
|---|---|---|---|---|
| YOLO11l | 0.690 | 0.744 | 0.918 | 0.793 |
| YOLO11n | 0.625 | 0.705 | 0.899 | 0.767 |
| OWLv2 | 0.099 | 0.151 | 0.285 | 0.347 |
| Grounding DINO | 0.090 | 0.136 | 0.238 | 0.444 |
| Qwen3-VL-8B (100 images) | 0.054 | 0.100 | 0.222 | 0.239 |
| Qwen2.5-VL-7B | 0.013 | 0.016 | 0.032 | 0.018 |

### 3.5 TIDE error breakdown (AP50 points lost; full test set unless marked)

| Model | Cls | Loc | Both | Dupe | Bkg | Miss | All false positives | All false negatives |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | 0.66 | 2.23 | 0.22 | 0.31 | 6.54 | 2.06 | 11.96 | 4.76 |
| YOLO11s | 0.64 | 1.82 | 0.19 | 0.18 | 5.44 | 2.00 | 8.83 | 4.30 |
| YOLO11m | 0.65 | 1.72 | 0.24 | 0.12 | 5.67 | 1.80 | 8.62 | 4.53 |
| YOLO11l | 0.47 | 1.66 | 0.21 | 0.15 | 5.95 | 1.87 | 8.71 | 4.21 |
| YOLOv8n | 0.67 | 1.78 | 0.21 | 0.28 | 6.68 | 2.32 | 11.50 | 5.05 |
| RT-DETR-L | 0.48 | 1.75 | 0.21 | 0.08 | 5.35 | 3.07 | 8.56 | 5.45 |
| OWLv2 | 22.0 | 3.0 | 0.2 | 0.2 | 4.7 | 6.9 | 14.5 | 31.5 |
| Grounding DINO | 19.5 | 2.2 | 0.2 | 0.3 | 5.6 | 3.6 | 31.2 | 11.5 |
| Qwen2.5-VL-7B | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 32.6 | 1.8 | 43.3 |
| Qwen3-VL-8B (100 images) | 0.7 | 0.4 | 0.0 | 0.0 | 9.7 | 8.6 | 22.8 | 13.6 |

### 3.6 COCO rank against DUO rank (6 fine-tuned models)

- **Spearman ρ = 0.83** between published COCO AP and DUO test AP.
- **Spread:** the COCO gap between best and worst is 16.1 points (53.4 vs 37.3). On DUO it is 6.6 points (0.690 vs 0.624).
- **Exception:** RT-DETR-L is 2nd on COCO and 4th on DUO. It also has the most missed objects (TIDE Miss 3.07, the highest of the six).
- **Evaluation code (B-A3):** Ultralytics' own mAP and pycocotools differ by at most 0.008 AP for every model.

### 3.7 Ablations

**B-A1: Latin vs common class names**

| Model | Common | Latin |
|---|---|---|
| Grounding DINO | 0.090 | 0.081 |
| OWLv2 | 0.099 | 0.086 |

Common names are slightly better for both models. Note: OWLv2 with Latin names produced 189 boxes entirely outside the image (dropped); with common names it produced none.

**B-A6: Grounding DINO, phrase decoding vs its own scoring**
- **Phrase decoding** (the Hugging Face pipeline): AP 0.048. 32% of its boxes (4,633 of 14,427) get a phrase matching no class. Examples: "seachin", a broken piece of "sea urchin"; "sea"; and "sea urchin starfish", which names two classes at once.
- **Its own scoring:** AP 0.090.

**B-A5: removing "Answer [] if there are none" (first 100 images)**

| Model | Default prompt: empty answers | AP, default (same 100 images) | Without the sentence: empty answers | AP, without the sentence |
|---|---|---|---|---|
| Qwen2.5-VL-3B | 1,111 of 1,111 (all images) | 0.000 | 0 of 100 | 0.017 |
| Qwen2.5-VL-7B | 1,005 of 1,111 (all images) | 0.018 | 0 of 100 | 0.018 |

Without the sentence, replies were cut off at the 1,536-token limit on 4 of 100 images (3B) and 18 of 100 (7B).

**B-A2: lower image resolution, Qwen2.5-VL-7B, full test set**

| Resolution | AP | AP50 | Boxes | Empty answers |
|---|---|---|---|---|
| Default (up to 1280×28×28 pixels) | 0.013 | 0.024 | 498 | 1,005 of 1,111 |
| Low (up to 640×28×28 pixels) | 0.007 | 0.013 | 653 | 1,017 of 1,111 |

At low resolution, 6 replies were cut off and 17 boxes fell outside the image (dropped). Fewer pixels give slightly more boxes but halve the score. Both numbers are tiny, so this is weak evidence.

**Qwen3-VL-8B reply behaviour (default prompt, first 100 images):**
- 17 empty answers;
- 19 replies cut off at the token limit (runaway lists of boxes);
- 53 boxes outside the image, all from 3 of the cut-off replies. Complete replies have correct coordinates.

### 3.8 Inference time per image (from the run logs; rough)

| Model | Seconds per image | Note |
|---|---|---|
| Qwen3-VL-8B | 15.9 | |
| Qwen2.5-VL-7B / 3B (answering) | 4–5 | |
| Qwen2.5-VL-3B (default prompt) | 0.2 | It answers `[]` immediately |
| OWLv2 | 0.3 | |
| Grounding DINO | 0.2 | Ran alongside other jobs, so the timing is not clean |

The fine-tuned detectors' inference time was not measured; their weights are still on the GPU PC.

---

## 4. What the evidence shows

Point estimates only, until the intervals are computed. Hypotheses H2–H5 were written down before the results.

1. **The zero-shot gap is very large (H2: supported).** The best zero-shot model (OWLv2, 0.099 AP) reaches about one sixth of the weakest fine-tuned detector (YOLOv8n, 0.624). This holds for every zero-shot model, on the full set and on the 100-image subset.
2. **Different failure modes by model type (H3 "misses dominate for every zero-shot model": not supported).**
   - **Qwen2.5-VL mostly misses objects.** The 7B loses 32.6 AP50 points to misses and almost nothing to anything else.
   - **Qwen3-VL-8B splits its losses** between boxes on background (9.7 points) and misses (8.6 points).
   - **The open-vocabulary detectors mostly put the wrong class on boxes.** OWLv2 loses 22.0 points and Grounding DINO 19.5 to wrong class.
   - **Merged-class AP says naming is only part of it.** With classes merged, OWLv2 rises from 0.099 to 0.151 and Grounding DINO from 0.090 to 0.136. That is still far below the two fine-tuned detectors we checked (0.705 and 0.744 merged). Wrong names cost these models about a third of their AP; finding and boxing the objects is the larger problem.
3. **No zero-shot model can detect sea cucumbers.** AP is 0.001–0.004 for every zero-shot model, against 0.61–0.69 for fine-tuned ones. The best any zero-shot model does on any class is 0.18 (OWLv2, sea urchin).
4. **Hardest class and size (H5: partly supported).**
   - Small objects have the lowest AP for every model that detects anything (all except Qwen2.5-VL-3B).
   - Scallop is the hardest class for all six fine-tuned models, but for zero-shot models it is sea cucumber.
5. **Newer and larger VLMs do better (H4: supported in direction).** On the same 100 images: Qwen3-VL-8B 0.054, Qwen2.5-VL-7B 0.018, Qwen2.5-VL-3B 0.000. Even the best VLM stays below both open-vocabulary detectors on those images (Grounding DINO 0.078, OWLv2 0.107), and it is about 50 times slower than OWLv2 (15.9 s against 0.3 s per image; section 3.8).
6. **Qwen2.5-VL's "nothing here" answers come from our prompt, but they hide little ability.**
   - With "Answer [] if there are none", the 3B model returned an empty answer on every test image and the 7B on 90%.
   - Without the sentence, neither returned an empty answer, but AP barely moved: 3B from 0.000 to 0.017, 7B unchanged at 0.018.
   - Prompt wording strongly changes VLM behaviour here, without changing what the model can see.
7. **How the output is decoded matters a lot for open-vocabulary detectors.** Grounding DINO's AP almost halves (0.090 to 0.048) when its output goes through the standard Hugging Face phrase decoding instead of its own scoring.
8. **COCO rank mostly carries over underwater, but COCO overstates the differences.** ρ = 0.83 over 6 models (descriptive only). A 16-point spread on COCO becomes 6.6 points on DUO, and RT-DETR-L drops two places.
9. **Fine-tuned detectors mainly fail by firing on background.** TIDE Bkg costs 5.4–6.7 AP50 points, against at most 3.1 for any other error type. So their main failure is false alarms, not missed objects.

**What we can't yet claim:**
- that any of this generalises beyond DUO (RUOD was not run);
- statistical significance (no intervals yet);
- anything about Qwen3-VL-8B beyond 100 images.

---

## 5. What did not get done

- **Qwen3-VL-8B on the full test set.** The run was stopped at 600 of 1,111 images when GPU access ended, and nothing was saved. It needs about 5 GPU hours (15.9 s per image).
- **RUOD (B-A4).** It would take about 14 GPU hours for everything, or about 7 for the zero-shot models alone.
- **The high-resolution half of B-A2,** and the B-A5 prompt test for Qwen3-VL-8B.
- **Inference time of the fine-tuned detectors.**

---

## 6. What else we need

### 6.1 Must do before submission (laptop, no GPU)

1. **Statistics.** 500-draw bootstrap over test images for every comparison: the zero-shot gap; the VLM comparisons on the same 100 images; class names; prompt. Also an interval for Spearman ρ.
2. **Final tables and figures:**
   - main table;
   - TIDE bar chart;
   - COCO vs DUO scatter;
   - per-class table;
   - ablation table.
3. **Positioning against related work.** The general effect is already known: RF100-VL (2025) shows zero-shot VLMs and open-vocabulary detectors fail on unusual domains, and fine-tuning wins. Closely related:
   - UWBench (2025): underwater VLM benchmark with a grounding task, but no detection AP;
   - MarineDet (2023): open-vocabulary marine detection on its own data.

   As far as we found, nobody has scored current VLMs and open-vocabulary detectors zero-shot on DUO next to fine-tuned detectors under one protocol, with an error breakdown. These papers still need to be read in full; so far we have seen abstracts and summaries only.
4. **Write the paper from scratch** on the AAAI-27 template, in `paper/`:
   - every number checked by script;
   - references verified;
   - anonymised.
5. **Admin:** HotCRP account, confirm in-person attendance, and check whether AAAI-27 wants an LLM-use statement.

### 6.2 Would make the paper clearly stronger (only if another GPU becomes available; the borrowed PC is gone)

In order of value:
1. **Qwen3-VL-8B on the full test set** (about 5 h).
2. **Crop test:** give each VLM the crop of each true box and ask only for the class. This separates "can't recognise" from "can't find", the main open question from finding 2. About 1–2 h.
3. **RUOD zero-shot** (about 7 h), and its fine-tuned models (about 7 h), to show the findings hold beyond one benchmark.
4. **High-resolution run** for Qwen2.5-VL-7B (15–30 min).

### 6.3 Weaknesses a reviewer will raise

- **"Zero-shot loses to fine-tuned" is expected.** The interest has to come from *why*: the two failure modes, sea cucumbers, prompt-induced abstention and decoding sensitivity.
- **One benchmark** (unless RUOD gets run).
- **Fit with the workshop.** The best-matching topic is "Robotic Foundation Models and Embodied Agents", but the workshop is mainly about alignment. The paper should frame itself around what perception a robot can rely on, and whether a model shows when it is failing (silent empty answers, runaway outputs).
- **VLMs have no confidence scores,** which handicaps their AP. We report AP50 and AR next to it, but should say so clearly.
- **ρ comes from 6 models** and is descriptive only.

### 6.4 Decisions for us

- Do we look for another GPU (6.2), or submit with what we have (DUO only)?
- Who writes which sections, and the internal deadline for a full draft (proposed: 23 Nov)?

---

## 7. Where things are

| What | Where |
|---|---|
| Experiment plan as run on the GPU PC | `EXPERIMENT_PLAN.md` |
| Scripts | `gpu_job/` (Job 4 = DUO fine-tuning; Job 5 = zero-shot; Job 7 = COCO scoring of the fine-tuned models; Job 6 = RUOD); see `gpu_job/README.md` |
| GPU PC's final report | `gpu_job/PROGRESS.txt` |
| Small result files (metrics, timings, VLM replies, RUOD duplicate list) | `results/` (same layout as the scripts' `outputs/`) |
| Full raw results with all detections (161 MB + 0.4 MB zips, too large for git) | GitHub release `gpu-part1` of our main project repo; unzip into `gpu_job/outputs/` |
| AAAI-27 style files (the paper goes here) | `paper/` |
