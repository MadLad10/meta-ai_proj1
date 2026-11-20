# Research assessment and paper framing

19 Nov 2026. This is a reviewer-style assessment of what our evidence supports. It comes before any paper text.

**Numbers.** All numbers come from the saved detections, recomputed with `analysis/assess.py` (output in `analysis/assess.json`).
- Intervals are 95% paired bootstrap intervals over test images (500 draws, seed 20261119).
- Our bootstrap code reproduces pycocotools' AP exactly when no resampling is done.
- "First 100" means the first 100 DUO test images. Qwen3-VL-8B was run only on those.

**Literature.** It is described from abstracts and secondary summaries; the full papers still need to be read (section 6.3).

---

## 1. Research understanding

**What we set out to do.** Measure how well zero-shot foundation models detect objects on an established underwater benchmark. DUO comes from underwater robot-picking contests; its four classes are sea cucumber, sea urchin, scallop and starfish.
- **Compared against:** six fine-tuned detectors.
- **Protocol:** one scoring protocol for all (COCO AP via pycocotools) and an error breakdown (TIDE).
- **Ablations:** class names, image resolution, prompt wording and output decoding.

**What the experiments show, taken together:**
1. **The zero-shot gap is very large, as expected.**
   - Weakest fine-tuned detector: AP 0.624 [0.593, 0.655].
   - Best zero-shot model, OWLv2: 0.099 [0.088, 0.111].
   - Difference: +0.525 [0.493, 0.554].
2. **Choices made while evaluating a model move its score by more than the differences between models.**
   - Turning Grounding DINO's output into labels with the standard Hugging Face phrase decoding, instead of its own scoring, costs 0.042 AP [0.032, 0.053]. That is about half its score.
   - The difference between Grounding DINO and OWLv2 is only +0.009 [−0.015, +0.031], not significant.
3. **The VLMs' lower AP is partly a consequence of their having no confidence scores.** On the first 100 images, AP ranks Qwen3-VL-8B (0.054) below OWLv2 (0.107; difference +0.053 [0.017, 0.078]). But when each detector keeps only as many distinct boxes as Qwen3-VL returned on each image, and every box counts equally, the picture changes (precision and recall at IoU 0.5 with the correct class; `analysis/e1.py`, corrected 19 Nov):

   | Model | Precision | Recall |
   |---|---|---|
   | Qwen3-VL-8B | 0.38 | 0.36 |
   | OWLv2 | 0.41 | 0.34 |
   | Grounding DINO | 0.26 | 0.25 |
   | YOLO11n (fine-tuned) | 0.61 | 0.57 |

   - Qwen3-VL beats Grounding DINO: +0.116 [0.073, 0.168] in precision and +0.112 [0.075, 0.153] in recall.
   - Qwen3-VL and OWLv2 cannot be told apart: precision −0.033 [−0.089, 0.019], recall +0.027 [−0.017, 0.080].
4. **One prompt sentence decides whether Qwen2.5-VL answers at all, without changing what it can see.**
   - With "Answer [] if there are none", the 3B returns no boxes on 1,111 of 1,111 images and the 7B on 1,005 of 1,111.
   - Without the sentence, neither returns an empty answer (first 100 images).
   - AP is unchanged for the 7B (+0.000 [−0.009, +0.011]) and rises only slightly for the 3B (+0.017 [0.009, 0.032]).
5. **The open-vocabulary detectors put boxes on many objects but cannot name them.**
   - **Overall share found:** a box of any class overlapping the object at IoU ≥ 0.5, among the image's 100 highest-scoring boxes. Grounding DINO 69%, OWLv2 62%, the fine-tuned YOLO11n 96%.
   - **Sea cucumbers (1,079 objects):**

     | Model | Found | Named correctly | Label on the best-overlapping box |
     |---|---|---|---|
     | Grounding DINO | 61% | 4% | sea urchin on 574 of 659 found objects |
     | OWLv2 | 53% | 8% | starfish on 389 of 566 |
     | YOLO11n (fine-tuned) | 97% | 96% | — |

   - **Per-class AP for sea cucumbers:** 0.004 [0.002, 0.008] and 0.001 [0.001, 0.002] respectively.
   - **Merged-class AP** (all classes treated as one, so names no longer matter) rises by about half: Grounding DINO 0.090 → 0.136 [0.127, 0.146], OWLv2 0.099 → 0.151 [0.139, 0.164]. That is still far below fine-tuned detectors (0.705 and 0.744 merged).
6. **Renaming the classes moves the confusion; it does not remove it.** With Latin names:
   - Grounding DINO names 34% of sea cucumbers correctly, but now calls 1,260 sea urchins "holothurian" and 1,014 "scallop".
   - OWLv2 names 0% of sea cucumbers correctly and calls 517 of them starfish.
   - AP does not improve: Grounding DINO common minus Latin is +0.009 [−0.001, +0.016]; for OWLv2, +0.013 [0.001, 0.025].
7. **What does not explain the failure:**
   - Lower image resolution for Qwen2.5-VL-7B: +0.005 [−0.002, +0.009], not significant.
   - Model size and generation help the VLMs (7B minus 3B +0.013 [0.009, 0.018]; Qwen3-VL-8B minus Qwen2.5-VL-7B +0.036 [0.021, 0.066] on the first 100), but all stay far below fine-tuned detectors.
8. **Findings that do not belong to this paper's question:**
   - COCO rank predicts DUO rank among the 6 fine-tuned detectors: ρ = 0.83 [0.66, 0.94].
   - 322 images are duplicated across RUOD's official train and test split. RUOD was never run, so this has no results attached.

---

## 2. Candidate contributions, ranked

| Rank | Contribution | Evidence | What is already known | What is ours | Verdict |
|---|---|---|---|---|---|
| 1 | **Found but misnamed.** Zero-shot detectors localize a majority of objects but assign marine class names inconsistently. The assignment follows the wording of the class list, not what the object looks like (findings 5, 6) | Strong: full test set, 10,517 objects, intervals, two models, two vocabularies | Semantic confusion is the main bottleneck for open-vocabulary detectors on aerial imagery (arXiv 2601.22164). Zero-shot detection fails on unusual domains (RF100-VL, NeurIPS 2025) | Per-object found-vs-named decomposition on an underwater benchmark. Class-specific, systematic confusions (sea cucumber → sea urchin or starfish). Renaming moves them without fixing them | **Central** |
| 2 | **Evaluation choices change the verdict** about as much as the model does: output decoding, missing scores in VLM output, prompt wording (findings 2–4) | Decoding: strong. Matched box budget: moderate (100 images, one budget, no intervals yet). Prompt: moderate (100 images, two models) | VLM detectors have low recall and fail to reject absent objects (WeDetect, CVPR 2026, on COCO). Qwen2.5-VL scores below Grounding DINO zero-shot (RF100-VL) | Quantified, with intervals: decoding effect vs model difference; AP-based VLM-vs-detector comparisons reflect ranking, not only perception; one sentence flips abstention from 100% to 0% with no AP change | **Second pillar**: the controls needed before contribution 1 can be read |
| 3 | Size of the zero-shot gap (finding 1) | Strong | Known in general (RF100-VL includes aquatic datasets) | Numbers on DUO under one protocol | **Context only** |
| 4 | COCO rank predicts underwater rank (ρ = 0.83) | Weak: 6 models from one library | Transfer correlations are well studied | Little | **Appendix or cut** |
| 5 | RUOD train/test duplicates | Strong, but no experiment uses RUOD | Leakage is common in vision datasets; none found for RUOD | The list itself | **Cut.** Report to the dataset maintainers instead |

**Why not "can VLMs see underwater?"** That framing answers itself (no), and RF100-VL and WeDetect already show the general pattern. Its only distinctive element, the VLM comparison, is the part our controls show to be most confounded by the absence of scores.

---

## 3. Recommended framing

**Research question.** When zero-shot foundation models fail on an underwater detection benchmark, do they fail to see the objects, fail to name them, or appear to fail because of how they are evaluated?

**Central thesis.** On DUO, zero-shot detectors and VLMs place boxes on most objects that a fine-tuned detector finds, but cannot reliably name marine classes. Their class assignment follows the wording of the class list, not the object's appearance. Common evaluation choices change their scores and rankings as much as the choice of model does, and so hide where the failure lies.

**Novelty, relative to prior work:**
- **RF100-VL and WeDetect** establish *that* zero-shot detection fails and that VLM detectors have low recall. We show *where* it fails on one benchmark: per object, found vs named.
- **The aerial study (2601.22164)** finds semantic confusion dominates through vocabulary reduction. We show it object by object, with class-to-class confusions, and show that renaming the classes moves the confusions without fixing them.
- **Evaluation choices.** To our knowledge, no prior work quantifies with intervals how output decoding, score-less VLM output and prompt wording each compare with the differences between models. This must be confirmed when the related papers are read in full.

**Contribution statements, as they would appear in the paper:**
1. A controlled evaluation of six fine-tuned detectors, two open-vocabulary detectors and three VLMs on the DUO underwater benchmark. Every comparison has paired bootstrap intervals.
2. Evidence that three evaluation choices move zero-shot scores by as much as the difference between models, with recommendations for reporting:
   - the decoding path;
   - a matched box budget for score-less VLMs;
   - prompt sensitivity.
3. A per-object decomposition showing that zero-shot detectors localize a majority of objects but misname them in systematic, wording-dependent ways. Sea cucumbers, a main target of underwater picking robots, are found in over half the cases but almost never named correctly.

**Embodied relevance (the workshop hook).**
- **Topic:** "Robotic Foundation Models and Embodied Agents". A picking robot needs the species, not just a box; the failure we isolate is exactly the one that matters for that decision.
- **Possible second topic:** "Evaluation" (benchmarks and metrics), through contribution 2.
- **Fit with the workshop's alignment focus is thin;** the paper should not stretch to claim more.

---

## 4. Titles

1. **Found but Misnamed: Diagnosing Zero-Shot Foundation-Model Detection on Underwater Robot-Picking Imagery** *(recommended)*
2. Where Zero-Shot Detection Breaks Underwater: Naming, Ranking and Evaluation Choices
3. Seeing Is Not Naming: Zero-Shot Detectors and Vision-Language Models on an Underwater Benchmark
4. How Evaluation Choices Shape Zero-Shot Detection Verdicts: An Underwater Case Study
5. Can Vision-Language Models See Underwater? Localization, Naming and Evaluation on DUO

**Why title 1.**
- "Found but misnamed" is literally what we measure (found share against named share), and it is the central, best-supported finding.
- "Diagnosing" signals an analysis paper, not a new method.
- "Underwater robot-picking imagery" states the scope (one benchmark, DUO) without claiming generality.

**What the alternatives risk.**
- Title 3 overclaims: the models localize 62–69% of objects, not all.
- Title 5 promises an answer to a broad question that one benchmark cannot settle.
- Title 2 is accurate but less focused. Use it if contribution 2 grows after the remaining analyses.

---

## 5. Paper outline (AAAI-27, at most 8 pages)

**Abstract** (one paragraph):
- the problem (robots need species-level detection without labels);
- the gap (known *that* zero-shot fails, not *where*);
- method (controlled evaluation on DUO, per-object decomposition);
- the two findings with their key numbers;
- the recommendation.

**1. Introduction** (about 1 page)
- **Motivation.** Underwater picking and inspection robots need to recognize specific species. Foundation models promise zero-shot detection.
- **Prior work.** It shows large zero-shot gaps (RF100-VL) and low VLM recall (WeDetect), but does not separate localization, naming and evaluation artefacts.
- **Question and approach.**
- **Fig. 1 (teaser).** One DUO image: a sea cucumber boxed by Grounding DINO and labelled "sea urchin", next to the per-class found-vs-named bars. DUO is CC BY 4.0, so we can show its images with attribution.
- **Three contribution bullets** (section 3).

**2. Related work** (about 0.75 page)
- Underwater detection benchmarks and their annotation quality: DUO, RUOD, UDD; "Why Domain Matters" (2026); RUOD-R.
- Open-vocabulary detection and domain transfer: GLIP, Grounding DINO, OWLv2, ODinW, RF100-VL, the aerial study, robustness under distribution shift.
- VLMs as detectors: Qwen2.5-VL, Qwen3-VL, WeDetect, UWBench (underwater grounding).
- Error analysis: TIDE.

**3. Experimental setup** (about 1.25 pages)
- DUO and its splits; the fine-tuning recipe.
- Zero-shot models and prompts, word for word.
- **Scoring choices made explicit:**
  - Grounding DINO's own scoring vs phrase decoding;
  - VLM boxes scored 1.0;
  - score thresholds.
- **Measures:**
  - COCO AP;
  - found share and named share;
  - merged-class AP;
  - precision and recall at a matched box budget.
- Paired bootstrap.

**4. Results**
- **4.1 The gap** (short). **Table 1:** all models, AP / AP50 / AR100 with intervals.
- **4.2 Evaluation choices change the verdict.** **Table 2:**
  - decoding: 0.090 vs 0.048, against the 0.009 model difference;
  - AP vs matched-budget precision and recall for Qwen3-VL, OWLv2 and Grounding DINO;
  - prompt sentence: empty answers 100% → 0%, ΔAP ≈ 0.

  Conclusion: decide these choices before comparing models; this is why section 4.3 uses budget-free, per-object measures.
- **4.3 Found but misnamed.**
  - **Fig. 2:** found vs named share per class and model.
  - **Table 3:** what sea cucumbers, scallops and starfish are called, with common and with Latin names.
  - Merged-class AP. Renaming moves the confusion.
- **4.4 What does not explain it.** Resolution (n.s.), scale and generation (help, but still far below). One paragraph or a small table.

**5. Discussion** (about 0.75 page)
- **For robots:** zero-shot models can propose regions, but naming needs labelled data or a different interface. For example, combine zero-shot boxes with a classifier trained on a few labels; this is a hypothesis we have not tested.
- **For evaluation:**
  - report the decoding path;
  - compare score-less VLMs at a matched budget;
  - report prompt sensitivity;
  - report per-class found and named shares.
- **Limitations** (section 6.2).

**6. Conclusion** (a few sentences)

**Appendix / supplement, if allowed:**
- fine-tuning details and full per-class tables;
- TIDE tables, with the caveat in section 6.2;
- COCO-vs-DUO rank;
- prompts;
- the Qwen3-VL cut-off replies.

**Removed from the main paper:**
- COCO rank transfer;
- RUOD duplicates;
- the TIDE "VLMs miss, detectors mislabel" contrast, which the matched-budget control shows to depend on how many boxes each model outputs;
- low resolution, except as one line.

---

## 6. Evidence audit

### 6.1 Strongly supported (full test set, intervals exclude zero)
- **The zero-shot gap:** +0.525 [0.493, 0.554].
- **Decoding:** +0.042 [0.032, 0.053], against a non-significant model difference of +0.009 [−0.015, +0.031].
- **Sea-cucumber misnaming by both open-vocabulary detectors:** found 61% / 53%, named 4% / 8%, n = 1,079. Per-class AP intervals near zero.
- **Merged-class gains** for both detectors, with intervals.
- **Renaming shifts the confusions** (full test set), while AP does not improve for Grounding DINO.
- **Qwen2.5-VL's empty answers are caused by the prompt** (counts on 1,111 and 100 images). No AP change for the 7B.
- **VLM ordering:** 3B < 7B, and 7B < Qwen3-VL-8B (first 100).

### 6.2 Needs qualification
- **Matched-budget comparison.**
  - 100 images and one way of setting the budget; no intervals yet.
  - OWLv2 could fill only 9.6 of the 11.3 boxes per image.
  - Qwen3-VL had 19 cut-off replies out of 100.

  State it as "comparable at a matched budget", not "equal".
- **"Found" is a generous criterion:** any of the top 100 boxes, at IoU ≥ 0.5. Report sensitivity (top 10; IoU 0.7). Localization is also clearly weaker than fine-tuned (62–69% vs 96%), so the paper must not say the models "see" everything.
- **The naming result holds for the open-vocabulary detectors.** Qwen3-VL mostly *misses* sea cucumbers (found 9%, 110 objects in 100 images). Its pattern differs and rests on little data.
- **TIDE's wrong-class figures (19–22 AP50 points) are larger than the merged-class gain (9–10 AP50 points).** TIDE's fixes are not additive, and these detectors put several class labels on the same box. Use TIDE as supporting evidence only.
- **One benchmark and one prompt template per model.** Generality beyond DUO is unknown. RF100-VL and the aerial study suggest the pattern is broader, but we cannot show it.
- **Alternative explanations to discuss:**
  - DUO annotation conventions and label noise. Underwater label quality is a documented issue (Why Domain Matters; RUOD-R).
  - Text-side effects. "Sea cucumber" and "sea urchin" share the word "sea", which may explain Grounding DINO's bias toward "sea urchin". The Latin-name run, which has no shared word, changes the bias; that is consistent with this explanation.
  - Class imbalance: 68% of objects are sea urchins.
- **Correlation vs cause.** The renaming experiment is the only intervention on naming. Everything else is observational.

### 6.3 Critical gaps

**Essential, and doable without a GPU:**
- **E1 (laptop, under an hour).**
  - Intervals for the found and named shares, and for matched-budget precision and recall.
  - Sensitivity to top-k (10 / 100) and IoU (0.5 / 0.7).
  - A second budget definition: the number of true objects per image.
- **E2 (laptop CPU, about 1–2 h): the box-crop naming test** for OWLv2 and Grounding DINO. Give each model the true box (a crop) and ask only for the class, on a stratified sample: all 217 scallops plus about 300 of each other class.
  - This is the direct test of "cannot name even with the right box". It turns contribution 1 from observational into experimental.
  - Needs transformers from PyPI and the two models from Hugging Face (about 1.5 GB), run under the memory cap.
  - A VLM crop test is not feasible on the laptop (not enough memory).

**Must also be done:**
- Read RF100-VL, WeDetect, the aerial study, UWBench and Why Domain Matters in full, to confirm the novelty statements in section 3.

**Not possible without a GPU (state as limitations, not gaps to hide):**
- a second benchmark (RUOD);
- Qwen3-VL on the full test set;
- VLM confidence scores, e.g. from token probabilities, which would let VLMs be ranked like detectors.

---

## 7. Final recommendation

**Top-tier main conference (CVPR, ICCV, ECCV, NeurIPS, ICLR): not as it stands.**
- One benchmark, two open-vocabulary detectors and three VLMs (one on 100 images), and no method.
- The general effect is known.
- A main-track version would need several underwater benchmarks, more VLMs including ones that output confidences, the crop test for VLMs, and the evaluation-choice analysis repeated across domains. It would be a benchmark-track paper.

**AAAI-27 workshop (non-archival): yes, defensible, provided that:**
1. it is framed as a diagnostic case study ("found but misnamed", with evaluation choices as the necessary controls), not as a broad answer to "can VLMs see underwater";
2. E1 and E2 are done, E2 especially;
3. limitations are stated plainly: one benchmark, Qwen3-VL on 100 images, no VLM confidences.

**The strongest single message:** zero-shot detectors on DUO put a box on most sea cucumbers but call them sea urchins or starfish, and renaming the classes only moves the error. That message is new in this domain as far as we can tell, supported by the full test set, and directly relevant to what a picking robot must decide.
