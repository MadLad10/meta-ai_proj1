"""Tables and macros for the two-benchmark paper (called at the end of make_tex.py, which defines mac, macros, A, E, CR,
f3, ci, pct, pci, N, M, Z, O, P, ROOT, TIME). Inputs added here: analysis/ruod.json, analysis/crops.json (ruod, default_labels)."""
import json

R = json.loads((ROOT / 'analysis' / 'ruod.json').read_text()); RM = R['models']; RC = CR['ruod']['models']; DL = CR['default_labels']
SHORTLAB = dict(holothurian='sea cucumber', echinus='sea urchin', scallop='scallop', starfish='starfish', fish='fish', corals='coral',
                diver='diver', cuttlefish='cuttlefish', turtle='turtle', jellyfish='jellyfish')
NM = {'yolov8n': 'YoloEightN', 'yolo11n': 'YoloN', 'yolo11s': 'YoloS', 'yolo11m': 'YoloM', 'yolo11l': 'YoloL', 'rtdetr-l': 'Rtdetr',
      'gdino': 'GD', 'owlv2': 'OWL', 'qwen25_3b': 'QThree', 'qwen25_7b': 'QSeven', 'qwen3_8b': 'QwenVL'}

# ---- RUOD macros
for k, nm in NM.items():
    r = RM[k]
    mac('rAp' + nm, f3(r['ap']['ap'])); mac('rApci' + nm, ci(r['ap']['ci']))
    mac('rSea' + nm, f3(r['seafloor_ap']['ap'])); mac('rSeaci' + nm, ci(r['seafloor_ap']['ci']))
    mac('rOth' + nm, f3(r['other_ap']['ap'])); mac('rOthci' + nm, ci(r['other_ap']['ci']))
    fn = r['found_named']
    for grp, g in (('seafloor', 'Sea'), ('other', 'Oth')):
        mac('rFound' + g + nm, pct(fn[grp]['found']['share'])); mac('rNamed' + g + nm, pct(fn[grp]['named']['share']))
        mac('rFoundci' + g + nm, pci(fn[grp]['found']['ci'])); mac('rNamedci' + g + nm, pci(fn[grp]['named']['ci']))
    if 'ap_dedup' in r:
        mac('rDedup' + nm, f3(r['ap_dedup']['ap']))
for k in ['gdino', 'owlv2', 'qwen3_8b', 'yolo11n']:
    b = RM[k]['budget']; nm = NM[k]
    mac('rbP' + nm, f3(b['precision'])); mac('rbR' + nm, f3(b['recall'])); mac('rbBoxes' + nm, f'{b["boxes_per_image"]:.1f}')
for key, nm in [('budget Qwen3-VL-8B minus gdino precision', 'PGD'), ('budget Qwen3-VL-8B minus gdino recall', 'RGD'),
                ('budget Qwen3-VL-8B minus owlv2 precision', 'POWL'), ('budget Qwen3-VL-8B minus owlv2 recall', 'ROWL')]:
    d = R['differences'][key]; mac('rbd' + nm, sg(d['diff'])); mac('rbdci' + nm, ci(d['ci']))
d = R['differences']['weakest fine-tuned minus best zero-shot detector']; mac('rdGap', sg(d['diff'])); mac('rdciGap', ci(d['ci']))
mac('rGapWeakest', {'yolov8n': 'YOLOv8n', 'yolo11n': 'YOLO11n'}.get(d['models'][0], d['models'][0])); mac('rGapBest', {'gdino': 'Grounding DINO', 'owlv2': 'OWLv2'}[d['models'][1]])
for key, nm in [('OWLv2 minus Grounding DINO', 'OwlGd'), ('Grounding DINO minus Qwen3-VL-8B (subset)', 'GdQwen'), ('OWLv2 minus Qwen3-VL-8B (subset)', 'OwlQwen'),
                ('Qwen3-VL-8B minus Qwen2.5-VL-7B (subset)', 'QwenGen')]:
    d = R['differences'][key]; mac('rd' + nm, sg(d['diff'])); mac('rdci' + nm, ci(d['ci']))
for k, nm in NM.items():
    d = R['seafloor_gap'][k]; mac('rSeaGap' + nm, f3(d['diff']))
mac('rTestImages', N(4200)); mac('rTestObjects', N(len(json.loads((O / 'gt' / 'ruod_instances_test.json').read_text())['annotations'])))
mac('rSubset', N(1000)); dupl = json.loads((O / 'gt' / 'ruod_test_duplicates.json').read_text()); mac('rDupImages', N(dupl['duplicate_test_images']))
mac('rDupAnn', N(dupl['test_annotations_on_duplicates']))
mac('rCrops', N(CR['ruod']['crops']))
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL'), ('qwen25_3b', 'QThree'), ('qwen25_7b', 'QSeven'), ('qwen3_8b', 'QwenVL'), ('gdino_latin', 'GDlatin'), ('owlv2_latin', 'OWLlatin')]:
    m = RC[k]; mac('rCrop' + nm, pct(m['mean_class_accuracy'])); mac('rCropci' + nm, pci(m['mean_ci']))
    mac('rCropSea' + nm, pct(m['seafloor_accuracy'])); mac('rCropSeaci' + nm, pci(m['seafloor_ci']))
    mac('rCropOth' + nm, pct(m['other_accuracy'])); mac('rCropOthci' + nm, pci(m['other_ci']))
    for b in ('duo', 'ruod'):
        dl = DL[f'{b}:{k}']; p = 'dlDuo' if b == 'duo' else 'dlRuod'
        mac(p + 'Lab' + nm, SHORTLAB[dl['label']] if not ('latin' in k and dl['label'] in ('holothurian', 'echinus')) else dl['label'])
        mac(p + nm, pct(dl['share'])); mac(p + 'ci' + nm, pci(dl['ci'])); mac(p + 'Cor' + nm, pct(dl['correct_seafloor']))
mac('rCropScalMax', pct(max(RC[k]['per_class']['scallop']['acc'] for k in ['gdino', 'owlv2', 'qwen25_3b', 'qwen25_7b', 'qwen3_8b'])))
mac('rCropCuttleGD', pct(RC['gdino']['per_class']['cuttlefish']['acc'])); mac('rCropFishOWL', pct(RC['owlv2']['per_class']['fish']['acc']))
ff = RM['gdino']['found_seafloor_labels']; mac('rFoundSeaLabGD', SHORTLAB[ff['most_common']]); mac('rFoundSeaLabShareGD', pct(ff['share']))
ff = RM['owlv2']['found_seafloor_labels']; mac('rFoundSeaLabOWL', SHORTLAB[ff['most_common']]); mac('rFoundSeaLabShareOWL', pct(ff['share']))
# prompt variants on DUO (first 100 images)
for lab, nm in [('B-A5 Qwen3: no-empty minus default prompt', 'PromptQwen'), ('single 3B: single-class minus default prompt', 'SingleThree'),
                ('single 7B: single-class minus default prompt', 'SingleSeven'), ('single Qwen3: single-class minus default prompt', 'SingleQwen')]:
    d = A['differences_first100'][lab]; mac('d' + nm, sg(d['diff'])); mac('dci' + nm, ci(d['ci']))
for k, nm in [('qwen25_3b_single100', 'QThreeSingle'), ('qwen25_7b_single100', 'QSevenSingle'), ('qwen3_8b_single100', 'QwenVLSingle'), ('qwen3_8b_noempty100', 'QwenVLNoEmpty'), ('qwen3_8b', 'QwenVLHund')]:
    mac('apH' + nm, f3(A['ap_first100'][k]['ap']))

# ---- Table 1: detection on both benchmarks
rows = [('YOLOv8n', 'yolov8n'), ('YOLO11n', 'yolo11n'), ('YOLO11s', 'yolo11s'), ('YOLO11m', 'yolo11m'), ('YOLO11l', 'yolo11l'), ('RT-DETR-L', 'rtdetr-l'),
        None, ('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2'), None, ('Qwen2.5-VL-3B', 'qwen25_3b'), ('Qwen2.5-VL-7B', 'qwen25_7b'), ('Qwen3-VL-8B', 'qwen3_8b')]
groups = {'yolov8n': 'Fine-tuned on each benchmark', 'gdino': 'Zero-shot open-vocabulary detectors', 'qwen25_3b': 'Zero-shot VLMs'}
t = ['\\begin{table*}[t]', '\\centering', '\\small', '\\begin{tabular}{lccccc}', '\\toprule',
     ' & DUO & \\multicolumn{3}{c}{RUOD} & \\\\', '\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}',
     'Model & AP (95\\% CI) & AP (95\\% CI) & Picking-class AP & Other-class AP & ms per image \\\\', '\\midrule']
for r in rows:
    if r is None:
        t.append('\\midrule'); continue
    name, k = r
    if k in groups:
        t.append(f'\\multicolumn{{6}}{{l}}{{\\textit{{{groups[k]}}}}} \\\\')
    rr = RM[k]; dag = '$^\\dagger$' if k in ('qwen25_3b', 'qwen25_7b', 'qwen3_8b') else ''
    t.append(f"{name} & {f3(af[k]['ap'])} {ci(af[k]['ci'])} & {f3(rr['ap']['ap'])} {ci(rr['ap']['ci'])}{dag} & {f3(rr['seafloor_ap']['ap'])} & {f3(rr['other_ap']['ap'])} & {ms(k)} \\\\")
t += ['\\bottomrule', '\\end{tabular}',
      '\\caption{Detection results (COCO AP, pycocotools) on the DUO test set (\\nTestImages{} images, the four picking classes) and the RUOD test set '
      '(\\rTestImages{} images, ten classes). Picking-class AP is the mean over RUOD\'s four picking classes (sea cucumber, sea urchin, scallop, starfish), '
      'other-class AP the mean over its six other classes. Intervals are 95\\% paired bootstrap intervals over test images. $^\\dagger$VLMs on RUOD: fixed '
      'random subset of \\rSubset{} test images, on which the zero-shot detectors score \\rApSubGD{} (Grounding DINO) and \\rApSubOWL{} (OWLv2). VLM boxes carry '
      'no confidence score and are all scored 1.0; Qwen2.5-VL results use the prompt with ``Answer [] if there are none\'\', which makes them return empty lists (Section~\\ref{sec:vlm}). Time per image on one RTX 5090 (DUO): fine-tuned detectors, median over 200 images processed one at a time '
      'on an idle GPU; zero-shot models, wall-clock time over the run. $^\\ddagger$Grounding DINO shared the GPU with another job.}', '\\label{tab:main}', '\\end{table*}']
mac('rApSubGD', f3(RM['gdino']['ap_subset']['ap'])); mac('rApSubOWL', f3(RM['owlv2']['ap_subset']['ap']))
(P / 'tab_main.tex').write_text('\n'.join(t) + '\n')

# ---- Table 2: finding, naming, naming given the box, default answer
fnd = E['found_named']; S_ = 'top100_iou50'
def duo_fn(k):
    a = fnd[k][S_]['all']; return pct(a['found']), pct(a['named'])
t = ['\\begin{table*}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{2.0pt}', '\\begin{tabular}{lccccccccll}', '\\toprule',
     ' & \\multicolumn{3}{c}{DUO, full images} & \\multicolumn{2}{c}{RUOD picking, full} & \\multicolumn{3}{c}{Named, box crop} & \\multicolumn{2}{c}{Default answer, picking crops} \\\\',
     '\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}',
     'Model & Found & Named & Strict & Found & Named & DUO & RUOD pick. & RUOD other & DUO & RUOD \\\\', '\\midrule']
def row(name, k, ck=None, full=True):
    ck = ck or k
    _nobj = len(GTD['annotations'])
    if full and k in fnd:
        f1, n1 = duo_fn(k)
        s1 = pct(sum(E['labels'][k][c][c] for c in E['labels'][k]) / _nobj) if k in E['labels'] else '--'
    elif full and not M[k].get('detections'):              # no boxes at all: nothing found or named
        f1 = n1 = s1 = '0'
    else:
        f1 = n1 = s1 = '--'
    if full and k in RM:
        fs = RM[k]['found_named']['seafloor']; f2, n2 = pct(fs['found']['share']), pct(fs['named']['share'])
    else:
        f2 = n2 = '--'
    cd = pct(CR['models'][ck]['mean_class_accuracy']) if ck in CR['models'] else '--'
    cs = pct(RC[ck]['seafloor_accuracy']) if ck in RC else '--'; co = pct(RC[ck]['other_accuracy']) if ck in RC else '--'
    def dl(b):
        key = f'{b}:{ck}'
        if key not in DL: return '--'
        lab = DL[key]['label']; lab = lab if ('latin' in ck and lab in ('holothurian', 'echinus')) else SHORTLAB[lab]
        return f"{lab} {pct(DL[key]['share'])}\\%"
    return f"{name} & {f1} & {n1} & {s1} & {f2} & {n2} & {cd} & {cs} & {co} & {dl('duo')} & {dl('ruod')} \\\\"
t += [row('YOLO11n (fine-t.)', 'yolo11n'), '\\midrule', row('Grounding DINO', 'gdino'), row('\\quad Latin names', 'gdino', 'gdino_latin', full=False),
      row('OWLv2', 'owlv2'), row('\\quad Latin names', 'owlv2', 'owlv2_latin', full=False), '\\midrule',
      row('Qwen2.5-VL-3B', 'qwen25_3b'), row('Qwen2.5-VL-7B', 'qwen25_7b'), row('Qwen3-VL-8B', 'qwen3_8b'), '\\bottomrule', '\\end{tabular}',
      '\\caption{Finding and naming (\\%). Found: share of true objects overlapped (IoU $\\geq$ 0.5) by any of the image\'s 100 highest-scoring boxes, of any class; '
      'named: by such a box with the correct class; strict: the best-overlapping such box carries the correct class (DUO: all objects; RUOD: the four picking classes; VLMs on RUOD: the \\rSubset-image subset). '
      'Named, box crop: mean per-class accuracy when the model sees only the crop of a true box (chance 25\\% on DUO, 10\\% on RUOD; '
      'RUOD columns averaged over the four picking and the six other classes). Default answer: the answer a model gives most often for picking-class crops, '
      'and the share of those crops that receive it. YOLO11n was not run on crops. Qwen2.5-VL full-image results use the prompt with ``Answer [] if there are none\'\', which makes them return empty lists (Section~\\ref{sec:vlm}).}', '\\label{tab:naming}', '\\end{table*}']
(P / 'tab_naming.tex').write_text('\n'.join(t) + '\n')

# ---- Table 3: evaluation choices
a100 = A['ap_first100']; df_ = A['differences_full']; d100_ = A['differences_first100']
neg = lambda d: f"{sg(-d['diff'])} [{f3(-d['ci'][1])}, {f3(-d['ci'][0])}]"
B_ = E['budget']['qwen3_count_full']['class_aware']
t = ['\\begin{table}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{lc}', '\\toprule', 'Setting & Value (95\\% CI) \\\\', '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(a) Grounding DINO decoding (DUO AP)}} \\\\',
     f"Own scoring $-$ phrase decoding & {macros['dDecoding']} {macros['dciDecoding']} \\\\",
     f"OWLv2 $-$ Grounding DINO & {macros['dOwlGd']} {macros['dciOwlGd']} \\\\", '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(b) VLM prompt, $\\Delta$AP on DUO (first 100 images)}} \\\\',
     "\\multicolumn{2}{l}{\\quad no ``Answer []'' / one class per prompt:} \\\\",
     f"\\quad Qwen2.5-VL-3B & {macros['dPromptThree']} / {macros['dSingleThree']} \\\\",
     f"\\quad Qwen2.5-VL-7B & {macros['dPromptSeven']} / {macros['dSingleSeven']} \\\\",
     f"\\quad Qwen3-VL-8B & {macros['dPromptQwen']} / {macros['dSingleQwen']} \\\\", '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(c) Qwen3-VL-8B minus Grounding DINO}} \\\\',
     f"DUO AP & {neg(df_['Grounding DINO minus Qwen3-VL-8B (full)'])} \\\\",
     f"DUO P / R, matched boxes & {macros['bdPGD']} / {macros['bdRGD']} \\\\",
     f"RUOD AP (subset) & {neg(R['differences']['Grounding DINO minus Qwen3-VL-8B (subset)'])} \\\\",
     f"RUOD P / R, matched boxes & {macros['rbdPGD']} / {macros['rbdRGD']} \\\\", '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(d) Qwen3-VL-8B minus OWLv2}} \\\\',
     f"DUO AP & {neg(df_['OWLv2 minus Qwen3-VL-8B (full)'])} \\\\",
     f"DUO P / R, matched boxes & {macros['bdPOWL']} / {macros['bdROWL']} \\\\",
     f"RUOD AP (subset) & {neg(R['differences']['OWLv2 minus Qwen3-VL-8B (subset)'])} \\\\",
     f"RUOD P / R, matched boxes & {macros['rbdPOWL']} / {macros['rbdROWL']} \\\\",
     '\\bottomrule', '\\end{tabular}',
     '\\caption{Evaluation choices. (a) The same Grounding DINO outputs scored as in its own COCO evaluation or through phrase decoding. (b) Change in AP '
     'when the sentence ``Answer [] if there are none\'\' is removed, or when the four classes are asked for one at a time (intervals in the text). '
     '(c, d) P, R: precision and recall. At a matched box budget (each detector keeps as many boxes per image as the VLM returned, '
     'all counting equally; IoU $\\geq$ 0.5, correct class) the VLM matches or exceeds them; at IoU $\\geq$ 0.75 its advantage shrinks (Section~\\ref{sec:choices}). Differences are Qwen3-VL-8B minus the detector.}',
     '\\label{tab:choices}', '\\end{table}']
(P / 'tab_choices.tex').write_text('\n'.join(t) + '\n')
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- numbers used in the text that come straight from the outputs
mac('rDedupMax', f"{max(abs(RM[k]['ap']['ap'] - RM[k]['ap_dedup']['ap']) for k in ['yolov8n', 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'rtdetr-l']):.3f}")
mac('rCropFishGD', pct(RC['gdino']['per_class']['fish']['acc']))
hw = [max(v['ci'][1] - v['acc'], v['acc'] - v['ci'][0]) for src in (CR['models'], RC) for k in ['gdino', 'owlv2', 'qwen25_3b', 'qwen25_7b', 'qwen3_8b'] for v in src[k]['per_class'].values()]
mac('cropMaxHalf', str(-int(-100 * max(hw) // 1)))          # largest distance from estimate to interval end, rounded up
GR = json.loads((O / 'gt' / 'ruod_instances_test.json').read_text()); rcn = {c['id']: c['name'] for c in GR['categories']}; rann = {a['id']: a for a in GR['annotations']}
SEA4 = {'holothurian', 'echinus', 'scallop', 'starfish'}; byimg = {}
for a_ in GR['annotations']:
    byimg.setdefault(a_['image_id'], set()).add(rcn[a_['category_id']])
mac('rSeaImages', N(sum(1 for s in byimg.values() if s & SEA4))); mac('rMixedImages', N(sum(1 for s in byimg.values() if s & SEA4 and s - SEA4)))
for k, nm, run in [('gdino', 'GD', 'grounding-dino_common'), ('owlv2', 'OWL', 'owlv2_common')]:
    PP = [json.loads(l) for l in (O / 'job8' / 'ruod' / run / 'predictions.jsonl').read_text().splitlines() if l.strip()]
    big = [p for p in PP if min(rann[p['ann_id']]['bbox'][2:]) >= 150]
    for grp, sel in (('Sea', [p for p in big if p['gt'] in SEA4]), ('Oth', [p for p in big if p['gt'] not in SEA4])):
        mac('rBig' + grp + nm, pct(sum(p['pred'] == p['gt'] for p in sel) / len(sel)))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))
_w = R['differences']['weakest fine-tuned minus best zero-shot detector']['models'][0]; mac('rApWeakest', f3(RM[_w]['ap']['ap']))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- DAI revision: score-calibration control (calib.py), matched budget at IoU 0.75 (budget_iou.py), same-image vocabulary test (vocab_summary.py)
CAL = json.loads((ROOT / 'analysis' / 'calib.json').read_text())
for bench, B in (('duo', 'Duo'), ('ruod', 'Ruod')):
    for model, nm in (('grounding-dino', 'GD'), ('owlv2', 'OWL')):
        c = CAL[f'{bench}:{model}']
        mac(f'cal{B}Raw{nm}', pct(c['raw']['seafloor'])); mac(f'cal{B}{nm}', pct(c['calibrated']['seafloor'])); mac(f'cal{B}ci{nm}', pci(c['calibrated']['seafloor_ci']))
        mac(f'cal{B}Top{nm}', pct(c['calibrated']['top_share_seafloor']))
        mac(f'cal{B}TopLab{nm}', {'holothurian': 'sea cucumber', 'echinus': 'sea urchin'}.get(c['calibrated']['top_answer_seafloor'], c['calibrated']['top_answer_seafloor']))
        if c['raw']['other'] is not None:
            mac(f'cal{B}OthRaw{nm}', pct(c['raw']['other'])); mac(f'cal{B}Oth{nm}', pct(c['calibrated']['other']))
BI = json.loads((ROOT / 'analysis' / 'budget_iou.json').read_text())
for k, nm in (('gdino', 'GD'), ('owlv2', 'OWL'), ('yolo11n', 'YoloN'), ('qwen3_8b', 'QwenVL')):     # share of correct boxes at IoU 0.5 that also reach 0.75
    mac('keepTight' + nm, pct(BI['0.75'][k]['precision'] / BI['0.5'][k]['precision']))
for k, nm in (('gdino', 'GD'), ('owlv2', 'OWL')):
    for m, mm in (('precision', 'P'), ('recall', 'R')):
        v = BI['0.75'][f'qwen3_minus_{k}_{m}']; mac(f'btd{mm}{nm}', sg(v['diff'])); mac(f'btdci{mm}{nm}', ci(v['ci']))
VP = ROOT / 'analysis' / 'vocab.json'
VC = json.loads(VP.read_text()) if VP.exists() else {}
if VC:
    for run, nm in (('gdino_duo4_gpu', 'GDfour'), ('owlv2_duo4_gpu', 'OWLfour'), ('gdinolatin_duo4_gpu', 'GDlatinfour'), ('gdino_ruod10', 'GDten'),
                    ('owlv2_ruod10', 'OWLten'), ('gdino_duo4rev', 'GDrev')):
        if run in VC:
            v = VC[run]; mac('vTopLab' + nm, v['top']); mac('vTop' + nm, pct(v['top_share'])); mac('vTopci' + nm, pci(v['top_ci']))
            mac('vAcc' + nm, pct(v['accuracy'])); mac('vAccci' + nm, pci(v['accuracy_ci']))
            if 'box_covers_object' in v:
                mac('vCov' + nm, pct(v['box_covers_object'])); mac('vCovci' + nm, pci(v['box_covers_ci']))
                mac('vCovTop' + nm, pct(v['covered']['top_share'])); mac('vCovCor' + nm, pct(v['covered']['correct'])); mac('vCovCorci' + nm, pci(v['covered']['correct_ci']))
    mac('vCrops', N(next(iter(VC.values()))['crops']))
_rev = ROOT / 'analysis' / 'vocab_cpu' / 'gdino_duo4rev_pc60.jsonl'
if _rev.exists():                                            # reversed caption order: crops with the same answer as the original order
    _ID = {'sea cucumber': 'holothurian', 'sea urchin': 'echinus', 'scallop': 'scallop', 'starfish': 'starfish'}
    _g = {json.loads(l)['ann_id']: json.loads(l)['pred'] for l in (O / 'job8' / 'grounding-dino_common' / 'predictions.jsonl').read_text().splitlines()}
    _r = [json.loads(l) for l in _rev.read_text().splitlines() if l.strip()]
    mac('vRevSame', N(sum(_ID[r['pred']] == _g[r['ann_id']] for r in _r))); mac('vRevN', N(len(_r)))
RP = ROOT / 'analysis' / 'vocab_repro.json'
if RP.exists():                                              # CPU code against the GPU crop test, original class names
    VR = json.loads(RP.read_text()); mac('vReproCrops', str(min(v['crops'] for v in VR.values())))
    mac('vReproSame', 'all' if all(v['same_answer'] == v['crops'] for v in VR.values()) else 'not all')
    mac('vReproDiff', f"{max(v['max_abs_score_diff'] for v in VR.values()):.3f}")
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- second review: TIDE (tide_run.py), strict naming and prompt check (review_extras.py), RUOD full-image labels, truncation
TD = {f.stem: json.loads(f.read_text()) for f in (ROOT / 'analysis' / 'tide').glob('*.json')}
f1 = lambda x: f'{x:.1f}'
for key, nm in (('duo_gdino', 'DuoGD'), ('duo_owlv2', 'DuoOWL'), ('duo_qwen3_8b', 'DuoQwenVL'), ('duo_yolo11n', 'DuoYoloN'),
                ('ruod_gdino', 'RuodGD'), ('ruod_owlv2', 'RuodOWL'), ('ruod_yolo11n', 'RuodYoloN'), ('duo_gdino_one_label', 'DuoGDone'), ('ruod_gdino_one_label', 'RuodGDone')):
    e = TD[key]['errors']; mac('tideAp' + nm, f1(TD[key]['ap50']))
    for err, en in (('Cls', 'Cls'), ('Loc', 'Loc'), ('Bkg', 'Bkg'), ('Miss', 'Miss')):
        mac(f'tide{en}{nm}', f1(e[err]))
    mac('tideOtherMax' + nm, f1(max(v for k, v in e.items() if k != 'Cls')))
mac('tideRuodImages', N(TD['ruod_gdino']['images']))
t = ['\\begin{table}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{2.2pt}', '\\begin{tabular}{lrrrrr|rrrr}', '\\toprule',
     ' & \\multicolumn{5}{c|}{DUO} & \\multicolumn{4}{c}{RUOD picking} \\\\',
     'Model & AP50 & Cls & Loc & Bkg & Miss & Cls & Loc & Bkg & Miss \\\\', '\\midrule']
for nm, lab in (('YoloN', 'YOLO11n'), ('GD', 'G.\\ DINO'), ('OWL', 'OWLv2'), ('QwenVL', 'Qwen3-VL')):
    d = [macros[f'tide{x}Duo{nm}'] for x in ('Ap', 'Cls', 'Loc', 'Bkg', 'Miss')]
    r = [macros[f'tide{x}Ruod{nm}'] for x in ('Cls', 'Loc', 'Bkg', 'Miss')] if f'tideClsRuod{nm}' in macros else ['--'] * 4
    t.append(f"{lab} & " + ' & '.join(d) + ' & ' + ' & '.join(r) + ' \\\\')
t += ['\\bottomrule', '\\end{tabular}',
      '\\caption{TIDE error breakdown \\citep{bolya2020tide}: AP50 and the AP50 gained by fixing each error type (classification, localization, '
      f"background, missed; duplicate and combined errors are at most {-(-10 * max(max(TD[k]['errors']['Both'], TD[k]['errors']['Dupe']) for k in TD if 'one_label' not in k) // 1) / 10:.1f} for every model). RUOD: the {macros['tideRuodImages']} test images that contain "
      'picking classes, with all their boxes. Qwen3-VL-8B was run only on a RUOD subset. Rows are not fully comparable across model types: Qwen3-VL-8B\'s boxes all score 1.0, so its values depend on tie order, and detectors that keep many low-scoring boxes have few missed errors by construction.}', '\\label{tab:tide}', '\\end{table}']
(P / 'tab_tide2.tex').write_text('\n'.join(t) + '\n')
RX = json.loads((ROOT / 'analysis' / 'review_extras.json').read_text())
for k, nm in (('gdino', 'GD'), ('owlv2', 'OWL'), ('qwen3_8b', 'QwenVL'), ('yolo11n', 'YoloN')):
    mac('strictNamed' + nm, pct(RX['strict_named'][k])); mac('urchBest' + nm, pct(RX['urchin_best_box_correct_of_found'][k]))
for k, nm in (('gdino', 'GD'), ('owlv2', 'OWL')):
    fnd = json.loads((ROOT / 'analysis' / 'e1.json').read_text())['found_named'][k]['top100_iou50']['all']['found']; mac('missed' + nm, pct(1 - fnd)); mac('misnamedStrict' + nm, pct(fnd - RX['strict_named'][k]))
NE = RX['noempty_first100']
for k, nm in (('qwen25_3b', 'QThree'), ('qwen25_7b', 'QSeven')):
    mac('neFoundDef' + nm, pct(NE[k]['found'])); mac('neFound' + nm, pct(NE[k + '_noempty100']['found']))
for k, nm in (('gdino', 'GD'), ('owlv2', 'OWL')):
    L = RM[k]['found_seafloor_labels']['by_label']; tot = sum(L.values()); top2 = sorted(L.items(), key=lambda kv: -kv[1])[:2]
    for i, (lab, n) in enumerate(top2):
        mac(f'rFullLab{"AB"[i]}{nm}', lab); mac(f'rFullShare{"AB"[i]}{nm}', pct(n / tot))
_n, _e, _c = replies('qwen3-vl-8b_common'); mac('pctCutQwenVL', pct(_c / _n))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- Table: same DUO crops, different class lists (vocab.json)
_vhw = str(-int(-100 * max(max(abs(x[c][i] - x[m]) for m, c in (('top_share', 'top_ci'), ('accuracy', 'accuracy_ci'), ('box_covers_object', 'box_covers_ci')) if m in x for i in (0, 1)) for x in VC.values()) // 1))          # rounded up
LATIN = {'sea cucumber': 'holothurian', 'sea urchin': 'echinus'}
t = ['\\begin{table}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{llrrr}', '\\toprule',
     'Class list & Default & Share & Acc. & On obj. \\\\', '\\midrule', '\\multicolumn{5}{l}{\\emph{Grounding DINO}} \\\\']
rows = [('gdino_duo4_gpu', 'DUO, 4 names', False), ('gdino_duo4rev', 'DUO, reversed', False), ('gdinolatin_duo4_gpu', 'DUO, Latin names', True),
        ('gdino_ruod10', "RUOD's 10 names", False), None, ('owlv2_duo4_gpu', 'DUO, 4 names', False), ('owlv2latin_duo4_gpu', 'DUO, Latin names', True),
        ('owlv2_ruod10', "RUOD's 10 names", False)]
for r in rows:
    if r is None:
        t += ['\\midrule', '\\multicolumn{5}{l}{\\emph{OWLv2}} \\\\']; continue
    v = VC[r[0]]; lab = LATIN.get(v['top'], v['top']) if r[2] else v['top']
    cov = pct(v['box_covers_object']) if 'box_covers_object' in v else '--'
    t.append(f"{r[1]} & {lab} & {pct(v['top_share'])} & {pct(v['accuracy'])} & {cov} \\\\")
t += ['\\bottomrule', '\\end{tabular}',
      f"\\caption{{The same {macros['vCrops']} DUO crops (60 per class) under different class lists. Share: crops (\\%) given the default answer; "
      'Acc.: mean accuracy over the four classes (\\%, chance 25); On obj.: crops (\\%) whose highest-scoring box covers the object '
      f"(IoU $\\geq$ 0.5), recorded only in the CPU runs; the reversed run gives the same answers as the original order, so its value also describes the four-name list. 95\\% intervals are within $\\pm${_vhw} points.}}", '\\label{tab:vocab}', '\\end{table}']
(P / 'tab_vocab.tex').write_text('\n'.join(t) + '\n')

ON = json.loads((ROOT / 'analysis' / 'owl_nms.json').read_text())
for b, B in (('duo', 'Duo'), ('ruod', 'Ruod')):
    mac(f'owlAp{B}', f3(ON[b]['ap'])); mac(f'owlNms{B}', f3(ON[b]['ap_nms']))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- third review: wrong-label concentration in full images, full-image class-list test (full_labels.py)
FL = ROOT / 'analysis' / 'full_labels'
SHORT2 = {'holothurian': 'sea cucumber', 'echinus': 'sea urchin'}
for b, B in (('duo', 'Duo'), ('ruod', 'Ruod')):
    for m, nm in (('gdino', 'GD'), ('owlv2', 'OWL')):
        f = FL / f'wrong_{b}_{m}.json'
        if f.exists():
            w = json.loads(f.read_text()); mac(f'wTop{B}{nm}', pct(w['top_wrong_share'])); mac(f'wTopTwo{B}{nm}', pct(w['top2_wrong_share']))
            mac(f'wTopLab{B}{nm}', SHORT2.get(w['top_wrong'], w['top_wrong']))
if (FL / 'vocab.json').exists():
    FV = json.loads((FL / 'vocab.json').read_text()); mac('fvImages', N(FV['images'])); mac('fvObjects', N(FV['objects']))
    for m, nm in (('gdino', 'GD'), ('owlv2', 'OWL')):
        for v, vn in (('duo4', 'four'), ('ruod10', 'ten')):
            r = FV[m][v]; t_ = f'{nm}{vn}'
            mac('fvFound' + t_, pct(r['found'])); mac('fvNamed' + t_, pct(r['named_strict'])); mac('fvTopLab' + t_, SHORT2.get(r['top_wrong'], r['top_wrong']))
            mac('fvTop' + t_, pct(r['top_wrong_share'])); mac('fvCls' + t_, f1(r['tide']['errors']['Cls'])); mac('fvAp' + t_, f1(r['tide']['ap50']))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))
