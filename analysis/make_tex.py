"""Write every number, table and the figure used by the paper from the analysis outputs (no hand-typed results).

Inputs: analysis/assess.json, analysis/e1.json, gpu_job/outputs/*/metrics.json and replies.jsonl.
Outputs: paper/numbers.tex (macros), paper/tab_*.tex, paper/fig_found_named.pdf (text converted to outlines).
"""
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'; O = ROOT / 'gpu_job' / 'outputs'
A = json.loads((ROOT / 'analysis' / 'assess.json').read_text()); E = json.loads((ROOT / 'analysis' / 'e1.json').read_text())
Z = O / 'job5' / 'duo'
MET = {**{m: O / 'job7' / 'duo' / m / 'metrics.json' for m in ['yolov8n', 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'rtdetr-l']},
       'gdino': Z / 'grounding-dino_common_posmap/metrics.json', 'gdino_text': Z / 'grounding-dino_common/metrics.json',
       'owlv2': Z / 'owlv2_common/metrics.json', 'qwen25_3b': Z / 'qwen2.5-vl-3b_common/metrics.json', 'qwen25_7b': Z / 'qwen2.5-vl-7b_common/metrics.json',
       'qwen3_8b_100': Z / 'qwen3-vl-8b_common_first100/metrics.json'}
M = {k: json.loads(p.read_text()) for k, p in MET.items()}
GTD = json.loads((O / 'gt' / 'duo_instances_test.json').read_text())

f3 = lambda x: f'{x:.3f}'.replace('-', '$-$')
sg = lambda x: ('+' if x >= 0 else '$-$') + f'{abs(x):.3f}'
ci = lambda c: f'[{f3(c[0])}, {f3(c[1])}]'
pct = lambda x: f'{100 * x:.0f}'
pci = lambda c: f'[{100 * c[0]:.0f}, {100 * c[1]:.0f}]'
N = lambda n: f'{n:,}'.replace(',', '{,}')
macros = {}
def mac(name, val):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    assert name not in macros, name
    macros[name] = val


def replies(name):
    rs = [json.loads(l) for l in (Z / name / 'replies.jsonl').read_text().splitlines() if l.strip()]
    empty = sum(1 for r in rs if r['reply'].strip().strip('`').replace('json', '').strip() in ('[]', ''))
    cut = sum(1 for r in rs if r['reply'].strip() and not r['reply'].rstrip().rstrip('`').rstrip().endswith(']'))
    return len(rs), empty, cut


# ---- dataset
cnt = {}
for a in GTD['annotations']:
    cnt[a['category_id']] = cnt.get(a['category_id'], 0) + 1
cid = {c['name']: c['id'] for c in GTD['categories']}
mac('nTestImages', N(len(GTD['images']))); mac('nTestObjects', N(len(GTD['annotations'])))
for nm, key in [('holothurian', 'SeaCucumber'), ('echinus', 'SeaUrchin'), ('scallop', 'Scallop'), ('starfish', 'Starfish')]:
    mac('n' + key, N(cnt[cid[nm]]))
mac('shareSeaUrchin', pct(cnt[cid['echinus']] / len(GTD['annotations'])))

# ---- AP and differences
af, a100, df, d100 = A['ap_full'], A['ap_first100'], A['differences_full'], A['differences_first100']
for k, nm in [('yolov8n', 'YoloEightN'), ('yolo11n', 'YoloN'), ('yolo11l', 'YoloL'), ('gdino', 'GD'), ('gdino_text', 'GDtext'), ('owlv2', 'OWL'),
              ('qwen25_3b', 'QThree'), ('qwen25_7b', 'QSeven'), ('gdino_latin', 'GDlatin'), ('owlv2_latin', 'OWLlatin'), ('qwen25_7b_low', 'QSevenLow')]:
    mac('ap' + nm, f3(af[k]['ap'])); mac('ci' + nm, ci(af[k]['ci']))
for k, nm in [('qwen3_8b_100', 'QwenVL'), ('gdino', 'GD'), ('owlv2', 'OWL'), ('qwen25_7b', 'QSeven'), ('qwen25_3b', 'QThree'),
              ('qwen25_7b_noempty100', 'QSevenNoEmpty'), ('qwen25_3b_noempty100', 'QThreeNoEmpty')]:
    mac('apH' + nm, f3(a100[k]['ap'])); mac('ciH' + nm, ci(a100[k]['ci']))
for lab, nm in [('H2: weakest fine-tuned minus best zero-shot', 'Gap'), ('B-A6: own scoring minus phrase decoding', 'Decoding'),
                ('model difference OWLv2 minus Grounding DINO', 'OwlGd'), ('B-A1 Grounding DINO common minus Latin', 'LatinGd'),
                ('B-A1 OWLv2 common minus Latin', 'LatinOwl'), ('H4: Qwen2.5-VL 7B minus 3B', 'SevenThree'),
                ('B-A2: default minus low resolution', 'Resolution'), ('largest minus smallest YOLO11', 'YoloSize')]:
    mac('d' + nm, sg(df[lab]['diff'])); mac('dci' + nm, ci(df[lab]['ci']))
for lab, nm in [('H4: Qwen3-VL-8B minus Qwen2.5-VL-7B', 'QwenGen'), ('Grounding DINO minus Qwen3-VL-8B', 'GdQwen'), ('OWLv2 minus Qwen3-VL-8B', 'OwlQwen'),
                ('B-A5 7B: no-empty minus default prompt', 'PromptSeven'), ('B-A5 3B: no-empty minus default prompt', 'PromptThree')]:
    mac('d' + nm, sg(d100[lab]['diff'])); mac('dci' + nm, ci(d100[lab]['ci']))
s = A['spearman_coco_duo']; mac('rhoCocoDuo', f'{s["rho"]:.2f}'); mac('rhoCocoDuoCI', f'[{s["ci"][0]:.2f}, {s["ci"][1]:.2f}]')
mg, mg100 = A['merged_class_ap_full'], A['merged_class_ap_first100']
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL'), ('yolo11n', 'YoloN'), ('yolo11l', 'YoloL'), ('qwen25_7b', 'QSeven')]:
    mac('mg' + nm, f3(mg[k]['ap'])); mac('mgci' + nm, ci(mg[k]['ci']))
mac('mgQwenVL', f3(mg100['qwen3_8b_100']['ap'])); mac('mgciQwenVL', ci(mg100['qwen3_8b_100']['ci']))
pc = A['per_class_ap_full']
mac('scApGD', f3(pc['gdino']['holothurian']['ap'])); mac('scApciGD', ci(pc['gdino']['holothurian']['ci']))
mac('scApOWL', f3(pc['owlv2']['holothurian']['ap'])); mac('scApciOWL', ci(pc['owlv2']['holothurian']['ci']))
mac('scApYoloN', f3(pc['yolo11n']['holothurian']['ap']))

# ---- found / named (top 100, IoU 0.5)
fn = E['found_named']; S = 'top100_iou50'; HOLO, URCH, SCAL, STAR = '1', '2', '3', '4'
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL'), ('yolo11n', 'YoloN'), ('gdino_latin', 'GDlatin'), ('owlv2_latin', 'OWLlatin'), ('qwen3_8b_100', 'QwenVL'), ('qwen25_7b', 'QSeven')]:
    a = fn[k][S]['all']; h = fn[k][S][HOLO]
    mac('found' + nm, pct(a['found'])); mac('named' + nm, pct(a['named'])); mac('foundci' + nm, pci(a['found_ci'])); mac('namedci' + nm, pci(a['named_ci']))
    mac('scFound' + nm, pct(h['found'])); mac('scNamed' + nm, pct(h['named'])); mac('scFoundci' + nm, pci(h['found_ci'])); mac('scNamedci' + nm, pci(h['named_ci']))
L = E['labels']
def lab(k, gtc):
    return L[k][gtc]
mac('scUrchinGD', N(lab('gdino', 'holothurian')['echinus'])); mac('scFoundNGD', N(sum(lab('gdino', 'holothurian').values())))
mac('scStarOWL', N(lab('owlv2', 'holothurian')['starfish'])); mac('scFoundNOWL', N(sum(lab('owlv2', 'holothurian').values())))
mac('urchinAsHoloGDlatin', N(lab('gdino_latin', 'echinus')['holothurian'])); mac('urchinAsScalGDlatin', N(lab('gdino_latin', 'echinus')['scallop']))
mac('urchinAsHoloGDlatinPct', pct(lab('gdino_latin', 'echinus')['holothurian'] / cnt[cid['echinus']]))
mac('scStarOWLlatin', N(lab('owlv2_latin', 'holothurian')['starfish']))
mac('starAsUrchinGD', N(lab('gdino', 'starfish')['echinus'])); mac('starAsStarGD', N(lab('gdino', 'starfish')['starfish']))
mac('scalAsStarOWL', N(lab('owlv2', 'scallop')['starfish'])); mac('scalAsUrchinOWL', N(lab('owlv2', 'scallop')['echinus'])); mac('scalAsScalOWL', N(lab('owlv2', 'scallop')['scallop']))
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL')]:
    mac('scNamedTopTen' + nm, pct(fn[k]['top10_iou50'][HOLO]['named'])); mac('scFoundTopTen' + nm, pct(fn[k]['top10_iou50'][HOLO]['found']))
    mac('scNamedIouSeventy' + nm, pct(fn[k]['top100_iou70'][HOLO]['named'])); mac('scFoundIouSeventy' + nm, pct(fn[k]['top100_iou70'][HOLO]['found']))
mac('scObjectsHundred', N(fn['qwen3_8b_100'][S][HOLO]['objects']))

# ---- matched budget (first 100, Qwen3-VL-8B's distinct box count)
B = E['budget']['qwen3_count']['class_aware']
for k, nm in [('qwen3_8b_100', 'QwenVL'), ('gdino', 'GD'), ('owlv2', 'OWL'), ('yolo11n', 'YoloN')]:
    r = B[k]; mac('bP' + nm, f3(r['precision'])); mac('bR' + nm, f3(r['recall'])); mac('bPci' + nm, ci(r['precision_ci'])); mac('bRci' + nm, ci(r['recall_ci']))
    mac('bBoxes' + nm, f'{r["boxes_per_image"]:.1f}')
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL')]:
    for m, mm in [('precision', 'P'), ('recall', 'R')]:
        r = B[f'qwen3_minus_{k}_{m}']; mac('bd' + mm + nm, sg(r['diff'])); mac('bdci' + mm + nm, ci(r['ci']))

# ---- VLM replies
for name, nm in [('qwen2.5-vl-3b_common', 'QThree'), ('qwen2.5-vl-7b_common', 'QSeven'), ('qwen2.5-vl-3b_common_noempty_first100', 'QThreeNoEmpty'),
                 ('qwen2.5-vl-7b_common_noempty_first100', 'QSevenNoEmpty'), ('qwen3-vl-8b_common_first100', 'QwenVL'), ('qwen2.5-vl-7b_common_low', 'QSevenLow')]:
    n, e, c = replies(name); mac('nRep' + nm, N(n)); mac('nEmpty' + nm, N(e)); mac('nCut' + nm, N(c))
mac('pctEmptyQSeven', pct(replies('qwen2.5-vl-7b_common')[1] / replies('qwen2.5-vl-7b_common')[0]))
mac('nUnmatchedText', N(M['gdino_text']['unmatched_labels'])); mac('nBoxesText', N(M['gdino_text']['unmatched_labels'] + M['gdino_text']['detections']))
mac('pctUnmatchedText', pct(M['gdino_text']['unmatched_labels'] / (M['gdino_text']['unmatched_labels'] + M['gdino_text']['detections'])))
mac('nOutsideQwenVL', N(M['qwen3_8b_100']['boxes_outside_image']))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' +
                               ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- Table 1: all models
COCO = dict(yolov8n='37.3', yolo11n='39.5', yolo11s='47.0', yolo11m='51.5', yolo11l='53.4', **{'rtdetr-l': '53.0'})
rows = [('YOLOv8n', 'yolov8n'), ('YOLO11n', 'yolo11n'), ('YOLO11s', 'yolo11s'), ('YOLO11m', 'yolo11m'), ('YOLO11l', 'yolo11l'), ('RT-DETR-L', 'rtdetr-l'),
        None, ('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2'), None, ('Qwen2.5-VL-3B', 'qwen25_3b'), ('Qwen2.5-VL-7B', 'qwen25_7b'), ('Qwen3-VL-8B', 'qwen3_8b_100')]
groups = {'yolov8n': 'Fine-tuned on DUO', 'gdino': 'Zero-shot detectors', 'qwen25_3b': 'Zero-shot VLMs'}
TIME = json.loads((O / 'job9' / 'timing.json').read_text())['models']
def ms(k):
    if k in TIME: return f"{TIME[k]['wall_ms']:.0f}"
    d = {'gdino': 'grounding-dino_common_posmap', 'owlv2': 'owlv2_common', 'qwen25_3b': 'qwen2.5-vl-3b_common', 'qwen25_7b': 'qwen2.5-vl-7b_common', 'qwen3_8b_100': 'qwen3-vl-8b_common_first100'}[k]
    r = json.loads((Z / d / 'record.json').read_text()); v = 1000 * r['wall_s'] / M[k]['images']
    return f'{v:,.0f}'.replace(',', '{,}') + ('$^\\ddagger$' if k == 'gdino' else '')
t = ['\\begin{table*}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4.5pt}', '\\begin{tabular}{llccccccc}', '\\toprule',
     ' & & \\multicolumn{5}{c}{Full test set (\\nTestImages{} images)} & First 100 images & \\\\', '\\cmidrule(lr){3-7}\\cmidrule(lr){8-8}',
     'Model & COCO AP & AP (95\\% CI) & AP$_{50}$ & AR$_{100}$ & Merged AP & Sea cuc.\\ AP & AP (95\\% CI) & ms/image \\\\', '\\midrule']
for r in rows:
    if r is None:
        t.append('\\midrule'); continue
    name, k = r
    if k in groups:
        t.append(f'\\multicolumn{{9}}{{l}}{{\\textit{{{groups[k]}}}}} \\\\')
    if k == 'qwen3_8b_100':
        full = '\\multicolumn{5}{c}{not run on the full test set}'
    else:
        m = M[k]; mgv = f3(mg[k]['ap']) if k in mg else '--'
        sc = f3(m['per_class_AP']['holothurian']) if m.get('per_class_AP') else '0.000'
        full = f"{f3(af[k]['ap'])} {ci(af[k]['ci'])} & {f3(m.get('AP50', 0))} & {f3(m.get('AR100', 0))} & {mgv} & {sc}"
    t.append(f"{name} & {COCO.get(k, '--')} & {full} & {f3(a100[k]['ap'])} {ci(a100[k]['ci'])} & {ms(k)} \\\\")
t += ['\\bottomrule', '\\end{tabular}',
      '\\caption{Detection results on the DUO test set (COCO protocol, pycocotools). Intervals are 95\\% paired bootstrap intervals over test images. '
      'COCO AP is the published value for the fine-tuned architectures (Ultralytics). Merged AP treats all four classes as one, so a box counts '
      'regardless of its label. Qwen3-VL-8B was run only on the first 100 test images; the last column compares all models on those images. '
      'VLM boxes carry no confidence score and are all scored 1.0. Time per image on one RTX 5090: for the fine-tuned detectors, the median over 200 test '
      'images processed one at a time on an idle GPU, including pre- and post-processing; for the zero-shot models, wall-clock time over the whole run, '
      'including image loading. $^\\ddagger$Grounding DINO shared the GPU with another job, so its time is an upper bound. Qwen2.5-VL is fast because '
      'it mostly returns an empty answer.}', '\\label{tab:main}', '\\end{table*}']
(P / 'tab_main.tex').write_text('\n'.join(t) + '\n')

# ---- Table 2: what found sea cucumbers are called
def lrow(name, k):
    h = fn[k][S][HOLO]; l = lab(k, 'holothurian'); nf = sum(l.values())
    cells = ' & '.join(f'{100 * l[c] / nf:.0f}' if nf else '--' for c in ['holothurian', 'echinus', 'scallop', 'starfish'])
    return f'{name} & {pct(h["found"])} & {pct(h["named"])} & {cells} \\\\'
t = ['\\begin{table}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{2.8pt}', '\\begin{tabular}{lcccccc}', '\\toprule',
     ' & & & \\multicolumn{4}{c}{Label of best box (\\%)} \\\\', '\\cmidrule(lr){4-7}',
     'Model, names & Found & Named & Cuc. & Urch. & Scal. & Star. \\\\', '\\midrule',
     '\\multicolumn{7}{l}{Grounding DINO} \\\\', lrow('\\quad common', 'gdino'), lrow('\\quad Latin', 'gdino_latin'),
     '\\multicolumn{7}{l}{OWLv2} \\\\', lrow('\\quad common', 'owlv2'), lrow('\\quad Latin', 'owlv2_latin'),
     '\\midrule', lrow('YOLO11n, fine-tuned', 'yolo11n'), lrow('Qwen3-VL-8B$^\\dagger$', 'qwen3_8b_100'), '\\bottomrule', '\\end{tabular}',
     f'\\caption{{What happens to the \\nSeaCucumber{{}} sea cucumbers in the DUO test set. Found: \\% of objects overlapped (IoU $\\geq$ 0.5) by any of the '
     f'image\'s 100 highest-scoring boxes, of any class. Named: \\% overlapped by such a box with the correct class. The right block gives the class of the '
     f'best-overlapping box, as \\% of the found sea cucumbers. $^\\dagger$First 100 images only ({fn["qwen3_8b_100"][S][HOLO]["objects"]} sea cucumbers).}}', '\\label{tab:seacucumber}', '\\end{table}']
(P / 'tab_seacucumber.tex').write_text('\n'.join(t) + '\n')

# ---- Table 3: evaluation choices
apc = lambda d: f"{f3(d['ap'])} {ci(d['ci'])}"
t = ['\\begin{table}[t]', '\\centering', '\\small', '\\begin{tabular}{lc}', '\\toprule',
     'Setting & AP or difference (95\\% CI) \\\\', '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(a) Grounding DINO decoding, full test set}} \\\\',
     f"Own scoring & {apc(af['gdino'])} \\\\",
     f"Phrase decoding & {apc(af['gdino_text'])} \\\\",
     f"Own $-$ phrase decoding & {macros['dDecoding']} {macros['dciDecoding']} \\\\",
     f"OWLv2 $-$ G.\\ DINO & {macros['dOwlGd']} {macros['dciOwlGd']} \\\\", '\\midrule',
     '\\multicolumn{2}{l}{\\textit{(b) Prompt sentence, first 100 images}} \\\\',
     f"Qwen2.5-VL-3B, with & {apc(a100['qwen25_3b'])} \\\\",
     f"\\quad without & {apc(a100['qwen25_3b_noempty100'])} \\\\",
     f"\\quad without $-$ with & {macros['dPromptThree']} {macros['dciPromptThree']} \\\\",
     f"Qwen2.5-VL-7B, with & {apc(a100['qwen25_7b'])} \\\\",
     f"\\quad without & {apc(a100['qwen25_7b_noempty100'])} \\\\",
     f"\\quad without $-$ with & {macros['dPromptSeven']} {macros['dciPromptSeven']} \\\\",
     '\\bottomrule', '\\end{tabular}',
     '\\caption{Two evaluation choices. (a) The same Grounding DINO forward pass, scored as in its own COCO evaluation or through phrase decoding '
     'followed by matching phrases to class names; the difference exceeds the difference between the two open-vocabulary detectors. '
     '(b) Removing the sentence ``Answer [] if there are none\'\' from the prompt turns almost-always-empty answers into non-empty ones '
     '(Section~\\ref{sec:prompt}) but barely changes AP.}',
     '\\label{tab:choices}', '\\end{table}']
(P / 'tab_choices.tex').write_text('\n'.join(t) + '\n')

# ---- Table 4: AP vs matched box budget
neg = lambda d: f"{sg(-d['diff'])} [{f3(-d['ci'][1])}, {f3(-d['ci'][0])}]"   # reverse a difference and its interval
t = ['\\begin{table*}[t]', '\\centering', '\\small', '\\begin{tabular}{lcccc}', '\\toprule',
     'Model & AP (95\\% CI) & Precision (95\\% CI) & Recall (95\\% CI) & Boxes/image \\\\', '\\midrule']
for name, k, nm in [('Qwen3-VL-8B', 'qwen3_8b_100', 'QwenVL'), ('OWLv2', 'owlv2', 'OWL'), ('Grounding DINO', 'gdino', 'GD'), ('YOLO11n (fine-tuned)', 'yolo11n', 'YoloN')]:
    r = B[k]
    t.append(f"{name} & {f3(a100[k]['ap'])} {ci(a100[k]['ci'])} & {f3(r['precision'])} {ci(r['precision_ci'])} & {f3(r['recall'])} {ci(r['recall_ci'])} & {r['boxes_per_image']:.1f} \\\\")
t += ['\\midrule',
      f"Qwen3-VL-8B $-$ OWLv2 & {neg(d100['OWLv2 minus Qwen3-VL-8B'])} & {macros['bdPOWL']} {macros['bdciPOWL']} & {macros['bdROWL']} {macros['bdciROWL']} & \\\\",
      f"Qwen3-VL-8B $-$ G.\\ DINO & {neg(d100['Grounding DINO minus Qwen3-VL-8B'])} & {macros['bdPGD']} {macros['bdciPGD']} & {macros['bdRGD']} {macros['bdciRGD']} & \\\\",
      '\\bottomrule', '\\end{tabular}',
      '\\caption{Ranking by AP against ranking at a matched box budget (first 100 test images). AP uses each model\'s full ranked output. '
      'For precision and recall (IoU $\\geq$ 0.5, correct class), each detector keeps on every image only as many of its highest-scoring distinct boxes as '
      'Qwen3-VL-8B returned there, and all boxes count equally. The last two rows give Qwen3-VL-8B minus the detector, with 95\\% paired bootstrap intervals.}',
      '\\label{tab:budget}', '\\end{table*}']
(P / 'tab_budget.tex').write_text('\n'.join(t) + '\n')

# ---- Appendix tables: sensitivity, TIDE, fine-tuned details
hw = []
t = ['\\begin{table}[!ht]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{2.8pt}', '\\begin{tabular}{llcccc}', '\\toprule',
     ' & & \\multicolumn{2}{c}{All objects} & \\multicolumn{2}{c}{Sea cucumbers} \\\\', '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
     'Model & Boxes, IoU & Found & Named & Found & Named \\\\', '\\midrule']
for name, k in [('YOLO11n', 'yolo11n'), ('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2')]:
    for st, sl in [('top100_iou50', '100, 0.5'), ('top10_iou50', '10, 0.5'), ('top100_iou70', '100, 0.7')]:
        a = fn[k][st]['all']; h = fn[k][st][HOLO]
        hw += [(a[q + '_ci'][1] - a[q + '_ci'][0]) / 2 for q in ('found', 'named')] + [(h[q + '_ci'][1] - h[q + '_ci'][0]) / 2 for q in ('found', 'named')]
        t.append(f"{name if st == 'top100_iou50' else ''} & {sl} & {pct(a['found'])} & {pct(a['named'])} & {pct(h['found'])} & {pct(h['named'])} \\\\")
t += ['\\bottomrule', '\\end{tabular}', '\\caption{Found and named shares (\\%, full test set) when only the 10 highest-scoring boxes per image count, '
      f'or with IoU $\\geq$ 0.7. All 95\\% bootstrap intervals are within $\\pm${100 * max(hw):.0f} percentage points.}}', '\\label{tab:sensitivity}', '\\end{table}']
(P / 'tab_sensitivity.tex').write_text('\n'.join(t) + '\n')

t = ['\\begin{table}[!ht]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{lcccccc}', '\\toprule',
     'Model & Cls & Loc & Both & Dupe & Bkg & Miss \\\\', '\\midrule']
for name, k in [('YOLOv8n', 'yolov8n'), ('YOLO11n', 'yolo11n'), ('YOLO11s', 'yolo11s'), ('YOLO11m', 'yolo11m'), ('YOLO11l', 'yolo11l'), ('RT-DETR-L', 'rtdetr-l'),
                ('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2'), ('Qwen2.5-VL-7B', 'qwen25_7b'), ('Qwen3-VL-8B$^\\dagger$', 'qwen3_8b_100')]:
    te = M[k]['tide_main_errors']
    t.append(name + ' & ' + ' & '.join(f'{te[c]:.1f}' for c in ['Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss']) + ' \\\\')
t += ['\\bottomrule', '\\end{tabular}', '\\caption{TIDE error breakdown: AP$_{50}$ points recovered by fixing each error type '
      '(Bolya et al.\\ 2020). $^\\dagger$First 100 images. The fixes are not additive, and the open-vocabulary detectors output several class labels '
      'for the same box, which inflates their Cls value; we therefore use TIDE only as supporting evidence.}', '\\label{tab:tide}', '\\end{table}']
(P / 'tab_tide.tex').write_text('\n'.join(t) + '\n')


# ---- derived numbers used in the text
mac('nImagesWithObjects', N(len({a['image_id'] for a in GTD['annotations']})))
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL')]:
    a = fn[k][S]['all']; mac('lossNaming' + nm, f"{100 * a['found'] - 100 * a['named']:.0f}")
mac('mgYoloEightN', f3(mg['yolov8n']['ap']))
ny = [fn['yolo11n'][S][c]['named'] / fn['yolo11n'][S][c]['found'] for c in (HOLO, URCH, SCAL, STAR)]
mac('namedOfFoundMinYoloN', pct(min(ny))); mac('namedOfFoundMaxYoloN', pct(max(ny)))
tm = [json.loads((O / 'job4' / m / 'record.json').read_text())['wall_s'] / 60 for m in COCO]
mac('trainMinMinutes', f'{min(tm):.0f}'); mac('trainMaxMinutes', f'{max(tm):.0f}')
for r, nm in [('qwen3-vl-8b_common_first100', 'QwenVL'), ('owlv2_common', 'OWL'), ('qwen2.5-vl-7b_common', 'QSeven')]:
    d = json.loads((Z / r / 'record.json').read_text()); m = json.loads((Z / r / 'metrics.json').read_text())
    mac('sec' + nm, f"{d['wall_s'] / m['images']:.1f}")
G2 = E['budget']['gt_count']['class_aware']
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL'), ('yolo11n', 'YoloN')]:
    mac('gtbP' + nm, f3(G2[k]['precision'])); mac('gtbR' + nm, f3(G2[k]['recall']))
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' +
                               ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))


# ---- box-crop naming test (Job 8)
CR = json.loads((ROOT / 'analysis' / 'crops.json').read_text()); CM = CR['models']
mac('nCrops', N(CR['crops'])); mac('nCropsScallop', N(CR['per_class_n']['scallop'])); mac('nCropsOther', N(CR['per_class_n']['holothurian']))
for k, nm in [('gdino', 'GD'), ('owlv2', 'OWL'), ('qwen25_3b', 'QThree'), ('qwen25_7b', 'QSeven'), ('qwen3_8b', 'QwenVL'), ('gdino_latin', 'GDlatin'), ('owlv2_latin', 'OWLlatin')]:
    m = CM[k]; mac('crop' + nm, pct(m['mean_class_accuracy'])); mac('cropci' + nm, pci(m['mean_ci']))
    mac('cropDefault' + nm, pct(m['share_most_common']))
    for c, cn in [('holothurian', 'Cuc'), ('echinus', 'Urch'), ('scallop', 'Scal'), ('starfish', 'Star')]:
        mac('crop' + cn + nm, pct(m['per_class'][c]['acc']))
mac('cropScalMax', pct(max(CM[k]['per_class']['scallop']['acc'] for k in ['gdino', 'owlv2', 'qwen25_3b', 'qwen25_7b', 'qwen3_8b'])))
for lab, nm in [('Qwen3-VL-8B minus Grounding DINO', 'GD'), ('Qwen3-VL-8B minus OWLv2', 'OWL'), ('Qwen3-VL-8B minus Qwen2.5-VL-7B', 'QSeven')]:
    d = CR['differences'][lab]; mac('cropd' + nm, f"{100 * d['diff']:+.0f}"); mac('cropdci' + nm, pci(d['ci']))
mac('msYoloMin', f"{min(v['wall_ms'] for v in TIME.values()):.0f}"); mac('msYoloMax', f"{max(v['wall_ms'] for v in TIME.values()):.0f}")
NAMEC = [('holothurian', 'Sea cuc.'), ('echinus', 'Sea urchin'), ('scallop', 'Scallop'), ('starfish', 'Starfish')]
chw = []
t = ['\\begin{table}[!ht]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{2.2pt}', '\\begin{tabular}{lcccccc}', '\\toprule',
     'Model & Cuc. & Urch. & Scal. & Star. & Mean & Most frequent \\\\', '\\midrule']
for name, k in [('Grounding DINO', 'gdino'), ('\\quad Latin names', 'gdino_latin'), ('OWLv2', 'owlv2'), ('\\quad Latin names', 'owlv2_latin'),
                ('Qwen2.5-VL-3B', 'qwen25_3b'), ('Qwen2.5-VL-7B', 'qwen25_7b'), ('Qwen3-VL-8B', 'qwen3_8b')]:
    m = CM[k]; mc = m['most_common_label']; lab = mc[:5] + '.' if 'Latin' in name and mc in ('holothurian', 'echinus') else dict(holothurian='cuc.', echinus='urch.', scallop='scal.', starfish='star.')[mc]
    chw += [(m['per_class'][c]['ci'][1] - m['per_class'][c]['ci'][0]) / 2 for c, _ in NAMEC]
    t.append(f"{name} & " + ' & '.join(pct(m['per_class'][c]['acc']) for c, _ in NAMEC) +
             f" & {pct(m['mean_class_accuracy'])} & {lab} {pct(m['share_most_common'])}\\% \\\\")
t += ['\\bottomrule', '\\end{tabular}', '\\caption{Box-crop naming test: \\% of crops named correctly per class (cucumber, urchin, scallop, starfish) and the mean over classes; chance is 25\\%. '
      f'Per-class 95\\% bootstrap intervals are within $\\pm${100 * max(chw):.0f} points, intervals of the mean within $\\pm${100 * max((CM[k]["mean_ci"][1] - CM[k]["mean_ci"][0]) / 2 for k in CM):.0f}. '
      'The last column gives the answer a model gives most often over all crops, and its share (holot.\\ = holothurian).}', '\\label{tab:crops}', '\\end{table}']
(P / 'tab_crops.tex').write_text('\n'.join(t) + '\n')
(P / 'numbers.tex').write_text('% Generated by analysis/make_tex.py; do not edit.\n' +
                               ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

# ---- Figure: found vs named per class
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
                     'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
models = [('YOLO11n (fine-tuned)', 'yolo11n', '#009E73'), ('Grounding DINO', 'gdino', '#0072B2'), ('OWLv2', 'owlv2', '#E69F00')]
classes = [(HOLO, 'Sea cucumber'), (URCH, 'Sea urchin'), (SCAL, 'Scallop'), (STAR, 'Starfish')]
fig, ax = plt.subplots(figsize=(3.3, 1.75)); w = 0.26
for j, (lbl, k, col) in enumerate(models):
    for i, (c, _) in enumerate(classes):
        r = fn[k][S][c]; x = i + (j - 1) * w
        ax.bar(x, 100 * r['found'], w * 0.92, color=col, alpha=0.32, linewidth=0)
        ax.bar(x, 100 * r['named'], w * 0.92, color=col, linewidth=0)
        for v, cc in ((r['found'], r['found_ci']), (r['named'], r['named_ci'])):
            ax.plot([x, x], [100 * cc[0], 100 * cc[1]], color='#333333', lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{n}\n(n={fn["yolo11n"][S][c]["objects"]:,})' for c, n in classes])
ax.set_ylabel('% of objects'); ax.set_ylim(0, 100); ax.yaxis.grid(True, lw=0.3, color='#cccccc'); ax.set_axisbelow(True)
from matplotlib.patches import Patch
handles = [Patch(color=col, label=lbl) for lbl, _, col in models] + [Patch(color='#777777', alpha=0.32, label='found'), Patch(color='#777777', label='named')]
ax.legend(handles=handles, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.30), handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.2)
raw = P / 'fig_found_named.raw.pdf'; fig.savefig(raw); plt.close(fig)
subprocess.run(['gs', '-q', '-o', str(P / 'fig_found_named.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', str(raw)], check=True)
raw.unlink()
print(len(macros), 'macros; tables and figure written')
