"""Evidence for the paper's claims, from the saved detections only (no GPU).

1. Paired bootstrap (500 draws over test images, seed 20261119) for each AP and each planned comparison.
2. Merged-class AP (all classes as one) and per-class AP, with intervals.
Per-object measures (found / named) and the matched box budget are in e1.py.
Run: python analysis/assess.py   (writes analysis/assess.json)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_bootstrap import load_gt, merge_dets, Matched, self_check, draws

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'gpu_job' / 'outputs'
GT = O / 'gt' / 'duo_instances_test.json'
COCO_AP = dict(yolo11n=39.5, yolo11s=47.0, yolo11m=51.5, yolo11l=53.4, yolov8n=37.3, **{'rtdetr-l': 53.0})
FT = list(COCO_AP)
Z = O / 'job5' / 'duo'
RUNS = {**{m: O / 'job7' / 'duo' / m for m in FT},
        'gdino': Z / 'grounding-dino_common_posmap', 'gdino_text': Z / 'grounding-dino_common', 'gdino_latin': Z / 'grounding-dino_latin_posmap',
        'owlv2': Z / 'owlv2_common', 'owlv2_latin': Z / 'owlv2_latin', 'qwen25_3b': Z / 'qwen2.5-vl-3b_common', 'qwen25_7b': Z / 'qwen2.5-vl-7b_common',
        'qwen25_7b_low': Z / 'qwen2.5-vl-7b_common_low', 'qwen25_3b_noempty100': Z / 'qwen2.5-vl-3b_common_noempty_first100',
        'qwen25_7b_noempty100': Z / 'qwen2.5-vl-7b_common_noempty_first100', 'qwen3_8b_100': Z / 'qwen3-vl-8b_common_first100',
        'qwen3_8b': Z / 'qwen3-vl-8b_common', 'qwen25_3b_single100': Z / 'qwen2.5-vl-3b_common_single_first100',
        'qwen25_7b_single100': Z / 'qwen2.5-vl-7b_common_single_first100', 'qwen3_8b_single100': Z / 'qwen3-vl-8b_common_single_first100',
        'qwen3_8b_noempty100': Z / 'qwen3-vl-8b_common_noempty_first100'}
FIRST100_ONLY = {'qwen25_3b_noempty100', 'qwen25_7b_noempty100', 'qwen3_8b_100', 'qwen25_3b_single100', 'qwen25_7b_single100', 'qwen3_8b_single100', 'qwen3_8b_noempty100'}


def ci(x):
    x = np.asarray(x, float); return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]   # draws without the class are skipped


def iou(a, b):          # a: (n,4) xywh, b: (m,4) xywh -> (n,m)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1) - np.maximum(ax0[:, None], bx0), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1) - np.maximum(ay0[:, None], by0), 0, None)
    inter = iw * ih; return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter + 1e-9)


def by_image(dets):
    d = {}
    for x in dets:
        d.setdefault(x['image_id'], []).append(x)
    return d


def main():
    g = json.loads(GT.read_text()); all_ids = sorted(im['id'] for im in g['images'])
    first100 = [im['id'] for im in g['images'][:100]]; first100_sorted = sorted(first100)
    G = load_gt(g); Gm = load_gt(g, merge_classes=True)
    out = dict(protocol='paired bootstrap over test images, 500 draws, seed 20261119; 95% percentile intervals')
    # one run at a time, keeping only the detections needed later (memory)
    KEEP = {'qwen3_8b_100', 'gdino', 'owlv2'}; MERGE = set(RUNS) - FIRST100_ONLY
    W_all, W_100 = draws(len(all_ids)), draws(100)
    full, sub, mg, dets, chk = {}, {}, {}, {}, {}
    s100 = set(first100)
    for k, path in RUNS.items():
        d = json.loads((path / 'detections.json').read_text())
        if k not in FIRST100_ONLY:
            full[k] = Matched(G, d, all_ids)
            if k in {'yolo11n', 'gdino', 'owlv2', 'qwen25_7b'}:
                chk[k] = self_check(G, d, all_ids, full[k])
        sub[k] = Matched(G, [x for x in d if x['image_id'] in s100], first100_sorted)
        if k in MERGE:
            mg[k] = Matched(Gm, merge_dets(d), all_ids)
        if k in KEEP:
            dets[k] = d
        del d
        print('matched', k, flush=True)
    out['self_check_pycocotools_vs_ours'] = chk
    assert all(abs(a - b) < 1e-6 for a, b in chk.values()), chk
    boot_full = {k: np.array([m.ap(w) for w in W_all]) for k, m in full.items()}
    boot_sub = {k: np.array([m.ap(w) for w in W_100]) for k, m in sub.items()}
    out['ap_full'] = {k: dict(ap=full[k].ap(), ci=ci(boot_full[k])) for k in full}
    out['ap_first100'] = {k: dict(ap=sub[k].ap(), ci=ci(boot_sub[k])) for k in sub}
    diffs_full = [('yolov8n', 'owlv2', 'H2: weakest fine-tuned minus best zero-shot'), ('gdino', 'gdino_text', 'B-A6: own scoring minus phrase decoding'),
                  ('owlv2', 'gdino', 'model difference OWLv2 minus Grounding DINO'), ('gdino', 'gdino_latin', 'B-A1 Grounding DINO common minus Latin'),
                  ('owlv2', 'owlv2_latin', 'B-A1 OWLv2 common minus Latin'), ('qwen25_7b', 'qwen25_3b', 'H4: Qwen2.5-VL 7B minus 3B'),
                  ('qwen25_7b', 'qwen25_7b_low', 'B-A2: default minus low resolution'), ('yolo11l', 'yolo11n', 'largest minus smallest YOLO11'),
                  ('qwen3_8b', 'qwen25_7b', 'H4 full: Qwen3-VL-8B minus Qwen2.5-VL-7B'), ('owlv2', 'qwen3_8b', 'OWLv2 minus Qwen3-VL-8B (full)'),
                  ('gdino', 'qwen3_8b', 'Grounding DINO minus Qwen3-VL-8B (full)')]
    diffs_sub = [('qwen3_8b_100', 'qwen25_7b', 'H4: Qwen3-VL-8B minus Qwen2.5-VL-7B'), ('gdino', 'qwen3_8b_100', 'Grounding DINO minus Qwen3-VL-8B'),
                 ('owlv2', 'qwen3_8b_100', 'OWLv2 minus Qwen3-VL-8B'), ('qwen25_7b_noempty100', 'qwen25_7b', 'B-A5 7B: no-empty minus default prompt'),
                 ('qwen25_3b_noempty100', 'qwen25_3b', 'B-A5 3B: no-empty minus default prompt'),
                 ('qwen3_8b_noempty100', 'qwen3_8b', 'B-A5 Qwen3: no-empty minus default prompt'),
                 ('qwen25_3b_single100', 'qwen25_3b', 'single 3B: single-class minus default prompt'),
                 ('qwen25_7b_single100', 'qwen25_7b', 'single 7B: single-class minus default prompt'),
                 ('qwen3_8b_single100', 'qwen3_8b', 'single Qwen3: single-class minus default prompt')]
    out['differences_full'] = {lab: dict(diff=full[a].ap() - full[b].ap(), ci=ci(boot_full[a] - boot_full[b])) for a, b, lab in diffs_full}
    out['differences_first100'] = {lab: dict(diff=sub[a].ap() - sub[b].ap(), ci=ci(boot_sub[a] - boot_sub[b])) for a, b, lab in diffs_sub}
    # Spearman rho, COCO AP vs DUO AP, over image resamples
    from scipy.stats import spearmanr
    coco = [COCO_AP[m] for m in FT]
    rho = [spearmanr(coco, [boot_full[m][d] for m in FT]).correlation for d in range(len(W_all))]
    out['spearman_coco_duo'] = dict(rho=float(spearmanr(coco, [full[m].ap() for m in FT]).correlation), ci=ci(rho))
    # merged-class AP with intervals
    out['merged_class_ap_full'] = {k: dict(ap=m.ap(), ci=ci([m.ap(w) for w in W_all]), four_class_ap=full[k].ap()) for k, m in mg.items()}
    mq = Matched(Gm, merge_dets([x for x in dets['qwen3_8b_100']]), first100_sorted)
    out['merged_class_ap_first100'] = {'qwen3_8b_100': dict(ap=mq.ap(), ci=ci([mq.ap(w) for w in W_100]), four_class_ap=sub['qwen3_8b_100'].ap())}
    for k in ['gdino', 'owlv2']:
        m = Matched(Gm, merge_dets([x for x in dets[k] if x['image_id'] in set(first100)]), first100_sorted)
        out['merged_class_ap_first100'][k] = dict(ap=m.ap(), ci=ci([m.ap(w) for w in W_100]), four_class_ap=sub[k].ap())
    # per-class AP with intervals (full set)
    out['per_class_ap_full'] = {}
    for k in ['yolo11n', 'yolo11l', 'gdino', 'owlv2', 'qwen25_7b', 'qwen3_8b']:
        pcs = np.array([full[k].ap(w, per_class=True) for w in W_all]); point = full[k].ap(per_class=True)
        out['per_class_ap_full'][k] = {g_['name']: dict(ap=float(point[n]), ci=ci(pcs[:, n])) for n, g_ in enumerate(sorted(g['categories'], key=lambda c: c['id']))}

    # per-object measures (found / named) and the matched box budget: see e1.py
    (Path(__file__).resolve().parent / 'assess.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:200], '...')


if __name__ == '__main__':
    main()
