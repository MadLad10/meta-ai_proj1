"""RUOD, one model per call (memory): matched detections for bootstrap AP, per-object found/named counts, and the
matched box budget against Qwen3-VL-8B on the fixed 1,000-image VLM subset. Writes analysis/cache/ruod_<model>.npz.
    python analysis/ruod_worker.py <model>        # see RUNS below
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_bootstrap import load_gt, Matched
from e1 import per_image, budget_counts
from assess import by_image

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'gpu_job' / 'outputs'; Z = O / 'job5' / 'ruod'
CACHE = ROOT / 'analysis' / 'cache'; CACHE.mkdir(exist_ok=True)
RUNS = {**{m: O / 'job7' / 'ruod' / m for m in ['yolov8n', 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'rtdetr-l']},
        'gdino': Z / 'grounding-dino_common_posmap', 'owlv2': Z / 'owlv2_common', 'qwen25_3b': Z / 'qwen2.5-vl-3b_common',
        'qwen25_7b': Z / 'qwen2.5-vl-7b_common', 'qwen3_8b': Z / 'qwen3-vl-8b_common'}
VLM = {'qwen25_3b', 'qwen25_7b', 'qwen3_8b'}


def main(k):
    g = json.loads((O / 'gt' / 'ruod_instances_test.json').read_text()); G = load_gt(g)
    all_ids = sorted(im['id'] for im in g['images'])
    sub = sorted(im['id'] for im in json.loads((Z / '_gt_1000.json').read_text())['images'])
    ids = sub if k in VLM else all_ids                         # VLMs ran only on the subset
    dets = json.loads((RUNS[k] / 'detections.json').read_text())
    m = Matched(G, dets, ids)
    arrays = dict(img_ids=np.array(ids))
    for c, pc in zip(m.cats, m.per_cat):
        for key in ('npig', 'S', 'IMG', 'TP', 'FP'):
            arrays[f'c{c}_{key}'] = pc[key]
    cats = [c['id'] for c in sorted(g['categories'], key=lambda c: c['id'])]
    for name, (top, thr) in {'top100_iou50': (100, 0.5)}.items():
        rows, labels = per_image(g, dets, set(ids), top, thr)
        arrays['fn_' + name] = np.array([[rows[i][c] for c in cats] for i in ids])      # images x classes x [objects, found, named]
        arrays['labels'] = np.array([[labels[c][e] for e in cats] for c in cats])       # true class x label of best box
    if k in {'gdino', 'owlv2', 'yolo11n', 'qwen3_8b'}:                                  # matched box budget on the subset
        q = by_image(json.loads((RUNS['qwen3_8b'] / 'detections.json').read_text()))
        budget = {i: len({tuple(round(v, 2) for v in x['bbox']) for x in q.get(i, [])}) for i in sub}
        if k == 'qwen3_8b':
            budget = {i: 10 ** 6 for i in sub}
        s = set(sub); d_sub = [x for x in dets if x['image_id'] in s]
        bc = budget_counts(g, d_sub, s, budget, agnostic=False)
        arrays['budget'] = np.array([bc[i] for i in sub])
    np.savez_compressed(CACHE / f'ruod_{k}.npz', **arrays)
    print(k, 'AP', round(m.ap(), 4), 'images', len(ids))


if __name__ == '__main__':
    main(sys.argv[1])
