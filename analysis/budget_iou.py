"""Matched box budget at stricter overlap (DUO, full test set): does Qwen3-VL-8B's advantage over the open-vocabulary
detectors at a matched box budget hold when a box must overlap its object with IoU >= 0.75 instead of 0.5? Each detector
keeps on every image as many distinct boxes as Qwen3-VL-8B returned there (as in e1.py); correct class required.
Bootstrap over test images (the same 500 draws as assess.py). Loads one detection file at a time. Writes
analysis/budget_iou.json."""
import gc, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_bootstrap import draws
from assess import GT, RUNS, by_image, ci
from e1 import budget_counts, pr_boot


def main():
    g = json.loads(GT.read_text()); all_ids = sorted(im['id'] for im in g['images']); s = set(all_ids); W = draws(len(all_ids))
    q = json.loads((RUNS['qwen3_8b'] / 'detections.json').read_text()); qi = by_image(q)
    budget = {i: len({tuple(round(v, 2) for v in x['bbox']) for x in qi.get(i, [])}) for i in all_ids}
    out = {}
    for thr in (0.5, 0.75):
        out[str(thr)] = {'qwen3_8b': pr_boot(budget_counts(g, q, s, {i: 10 ** 6 for i in all_ids}, False, thr), all_ids, W)}
    del q, qi; gc.collect()
    for k in ['gdino', 'owlv2', 'yolo11n']:
        d = json.loads((RUNS[k] / 'detections.json').read_text())
        for thr in (0.5, 0.75):
            out[str(thr)][k] = pr_boot(budget_counts(g, d, s, budget, False, thr), all_ids, W)
        del d; gc.collect(); print(k, flush=True)
    P = lambda t: t[:, 0] / np.maximum(t[:, 0] + t[:, 1], 1); R = lambda t: t[:, 0] / np.maximum(t[:, 2], 1)
    for thr, res in out.items():
        for k in ['gdino', 'owlv2', 'yolo11n']:
            for m, f in (('precision', P), ('recall', R)):
                res[f'qwen3_minus_{k}_{m}'] = dict(diff=float(res['qwen3_8b'][m] - res[k][m]), ci=ci(f(res['qwen3_8b']['_bs']) - f(res[k]['_bs'])))
        for v in res.values():
            v.pop('_bs', None)
    (Path(__file__).resolve().parent / 'budget_iou.json').write_text(json.dumps(out, indent=1) + '\n')
    for thr, res in out.items():
        print('IoU', thr, {k: (round(v['precision'], 3), round(v['recall'], 3)) for k, v in res.items() if 'minus' not in k})
        print('   ', {k: (round(v['diff'], 3), [round(c, 3) for c in v['ci']]) for k, v in res.items() if 'minus' in k})


if __name__ == '__main__':
    main()
