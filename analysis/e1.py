"""E1: intervals and sensitivity for the per-object measures (found / named) and the matched box budget.

Found: a ground-truth object overlapped (IoU >= thr) by any of the image's `top` highest-scoring boxes, of any class.
Named: overlapped by such a box that carries the object's class. Shares are bootstrapped over test images
(500 draws, seed 20261119, the same draws as assess.py). Writes analysis/e1.json.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_bootstrap import draws, merge_dets
from assess import O, GT, RUNS, iou, by_image, ci

SETTINGS = {'top100_iou50': (100, 0.5), 'top10_iou50': (10, 0.5), 'top100_iou70': (100, 0.7)}


def per_image(g, dets, img_ids, top, thr):
    """{image: {class_id: [objects, found, named]}} and best-box label counts per GT class."""
    D = by_image(dets); gt = {}
    for a in g['annotations']:
        if a['image_id'] in img_ids:
            gt.setdefault(a['image_id'], []).append(a)
    cats = [c['id'] for c in sorted(g['categories'], key=lambda c: c['id'])]
    rows = {i: {c: [0, 0, 0] for c in cats} for i in img_ids}; labels = {c: {d: 0 for d in cats} for c in cats}
    for i in img_ids:
        gs = gt.get(i, [])
        if not gs:
            continue
        ds = sorted(D.get(i, []), key=lambda x: -x['score'])[:top]
        M = iou(np.array([a['bbox'] for a in gs], float).reshape(-1, 4), np.array([x['bbox'] for x in ds], float).reshape(-1, 4))
        for j, a in enumerate(gs):
            r = rows[i][a['category_id']]; r[0] += 1
            hit = np.where(M[j] >= thr)[0] if M.shape[1] else []
            if len(hit):
                r[1] += 1
                r[2] += int(any(ds[h]['category_id'] == a['category_id'] for h in hit))
                labels[a['category_id']][ds[hit[np.argmax(M[j, hit])]]['category_id']] += 1
    return rows, labels


def shares(rows, img_ids, W, cats):
    A = np.array([[rows[i][c] for c in cats] for i in img_ids], float)        # images x classes x 3
    out = {}
    for name, sel in [('all', slice(None))] + [(c, [n]) for n, c in enumerate(cats)]:
        a = A[:, sel, :].sum(1)                                                 # images x 3
        point = a.sum(0); bs = np.array([w @ a for w in W]); bs[bs[:, 0] == 0] = np.nan   # draws without the class
        out[str(name)] = dict(objects=int(point[0]), found=float(point[1] / point[0]), named=float(point[2] / point[0]),
                              found_ci=ci(bs[:, 1] / bs[:, 0]), named_ci=ci(bs[:, 2] / bs[:, 0]))
    return out


def budget_counts(g, dets, img_ids, budget, agnostic, thr=0.5):
    """Per image [tp, fp, n_gt] at IoU >= thr (default 0.5) after cutting to `budget[i]` boxes (one box per location, best class)."""
    D = by_image(dets); gt = {}
    for a in g['annotations']:
        if a['image_id'] in img_ids:
            gt.setdefault(a['image_id'], []).append(a)
    out = {}
    for i in img_ids:
        ds = sorted(D.get(i, []), key=lambda x: -x['score'])
        seen, cut = set(), []
        for x in ds:                                     # keep the best-scoring class per location
            if len(cut) >= budget[i]:
                break
            k = tuple(round(v, 2) for v in x['bbox'])
            if k not in seen:
                seen.add(k); cut.append(x)
        gs = gt.get(i, []); used = set(); tp = fp = 0
        M = iou(np.array([x['bbox'] for x in cut], float).reshape(-1, 4), np.array([a['bbox'] for a in gs], float).reshape(-1, 4))
        for k, x in enumerate(cut):
            cand = [j for j in range(len(gs)) if j not in used and M[k, j] >= thr and (agnostic or gs[j]['category_id'] == x['category_id'])]
            if cand:
                used.add(max(cand, key=lambda j: M[k, j])); tp += 1
            else:
                fp += 1
        out[i] = [tp, fp, len(gs)]
    return out


def pr_boot(counts, img_ids, W):
    A = np.array([counts[i] for i in img_ids], float); p = A.sum(0)
    bs = np.array([w @ A for w in W])
    prec = lambda t: t[..., 0] / np.maximum(t[..., 0] + t[..., 1], 1); rec = lambda t: t[..., 0] / np.maximum(t[..., 2], 1)
    return dict(precision=float(prec(p)), recall=float(rec(p)), precision_ci=ci(prec(bs)), recall_ci=ci(rec(bs)),
                boxes_per_image=float((p[0] + p[1]) / len(img_ids)), _bs=bs)


def main():
    g = json.loads(GT.read_text()); cats = [c['id'] for c in sorted(g['categories'], key=lambda c: c['id'])]
    names = {c['id']: c['name'] for c in g['categories']}
    all_ids = sorted(im['id'] for im in g['images']); first100 = sorted(im['id'] for im in g['images'][:100])
    W_all, W_100 = draws(len(all_ids)), draws(100)
    out = dict(classes=names, found_named={}, labels={}, budget={})
    runs = {'yolo11n': all_ids, 'gdino': all_ids, 'owlv2': all_ids, 'gdino_latin': all_ids, 'owlv2_latin': all_ids,
            'qwen25_7b': all_ids, 'qwen3_8b': all_ids, 'qwen3_8b_100': first100, 'gdino@100': first100, 'owlv2@100': first100}
    kept = {}
    for k, ids in runs.items():
        base = k.split('@')[0]
        d = kept.get(base) or json.loads((RUNS[base] / 'detections.json').read_text())
        if base in {'gdino', 'owlv2', 'yolo11n', 'qwen3_8b_100', 'qwen3_8b'}:
            kept[base] = d
        W = W_all if ids is all_ids else W_100; s = set(ids)
        out['found_named'][k] = {}
        for st, (top, thr) in SETTINGS.items():
            rows, labels = per_image(g, [x for x in d if x['image_id'] in s], s, top, thr)
            out['found_named'][k][st] = shares(rows, ids, W, cats)
            if st == 'top100_iou50':
                out['labels'][k] = {names[c]: {names[e]: n for e, n in v.items()} for c, v in labels.items()}
        print('found/named', k, flush=True)
    # matched box budget on the full test set, from the full Qwen3-VL-8B run
    qf = by_image(kept['qwen3_8b']); sall = set(all_ids)
    gt_all = {i: 0 for i in all_ids}
    for a in g['annotations']:
        gt_all[a['image_id']] += 1
    for bname, b in {'qwen3_count_full': {i: len({tuple(round(v, 2) for v in x['bbox']) for x in qf.get(i, [])}) for i in all_ids}, 'gt_count_full': gt_all}.items():
        out['budget'][bname] = {}
        for agn in (False, True):
            key = 'class_agnostic' if agn else 'class_aware'; res = {}
            for k in ['qwen3_8b', 'gdino', 'owlv2', 'yolo11n']:
                if k == 'qwen3_8b' and bname != 'qwen3_count_full':
                    continue
                bb = b if k != 'qwen3_8b' else {i: 10 ** 6 for i in all_ids}
                res[k] = pr_boot(budget_counts(g, kept[k], sall, bb, agn), all_ids, W_all)
            if 'qwen3_8b' in res:
                for k in ['gdino', 'owlv2']:
                    for m in ('precision', 'recall'):
                        f = (lambda t: t[:, 0] / np.maximum(t[:, 0] + t[:, 1], 1)) if m == 'precision' else (lambda t: t[:, 0] / np.maximum(t[:, 2], 1))
                        res[f'qwen3_minus_{k}_{m}'] = dict(diff=float(res['qwen3_8b'][m] - res[k][m]), ci=ci(f(res['qwen3_8b']['_bs']) - f(res[k]['_bs'])))
            for v in res.values():
                v.pop('_bs', None)
            out['budget'][bname][key] = res
    # matched box budget on the first 100 images: (a) Qwen3-VL-8B's own box count; (b) the number of true objects
    q = by_image(kept['qwen3_8b_100']); s100 = set(first100)
    gt_n = {i: 0 for i in first100}
    for a in g['annotations']:
        if a['image_id'] in s100:
            gt_n[a['image_id']] += 1
    budgets = {'qwen3_count': {i: len({tuple(round(v, 2) for v in x['bbox']) for x in q.get(i, [])}) for i in first100}, 'gt_count': gt_n}   # distinct boxes
    for bname, b in budgets.items():
        out['budget'][bname] = {}
        for agn in (False, True):
            key = 'class_agnostic' if agn else 'class_aware'; res = {}
            for k in ['qwen3_8b_100', 'gdino', 'owlv2', 'yolo11n']:
                if k == 'qwen3_8b_100' and bname != 'qwen3_count':
                    continue
                d = [x for x in kept[k] if x['image_id'] in s100]
                bb = b if k != 'qwen3_8b_100' else {i: 10 ** 6 for i in first100}   # Qwen3-VL: all its boxes
                res[k] = pr_boot(budget_counts(g, d, s100, bb, agn), first100, W_100)
            if 'qwen3_8b_100' in res:
                for k in ['gdino', 'owlv2']:
                    for m, j in (('precision', 0), ('recall', 1)):
                        f = (lambda t: t[:, 0] / np.maximum(t[:, 0] + t[:, 1], 1)) if m == 'precision' else (lambda t: t[:, 0] / np.maximum(t[:, 2], 1))
                        dif = f(res['qwen3_8b_100']['_bs']) - f(res[k]['_bs'])
                        res[f'qwen3_minus_{k}_{m}'] = dict(diff=float(res['qwen3_8b_100'][m] - res[k][m]), ci=ci(dif))
            for v in res.values():
                v.pop('_bs', None)
            out['budget'][bname][key] = res
    (Path(__file__).resolve().parent / 'e1.json').write_text(json.dumps(out, indent=1) + '\n')
    print('written')


if __name__ == '__main__':
    main()
