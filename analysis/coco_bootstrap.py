"""COCO AP (pycocotools protocol) with a fast paired bootstrap over test images.

pycocotools matches detections to ground truth once per (class, image); AP for any resampling of images is then
re-accumulated from those matches with per-image weights (an image drawn twice counts twice). With all weights 1 the
result equals pycocotools' own AP (checked by `self_check`).
"""
import contextlib, io, json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REC_THRS = np.linspace(0.0, 1.00, 101)


def load_gt(path_or_dict, merge_classes=False):
    g = json.loads(Path(path_or_dict).read_text()) if not isinstance(path_or_dict, dict) else json.loads(json.dumps(path_or_dict))
    if merge_classes:
        g['categories'] = [dict(id=1, name='object')]
        g['annotations'] = [dict(a, category_id=1) for a in g['annotations']]
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(); G.dataset = g; G.createIndex()
    return G


def merge_dets(dets):
    """One box per (image, box) with the best-scoring class, relabelled as class 1 (for merged-class AP)."""
    best = {}
    for x in dets:
        k = (x['image_id'], tuple(round(v, 2) for v in x['bbox']))
        if k not in best or x['score'] > best[k]['score']:
            best[k] = dict(x, category_id=1)
    return list(best.values())


class Matched:
    """Per-class arrays of one model's matched detections, sorted as pycocotools sorts them."""

    def __init__(self, G, dets, img_ids):
        self.img_ids = list(img_ids); pos = {i: n for n, i in enumerate(self.img_ids)}
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(G, G.loadRes(dets) if dets else COCO(), 'bbox') if dets else None
            if E is not None:
                E.params.imgIds = self.img_ids; E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ['all']; E.params.maxDets = [100]
                E.evaluate()
        cats = sorted(G.getCatIds()); self.cats = cats; self.per_cat = []
        I = len(self.img_ids)
        for k, c in enumerate(cats):
            npig = np.zeros(I)
            for n, i in enumerate(self.img_ids):
                npig[n] = sum(1 for a in G.imgToAnns[i] if a['category_id'] == c and not a.get('iscrowd', 0))
            S, TP, V, IMG = [], [], [], []
            if E is not None:
                for n in range(I):
                    e = E.evalImgs[k * I + n]
                    if e is None or len(e['dtScores']) == 0:
                        continue
                    S.append(np.asarray(e['dtScores'])); IMG.append(np.full(len(e['dtScores']), n))
                    ig = np.asarray(e['dtIgnore'], bool); m = np.asarray(e['dtMatches']) > 0
                    TP.append(m & ~ig); V.append(~ig)
            if S:
                S = np.concatenate(S); IMG = np.concatenate(IMG); TP = np.concatenate(TP, 1); V = np.concatenate(V, 1)
                order = np.argsort(-S, kind='mergesort')
                S, IMG, TP, V = S[order], IMG[order], TP[:, order], V[:, order]
            else:
                S = np.zeros(0); IMG = np.zeros(0, int); TP = np.zeros((10, 0), bool); V = np.zeros((10, 0), bool)
            self.per_cat.append(dict(npig=npig, S=S, IMG=IMG, TP=TP.astype(float), FP=(V & ~TP.astype(bool)).astype(float)))

    def ap(self, w=None, per_class=False):
        """COCO AP (IoU 0.50:0.95) with image weights w (default all 1); NaN for a class with no ground truth."""
        w = np.ones(len(self.img_ids)) if w is None else np.asarray(w, float)
        out = []
        for pc in self.per_cat:
            npig = float(pc['npig'] @ w)
            if npig == 0:
                out.append(np.nan); continue
            if pc['S'].size == 0:
                out.append(0.0); continue
            ww = w[pc['IMG']]
            tp = np.cumsum(pc['TP'] * ww, 1); fp = np.cumsum(pc['FP'] * ww, 1)
            rc = tp / npig; pr = tp / (tp + fp + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], 1)[:, ::-1]           # precision envelope, as pycocotools
            q = np.zeros((pr.shape[0], len(REC_THRS)))
            for t in range(pr.shape[0]):
                inds = np.searchsorted(rc[t], REC_THRS, side='left'); ok = inds < pr.shape[1]
                q[t, ok] = pr[t, inds[ok]]
            out.append(q.mean())
        out = np.array(out)
        return out if per_class else float(np.nanmean(out))


def self_check(G, dets, img_ids, m):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, G.loadRes(dets), 'bbox'); E.params.imgIds = list(img_ids); E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0]), m.ap()


def draws(n_images, n=500, seed=20261119):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, n_images, n_images), minlength=n_images) for _ in range(n)]
