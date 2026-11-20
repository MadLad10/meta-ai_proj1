"""Summary of the same-image vocabulary test (analysis/vocab_cpu/*_pc60.jsonl, 60 DUO crops per class). For each model and
class list: the most frequent answer on these seafloor crops and its share, the accuracy over the four DUO classes, and the
share of winning boxes that cover the object (IoU >= 0.5 inside the crop). The original four-class list uses the GPU
results for the same crops. For the crops whose winning box covers the object, also the top-answer share and
the share correct. Bootstrap over crops within class (500 draws, seed 20261119). Writes analysis/vocab.json."""
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; V = ROOT / 'analysis' / 'vocab_cpu'; J = ROOT / 'gpu_job' / 'outputs' / 'job8'
NAME = dict(holothurian='sea cucumber', echinus='sea urchin', scallop='scallop', starfish='starfish')


def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def stats(rows):
    gt = np.array([NAME[r['gt']] for r in rows]); pred = np.array([r['pred'] for r in rows])
    classes = sorted(set(gt)); by = {c: np.where(gt == c)[0] for c in classes}
    vals, cnt = np.unique(pred, return_counts=True); top = vals[cnt.argmax()]
    rng = np.random.default_rng(20261119); W = [np.concatenate([rng.choice(by[c], len(by[c])) for c in classes]) for _ in range(500)]
    cor = pred == gt; hit = pred == top
    acc = lambda idx: np.mean([cor[idx][gt[idx] == c].mean() for c in classes])
    out = dict(crops=len(rows), top=str(top), top_share=float(hit.mean()), top_ci=ci([hit[w].mean() for w in W]),
               accuracy=float(acc(np.arange(len(rows)))), accuracy_ci=ci([acc(w) for w in W]),
               answers={str(v): int(c) for v, c in zip(vals, cnt)})
    if 'box_iou' in rows[0]:
        cov = np.array([r['box_iou'] >= 0.5 for r in rows]); out['box_covers_object'] = float(cov.mean()); out['box_covers_ci'] = ci([cov[w].mean() for w in W])
        out['covered'] = dict(crops=int(cov.sum()), top_share=float(hit[cov].mean()), top_share_ci=ci([hit[w][cov[w]].mean() for w in W]),
                              correct=float(cor[cov].mean()), correct_ci=ci([cor[w][cov[w]].mean() for w in W]))   # crops whose winning box covers the object
    return out


def main():
    res = {}
    sample = None
    for f in sorted(V.glob('*_pc60.jsonl')):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if len(rows) < 240:
            print(f.name, 'incomplete', len(rows)); continue
        res[f.stem.replace('_pc60', '')] = stats(rows); sample = sorted(r['ann_id'] for r in rows)
    if sample:                                                 # original four-class list: GPU results on the same crops
        for model, run in (('gdino', 'grounding-dino_common'), ('owlv2', 'owlv2_common'), ('gdinolatin', 'grounding-dino_latin'), ('owlv2latin', 'owlv2_latin')):
            P = {json.loads(l)['ann_id']: json.loads(l) for l in (J / run / 'predictions.jsonl').read_text().splitlines()}
            rows = [dict(gt=P[i]['gt'], pred=NAME.get(P[i]['pred'], P[i]['pred'])) for i in sample]
            if 'latin' in model:                               # Latin answers: map back to the common names for scoring
                rows = [dict(gt=r['gt'], pred=NAME.get(r['pred'], r['pred'])) for r in rows]
            res[f'{model}_duo4_gpu'] = stats(rows)
    (ROOT / 'analysis' / 'vocab.json').write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res.items():
        print(f"{k:22s} top '{v['top']}' {v['top_share']:.0%} {[round(100 * x) for x in v['top_ci']]} | accuracy {v['accuracy']:.0%} {[round(100 * x) for x in v['accuracy_ci']]}"
              + (f" | box covers object {v['box_covers_object']:.0%}; among those top {v['covered']['top_share']:.0%} correct {v['covered']['correct']:.0%}" if 'box_covers_object' in v else '') + f" | {v['answers']}")


if __name__ == '__main__':
    main()
