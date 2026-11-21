"""Full-image label analysis (per object: best-overlapping box among the image's 100 highest-scoring boxes, IoU >= 0.5).
  vocab  : the 300 DUO images of vocab_full.py, four names (GPU job5 outputs) vs RUOD's ten names (CPU); found, named
           (strict), the labels of found objects, the share of wrong names that go to the most common wrong label, and TIDE.
  wrong  : on the full test sets (DUO; RUOD picking classes), the share of wrong names that go to the most common wrong label.
    python analysis/full_labels.py vocab | wrong <duo|ruod> <gdino|owlv2>      writes analysis/full_labels/<...>.json"""
import json, random, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'gpu_job' / 'outputs'; OUT = ROOT / 'analysis' / 'full_labels'
RUN = {('duo', 'gdino'): O / 'job5/duo/grounding-dino_common_posmap', ('duo', 'owlv2'): O / 'job5/duo/owlv2_common',
       ('ruod', 'gdino'): O / 'job5/ruod/grounding-dino_common_posmap', ('ruod', 'owlv2'): O / 'job5/ruod/owlv2_common'}
SEA = {'holothurian', 'echinus', 'scallop', 'starfish'}
OTHER = {101: 'fish', 102: 'corals', 103: 'diver', 104: 'cuttlefish', 105: 'turtle', 106: 'jellyfish'}


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2]); y1 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


def best_labels(gts, dets_by_img):
    """[(gt class, best-box class or None)] for every ground-truth object in gts."""
    out = []
    for i, gs in gts.items():
        ds = sorted(dets_by_img.get(i, []), key=lambda d: -d[4])[:100]
        M = iou([a['bbox'] for a in gs], [d[:4] for d in ds]) if ds else np.zeros((len(gs), 0))
        for j, a in enumerate(gs):
            hit = np.where(M[j] >= 0.5)[0] if M.shape[1] else []
            out.append((a['category_id'], int(ds[hit[np.argmax(M[j, hit])]][5]) if len(hit) else None))
    return out


def summarize(pairs, name):
    n = len(pairs); found = [(g, l) for g, l in pairs if l is not None]; wrong = [l for g, l in found if l != g]
    wc = Counter(name.get(l, str(l)) for l in wrong); top = wc.most_common(2)
    return dict(objects=n, found=len(found) / n, named_strict=sum(g == l for g, l in found) / n, misnamed_of_found=len(wrong) / max(len(found), 1),
                top_wrong=top[0][0] if top else None, top_wrong_share=top[0][1] / len(wrong) if wrong else None,
                top2_wrong_share=sum(c for _, c in top) / len(wrong) if wrong else None,
                labels_of_found=dict(Counter(name.get(l, str(l)) for _, l in found).most_common()))


def load_gpu(bench, model, keep=None):
    d = json.loads((RUN[bench, model] / 'detections.json').read_text()); by = {}
    for x in d:
        if keep is None or x['image_id'] in keep:
            by.setdefault(x['image_id'], []).append(list(x['bbox']) + [x['score'], x['category_id']])
    return by


def tide(g, keep, by, extra_classes):
    for _a in ('float', 'int', 'bool'):
        if not hasattr(np, _a):
            setattr(np, _a, eval(_a))
    from tidecv import TIDE, Data
    from tidecv import ap as _tide_ap                          # like pycocotools: classes without ground truth are left out of mAP
    _tide_ap.APDataObject.is_empty = lambda self: self.num_gt_positives == 0
    gt, pr = Data('gt'), Data('pred')
    for a in g['annotations']:
        if a['image_id'] in keep:
            gt.add_ground_truth(a['image_id'], a['category_id'], a['bbox'])
    for i in keep:
        gt.add_image(i, str(i))
    for c in [c['id'] for c in g['categories']] + extra_classes:
        gt.add_class(c, str(c))
    for i, ds in by.items():
        for d in ds:
            pr.add_detection(i, int(d[5]), d[4], d[:4])
    t = TIDE(); t.evaluate(gt, pr, mode=TIDE.BOX)
    return dict(ap50=float(t.runs['pred'].ap), errors={getattr(k, 'short_name', k): float(v) for k, v in t.get_main_errors()['pred'].items()})


def main():
    OUT.mkdir(exist_ok=True); mode = sys.argv[1]
    if mode == 'vocab':
        g = json.loads((O / 'gt' / 'duo_instances_test.json').read_text()); name = {c['id']: c['name'] for c in g['categories']} | OTHER
        keep = set(sorted(random.Random(17).sample(sorted(im['id'] for im in g['images']), 300)))
        gts = {}
        for a in g['annotations']:
            if a['image_id'] in keep:
                gts.setdefault(a['image_id'], []).append(a)
        res = {}
        for model in ('gdino', 'owlv2'):
            f = ROOT / 'analysis' / 'vocab_full' / f'{model}_ruod10_n300.jsonl'
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            assert {r['image_id'] for r in rows} == keep, f'{model}: incomplete ({len(rows)} images)'
            ten = {r['image_id']: r['dets'] for r in rows}; four = load_gpu('duo', model, keep)
            res[model] = {'duo4': summarize(best_labels(gts, four), name) | {'tide': tide(g, keep, four, [])},
                          'ruod10': summarize(best_labels(gts, ten), name) | {'tide': tide(g, keep, ten, list(OTHER))}}
            print(model, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'labels_of_found'} for k, v in res[model].items()}), flush=True)
        res['images'] = len(keep); res['objects'] = sum(len(v) for v in gts.values())
        (OUT / 'vocab.json').write_text(json.dumps(res, indent=1) + '\n')
    else:
        bench, model = sys.argv[2], sys.argv[3]
        g = json.loads((O / 'gt' / f'{bench}_instances_test.json').read_text()); name = {c['id']: c['name'] for c in g['categories']}
        sea = {c['id'] for c in g['categories'] if c['name'] in SEA}; gts = {}
        for a in g['annotations']:
            if a['category_id'] in sea:
                gts.setdefault(a['image_id'], []).append(a)
        by = load_gpu(bench, model, set(gts)); r = summarize(best_labels(gts, by), name)
        (OUT / f'wrong_{bench}_{model}.json').write_text(json.dumps(r, indent=1) + '\n'); print(bench, model, json.dumps(r))


if __name__ == '__main__':
    main()
