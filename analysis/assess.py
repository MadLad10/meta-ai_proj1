"""Evidence for the paper's claims, from the saved detections only (no GPU).

1. Paired bootstrap (500 draws over test images, seed 20261119) for each AP and each planned comparison.
2. Sea cucumbers and the other classes: is a ground-truth object found by any box (IoU >= 0.5), and with which label?
3. Matched box budget: open-vocabulary detectors cut to Qwen3-VL-8B's box count per image, all boxes scored 1.0, so
   both kinds of model are compared at one operating point with the same number of boxes.
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
        'qwen25_7b_noempty100': Z / 'qwen2.5-vl-7b_common_noempty_first100', 'qwen3_8b_100': Z / 'qwen3-vl-8b_common_first100'}
FIRST100_ONLY = {'qwen25_3b_noempty100', 'qwen25_7b_noempty100', 'qwen3_8b_100'}


def ci(x):
    x = np.asarray(x); return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


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


def found_and_named(g, dets, img_ids, top=100):
    """Per GT class: share of objects overlapped (IoU >= 0.5) by any of the image's top boxes, share with a correctly
    labelled box, and the label of the best-overlapping box."""
    cats = {c['id']: c['name'] for c in g['categories']}; D = by_image(dets); res = {}
    gt = {}
    for a in g['annotations']:
        if a['image_id'] in img_ids:
            gt.setdefault(a['image_id'], []).append(a)
    for c in cats:
        res[cats[c]] = dict(objects=0, found=0, named=0, label_of_best={n: 0 for n in cats.values()})
    for i in img_ids:
        ds = sorted(D.get(i, []), key=lambda x: -x['score'])[:top]
        gs = gt.get(i, [])
        if not gs:
            continue
        M = iou(np.array([a['bbox'] for a in gs], float).reshape(-1, 4), np.array([x['bbox'] for x in ds], float).reshape(-1, 4))
        for j, a in enumerate(gs):
            r = res[cats[a['category_id']]]; r['objects'] += 1
            hit = np.where(M[j] >= 0.5)[0] if M.shape[1] else []
            if len(hit):
                r['found'] += 1
                if any(ds[h]['category_id'] == a['category_id'] for h in hit):
                    r['named'] += 1
                best = hit[np.argmax(M[j, hit])]; r['label_of_best'][cats[ds[best]['category_id']]] += 1
    for r in res.values():
        r['found_share'] = r['found'] / max(r['objects'], 1); r['named_share'] = r['named'] / max(r['objects'], 1)
    return res


def pr_at(g, dets, img_ids, agnostic=False):
    """Precision and recall at IoU 0.5, greedy matching in score order (ties: input order)."""
    D = by_image(dets); gt = {}
    for a in g['annotations']:
        if a['image_id'] in img_ids:
            gt.setdefault(a['image_id'], []).append(a)
    tp = fp = 0; n_gt = sum(len(v) for v in gt.values())
    for i in img_ids:
        ds = sorted(D.get(i, []), key=lambda x: -x['score']); gs = gt.get(i, []); used = set()
        M = iou(np.array([x['bbox'] for x in ds], float).reshape(-1, 4), np.array([a['bbox'] for a in gs], float).reshape(-1, 4))
        for k, x in enumerate(ds):
            cand = [j for j in range(len(gs)) if j not in used and M[k, j] >= 0.5 and (agnostic or gs[j]['category_id'] == x['category_id'])]
            if cand:
                j = max(cand, key=lambda j: M[k, j]); used.add(j); tp += 1
            else:
                fp += 1
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(n_gt, 1), boxes=tp + fp)


def main():
    g = json.loads(GT.read_text()); all_ids = sorted(im['id'] for im in g['images'])
    first100 = [im['id'] for im in g['images'][:100]]; first100_sorted = sorted(first100)
    G = load_gt(g); Gm = load_gt(g, merge_classes=True)
    out = dict(protocol='paired bootstrap over test images, 500 draws, seed 20261119; 95% percentile intervals')
    # one run at a time, keeping only the detections needed later (memory)
    KEEP = {'yolo11n', 'gdino', 'owlv2', 'qwen25_7b', 'qwen3_8b_100'}; MERGE = {'yolo11n', 'yolo11l', 'gdino', 'owlv2', 'qwen25_7b'}
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
                  ('qwen25_7b', 'qwen25_7b_low', 'B-A2: default minus low resolution'), ('yolo11l', 'yolo11n', 'largest minus smallest YOLO11')]
    diffs_sub = [('qwen3_8b_100', 'qwen25_7b', 'H4: Qwen3-VL-8B minus Qwen2.5-VL-7B'), ('gdino', 'qwen3_8b_100', 'Grounding DINO minus Qwen3-VL-8B'),
                 ('owlv2', 'qwen3_8b_100', 'OWLv2 minus Qwen3-VL-8B'), ('qwen25_7b_noempty100', 'qwen25_7b', 'B-A5 7B: no-empty minus default prompt'),
                 ('qwen25_3b_noempty100', 'qwen25_3b', 'B-A5 3B: no-empty minus default prompt')]
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
    for k in ['yolo11n', 'yolo11l', 'gdino', 'owlv2', 'qwen25_7b']:
        pcs = np.array([full[k].ap(w, per_class=True) for w in W_all]); point = full[k].ap(per_class=True)
        out['per_class_ap_full'][k] = {g_['name']: dict(ap=float(point[n]), ci=ci(pcs[:, n])) for n, g_ in enumerate(sorted(g['categories'], key=lambda c: c['id']))}

    # 2. found vs named, per class
    fn_runs = {'yolo11n': all_ids, 'gdino': all_ids, 'owlv2': all_ids, 'qwen25_7b': all_ids, 'qwen3_8b_100': first100}
    out['found_and_named'] = {k: found_and_named(g, dets[k], set(ids)) for k, ids in fn_runs.items()}

    # 3. matched box budget on the first 100 images
    q = by_image(dets['qwen3_8b_100']); budget = {i: len(q.get(i, [])) for i in first100}
    res = {'qwen3_8b_100': dict(boxes_per_image=float(np.mean(list(budget.values()))))}
    res['qwen3_8b_100'].update(class_aware=pr_at(g, dets['qwen3_8b_100'], set(first100)),
                               class_agnostic=pr_at(g, dets['qwen3_8b_100'], set(first100), agnostic=True))
    for k in ['gdino', 'owlv2', 'yolo11n']:
        D = by_image(merge_dets([x for x in dets[k] if x['image_id'] in set(first100)]))   # one box per location, best class
        labels = {}
        for x in dets[k]:
            if x['image_id'] in set(first100):
                kk = (x['image_id'], tuple(round(v, 2) for v in x['bbox']))
                if kk not in labels or x['score'] > labels[kk][0]:
                    labels[kk] = (x['score'], x['category_id'])
        cut = []
        for i in first100:
            for x in sorted(D.get(i, []), key=lambda x: -x['score'])[:budget[i]]:
                cut.append(dict(x, score=1.0, category_id=labels[(i, tuple(round(v, 2) for v in x['bbox']))][1]))
        res[k] = dict(boxes_per_image=len(cut) / 100, class_aware=pr_at(g, cut, set(first100)), class_agnostic=pr_at(g, cut, set(first100), agnostic=True),
                      ap_at_budget=Matched(G, cut, first100_sorted).ap())
    res['qwen3_8b_100']['ap_at_budget'] = sub['qwen3_8b_100'].ap()
    out['matched_budget_first100'] = res
    (Path(__file__).resolve().parent / 'assess.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1)[:200], '...')


if __name__ == '__main__':
    main()
