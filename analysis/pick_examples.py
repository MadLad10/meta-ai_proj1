"""Rank first-100 DUO test images as qualitative examples: sea cucumbers that both open-vocabulary detectors find but
misname, while the fine-tuned YOLO11n names them; few objects so the panels stay legible."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess import O, GT, RUNS, iou, by_image

def best_labels(gts, dets, top=100):
    ds = sorted(dets, key=lambda x: -x['score'])[:top]
    M = iou(np.array([a['bbox'] for a in gts], float).reshape(-1, 4), np.array([x['bbox'] for x in ds], float).reshape(-1, 4))
    out = []
    for j in range(len(gts)):
        hit = np.where(M[j] >= 0.5)[0] if M.shape[1] else []
        out.append(None if not len(hit) else ds[hit[np.argmax(M[j, hit])]]['category_id'])
    return out

g = json.loads(GT.read_text()); first = [im['id'] for im in g['images'][:100]]; ims = {im['id']: im for im in g['images']}
gt = {}
for a in g['annotations']:
    gt.setdefault(a['image_id'], []).append(a)
D = {k: by_image([x for x in json.loads((RUNS[k] / 'detections.json').read_text()) if x['image_id'] in set(first)]) for k in ['yolo11n', 'gdino', 'owlv2', 'qwen3_8b_100']}
rows = []
for i in first:
    gs = gt.get(i, [])
    if not gs: continue
    L = {k: best_labels(gs, D[k].get(i, [])) for k in D}
    sc = [j for j, a in enumerate(gs) if a['category_id'] == 1]
    good = [j for j in sc if L['yolo11n'][j] == 1 and L['gdino'][j] not in (None, 1) and L['owlv2'][j] not in (None, 1)]
    if not good: continue
    area = min(gs[j]['bbox'][2] * gs[j]['bbox'][3] for j in good) ** 0.5
    rows.append((len(good), -len(gs), area, i, ims[i]['file_name'], ims[i]['width'], ims[i]['height'], len(gs), len(sc),
                 {k: [L[k][j] for j in good] for k in D}))
rows.sort(reverse=True)
for r in rows[:12]:
    print(f"img {r[3]} {r[4]} {r[5]}x{r[6]} objects {r[7]} cucumbers {r[8]} misnamed-by-both {r[0]} min side {r[2]:.0f}px labels {r[9]}")
