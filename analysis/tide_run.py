"""TIDE error breakdown (Bolya et al. 2020; tidecv 1.0.1, default thresholds: foreground IoU 0.5, background 0.1) for
DUO (all images) and RUOD restricted to the test images that contain any of its four picking classes (all ground
truth and detections on those images kept, so a picking animal boxed as "fish" counts as a classification error; only
11 of these images contain any other class). Reports AP50 and the AP50 gained by fixing each error type: classification, localization, both,
duplicate, background, missed. One model per call (memory).
    python analysis/tide_run.py <duo|ruod> <model> [one_label]   writes analysis/tide/<bench>_<model>[_one_label].json
one_label: keep only the highest-scoring class of each box (Grounding DINO can return one box with several classes)."""
import json, sys
from pathlib import Path
import numpy as np
for _a in ('float', 'int', 'bool'):                          # tidecv 1.0.1 uses aliases removed from NumPy
    if not hasattr(np, _a):
        setattr(np, _a, eval(_a))
from tidecv import TIDE, Data
from tidecv import ap as _tide_ap                          # like pycocotools: classes without ground truth are left out of mAP
_tide_ap.APDataObject.is_empty = lambda self: self.num_gt_positives == 0

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'gpu_job' / 'outputs'
RUNS = {'duo': {'yolo11n': O / 'job7/duo/yolo11n', 'gdino': O / 'job5/duo/grounding-dino_common_posmap', 'owlv2': O / 'job5/duo/owlv2_common',
                'qwen3_8b': O / 'job5/duo/qwen3-vl-8b_common'},
        'ruod': {'yolo11n': O / 'job7/ruod/yolo11n', 'gdino': O / 'job5/ruod/grounding-dino_common_posmap', 'owlv2': O / 'job5/ruod/owlv2_common'}}
SEA = {'holothurian', 'echinus', 'scallop', 'starfish'}


def main():
    bench, model = sys.argv[1], sys.argv[2]
    g = json.loads((O / 'gt' / f'{bench}_instances_test.json').read_text())
    sea = {c['id'] for c in g['categories'] if c['name'] in SEA}; keep = {c['id'] for c in g['categories']}
    gt, pr = Data('gt'), Data('pred')
    imgs = {im['id'] for im in g['images']} if bench == 'duo' else {a['image_id'] for a in g['annotations'] if a['category_id'] in sea}
    for a in g['annotations']:
        if a['image_id'] in imgs:
            gt.add_ground_truth(a['image_id'], a['category_id'], a['bbox'])
    for i in imgs:
        gt.add_image(i, str(i))
    for c in keep:
        gt.add_class(c, str(c))
    dets = json.loads((RUNS[bench][model] / 'detections.json').read_text())
    one = len(sys.argv) > 3 and sys.argv[3] == 'one_label'
    if one:                                                  # one label per box: the best-scoring class
        best = {}
        for d in dets:
            k = (d['image_id'], tuple(round(v, 2) for v in d['bbox']))
            if k not in best or d['score'] > best[k]['score']:
                best[k] = d
        dets = list(best.values())
    for d in dets:
        if d['image_id'] in imgs:
            pr.add_detection(d['image_id'], d['category_id'], d['score'], d['bbox'])
    del dets
    t = TIDE(); t.evaluate(gt, pr, mode=TIDE.BOX); r = t.runs['pred']
    main_err = {getattr(k, 'short_name', k): v for k, v in t.get_main_errors()['pred'].items()}
    out = dict(bench=bench, model=model, one_label=one, images=len(imgs), ap50=float(r.ap), errors={k: float(v) for k, v in main_err.items()},
               special={getattr(k, 'short_name', str(k)): float(v) for k, v in t.get_special_errors()['pred'].items()})
    (ROOT / 'analysis' / 'tide').mkdir(exist_ok=True)
    (ROOT / 'analysis' / 'tide' / f'{bench}_{model}{"_one_label" if one else ""}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
