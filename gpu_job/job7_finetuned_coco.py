"""Job 7 (track B): score the fine-tuned detectors of Job 4 (DUO) and Job 6 (RUOD) with the same COCO protocol and TIDE
error breakdown as the zero-shot models of Job 5, so every model is compared on identical terms.

Needs the trained weights (outputs/job4/_runs/<model>/weights/best.pt, outputs/job6/_runs/...). Run before deleting _runs.
    python job7_finetuned_coco.py --dataset duo        # six models on the DUO test set
    python job7_finetuned_coco.py --dataset ruod       # six models on the RUOD test set (if Job 6 ran)
Output: outputs/job7/<dataset>/<model>/{detections.json, metrics.json}
"""
import argparse, importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def load(name, file):
    s = importlib.util.spec_from_file_location(name, HERE / file); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
job5 = load('job5', 'job5_duo_zeroshot.py'); job4 = job5.job4
OUT = HERE / 'outputs' / 'job7'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='duo', choices=['duo', 'ruod']); a = ap.parse_args()
    from ultralytics import YOLO, RTDETR
    job5.use_dataset(a.dataset)
    runs = HERE / 'outputs' / ('job4' if a.dataset == 'duo' else 'job6') / '_runs'
    gt_path, d = job4.coco_files()['test']
    cat_by_name = {c['name'].strip().lower(): c['id'] for c in d['categories']}
    classes = [c['name'].strip().lower() for c in sorted(d['categories'], key=lambda c: c['id'])]   # YOLO class index order (prepare)
    test_dir = job4.DATA / 'images' / 'test'
    for name in job4.MODELS:
        w = runs / name / 'weights' / 'best.pt'
        dest = OUT / a.dataset / name
        if not w.exists() or (dest / 'metrics.json').exists():
            print(f'{name}: {"no weights" if not w.exists() else "done"}, skipping'); continue
        dest.mkdir(parents=True, exist_ok=True)
        model = (RTDETR if name.startswith('rtdetr') else YOLO)(str(w)); dets = []
        for b in range(0, len(d['images']), 32):
            batch = d['images'][b:b + 32]
            for im, r in zip(batch, model.predict([str(test_dir / Path(i['file_name']).name) for i in batch], conf=0.001,
                                                  imgsz=job4.RECIPE['imgsz'], verbose=False, max_det=300)):
                for (x0, y0, x1, y1), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    dets.append(dict(image_id=im['id'], category_id=cat_by_name[classes[int(c)]], bbox=[x0, y0, x1 - x0, y1 - y0], score=float(s)))
        (dest / 'detections.json').write_text(json.dumps(dets))
        m = job5.evaluate(gt_path, dest / 'detections.json', cat_by_name); m.update(model=name, dataset=a.dataset, detections=len(dets))
        (dest / 'metrics.json').write_text(json.dumps(m, indent=1) + '\n')
        print(f"{name}: COCO AP {m['AP']:.3f}  AP50 {m['AP50']:.3f}  AP_small {m['AP_small']:.3f}")


if __name__ == '__main__':
    main()
