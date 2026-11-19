"""Job 4: fine-tune the six detector architectures on DUO (real underwater detection benchmark) and evaluate them.

DUO (Liu et al., arXiv 2106.05681): 7,782 underwater images, 4 classes (holothurian, echinus, scallop, starfish),
official split 6,671 train / 1,111 test. Download it first (EXPERIMENT_PLAN.md, section D4) into gpu_job/duo_raw/.

Run from this folder:
    python job4_duo.py check              # find the annotation files and images, print counts; nothing is written
    python job4_duo.py prepare            # COCO JSON -> YOLO labels; 10% of the official train set held out for
                                          #   checkpoint selection; the official test set is used only in 'evaluate'
    python job4_duo.py train --smoke      # 1 epoch of yolo11n on 5% of the data, to outputs/job4_smoke/
    python job4_duo.py train              # all six models (about 2.5-3 h on an RTX 5090); resumable
    python job4_duo.py evaluate           # test-set mAP per model, plus recall and class confusion by object size
"""
import argparse, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'duo_raw'
DATA = HERE / 'duo_yolo'
OUT = HERE / 'outputs' / 'job4'
CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']
EXPECTED = {'train': 6671, 'test': 1111}
MODELS = {'yolo11n': 'yolo11n.pt', 'yolo11s': 'yolo11s.pt', 'yolo11m': 'yolo11m.pt', 'yolo11l': 'yolo11l.pt',
          'yolov8n': 'yolov8n.pt', 'rtdetr-l': 'rtdetr-l.pt'}
RECIPE = dict(imgsz=640, epochs=40, batch=8, workers=4, seed=17, deterministic=True, patience=40, cos_lr=False,
              verbose=False, device=0)
VAL_FRACTION, SPLIT_SEED = 0.10, 17
MATCH_IOU, REPORT_CONF = 0.5, 0.25
SIZE_BINS = {'small': (0, 32 ** 2), 'medium': (32 ** 2, 96 ** 2), 'large': (96 ** 2, float('inf'))}   # COCO convention


def coco_files():
    found = {}
    for p in RAW.rglob('*.json'):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and {'images', 'annotations', 'categories'} <= set(d):
            name = p.name.lower()
            split = 'train' if 'train' in name else 'test' if 'test' in name else 'val' if 'val' in name else name
            found[split] = (p, d)
    return found


def image_index(split):
    """Images of one split, keyed by file name. Only files under a folder named after the split count: DUO numbers
    each split's images from 1.jpg, so the same name in images/train and images/test is a different image."""
    idx = {}
    for root, _, files in os.walk(RAW):
        if split not in (part.lower() for part in Path(root).relative_to(RAW).parts):
            continue
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if f in idx:
                    sys.exit(f'{split}: {f} found twice ({idx[f]} and {Path(root) / f}); stop and tell the user.')
                idx[f] = Path(root) / f
    return idx


def check():
    if not RAW.exists():
        sys.exit(f'{RAW} does not exist: download and extract DUO there first (EXPERIMENT_PLAN.md D4).')
    files = coco_files(); missing_any = False
    for split, (p, d) in sorted(files.items()):
        idx = image_index(split)
        cats = [c['name'] for c in sorted(d['categories'], key=lambda c: c['id'])]
        missing = sum(1 for im in d['images'] if Path(im['file_name']).name not in idx)
        missing_any |= split in EXPECTED and missing > 0
        print(f"{split}: {p.relative_to(HERE)}  images {len(d['images'])} (expected {EXPECTED.get(split, '?')}), "
              f"annotations {len(d['annotations'])}, categories {cats}, image files in the {split} folder {len(idx)}, "
              f"missing {missing}")
    ok = all(s in files and len(files[s][1]['images']) == n for s, n in EXPECTED.items()) and not missing_any
    print('SPLIT MATCHES THE PAPER (6,671 train / 1,111 test): ' + ('yes' if ok else 'NO - stop and tell the user'))
    return ok


def prepare():
    if not check():
        sys.exit('Split does not match; not preparing.')
    import random
    files = coco_files(); counts = {}
    if DATA.exists():
        shutil.rmtree(DATA)
    train_ids = sorted(im['id'] for im in files['train'][1]['images'])
    rng = random.Random(SPLIT_SEED); held = set(rng.sample(train_ids, round(VAL_FRACTION * len(train_ids))))
    for split in ('train', 'test'):
        _, d = files[split]
        idx = image_index(split)
        cat_index = {c['id']: CLASSES.index(c['name'].strip().lower()) for c in d['categories']}
        anns = {}
        for a in d['annotations']:
            anns.setdefault(a['image_id'], []).append(a)
        for im in d['images']:
            sub = 'test' if split == 'test' else ('val' if im['id'] in held else 'train')
            src = idx[Path(im['file_name']).name]
            (DATA / 'images' / sub).mkdir(parents=True, exist_ok=True); (DATA / 'labels' / sub).mkdir(parents=True, exist_ok=True)
            dst = DATA / 'images' / sub / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
            W, H = im['width'], im['height']; lines = []
            for a in anns.get(im['id'], []):
                x, y, w, h = a['bbox']
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{cat_index[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                counts.setdefault(sub, {}).setdefault(CLASSES[cat_index[a['category_id']]], 0)
                counts[sub][CLASSES[cat_index[a['category_id']]]] += 1
            (DATA / 'labels' / sub / (src.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
    (DATA / 'data.yaml').write_text(f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                    f"names:\n" + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))
    images = {s: len(list((DATA / 'images' / s).iterdir())) for s in ('train', 'val', 'test')}
    zips = sorted(RAW.parent.glob('DUO*.zip')) + sorted(RAW.glob('*.zip'))
    rec = dict(images=images, instances=counts, val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
               source_zip_sha256={z.name: hashlib.sha256(z.read_bytes()).hexdigest() for z in zips},
               licence_note=(RAW / 'LICENCE_NOTE.txt').read_text().strip() if (RAW / 'LICENCE_NOTE.txt').exists() else 'MISSING')
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'prepare.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1))


def train(only, smoke):
    import torch, ultralytics
    from ultralytics import YOLO, RTDETR
    if not torch.cuda.is_available():
        sys.exit('No CUDA GPU visible to PyTorch; stopping.')
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), ultralytics=ultralytics.__version__)
    print(json.dumps(env))
    out = OUT.parent / (OUT.name + '_smoke') if smoke else OUT
    names = ['yolo11n'] if smoke else [n for n in MODELS if not only or n in only]
    for name in names:
        dest = out / name
        if (dest / 'record.json').exists():
            print(f'{name}: already trained, skipping'); continue
        recipe = dict(RECIPE, **({'epochs': 1, 'fraction': 0.05} if smoke else {}))
        t0 = time.monotonic(); base = MODELS[name]
        (RTDETR if base.startswith('rtdetr') else YOLO)(base).train(data=str(DATA / 'data.yaml'), project=str(out / '_runs'),
                                                                     name=name, exist_ok=True, **recipe)
        run = out / '_runs' / name; dest.mkdir(parents=True, exist_ok=True)
        for f in ('results.csv', 'args.yaml'):
            if (run / f).exists():
                shutil.copy(run / f, dest / f)
        rec = dict(name=name, base=base, recipe=recipe, wall_s=time.monotonic() - t0,
                   best_sha256=hashlib.sha256((run / 'weights' / 'best.pt').read_bytes()).hexdigest(), env=env)
        (dest / 'record.json').write_text(json.dumps(rec, indent=1) + '\n')
        print(f'{name}: trained in {rec["wall_s"] / 60:.1f} min')


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def evaluate(smoke):
    from ultralytics import YOLO, RTDETR
    from PIL import Image
    out = OUT.parent / (OUT.name + '_smoke') if smoke else OUT
    test_images = sorted((DATA / 'images' / 'test').iterdir())
    if smoke:
        test_images = test_images[:50]
    for rec_path in sorted(out.glob('*/record.json')):
        name = rec_path.parent.name; dest = rec_path.parent
        if (dest / 'metrics.json').exists():
            print(f'{name}: already evaluated, skipping'); continue
        weights = out / '_runs' / name / 'weights' / 'best.pt'
        model = (RTDETR if name.startswith('rtdetr') else YOLO)(str(weights))
        m = model.val(data=str(DATA / 'data.yaml'), split='test', imgsz=RECIPE['imgsz'], batch=16, conf=0.001, iou=0.7,
                      plots=False, verbose=False, project=str(out / '_val'), name=name, exist_ok=True)
        per_class = {CLASSES[int(c)]: float(m.box.maps[int(c)]) for c in getattr(m.box, 'ap_class_index', [])}
        rows = []                                  # one row per ground-truth object: matched / confused / missed, by size
        for b in range(0, len(test_images), 32):
            batch = test_images[b:b + 32]
            for p, r in zip(batch, model.predict([str(x) for x in batch], conf=REPORT_CONF, imgsz=RECIPE['imgsz'], verbose=False)):
                W, H = Image.open(p).size
                preds = [(bx, int(c)) for bx, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
                gts = []
                lab = DATA / 'labels' / 'test' / (p.stem + '.txt')
                for line in lab.read_text().split('\n') if lab.exists() else []:
                    if line.strip():
                        c, cx, cy, w, h = line.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                        gts.append(([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], int(c)))
                used = set()
                for g, gc in sorted(gts, key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1])):
                    best, bi = 0.0, None
                    for i, (pb, pc) in enumerate(preds):
                        if i not in used and iou(g, pb) > best:
                            best, bi = iou(g, pb), i
                    state = 'missed'
                    if bi is not None and best >= MATCH_IOU:
                        used.add(bi); state = 'detected' if preds[bi][1] == gc else 'confused'
                    area = (g[2] - g[0]) * (g[3] - g[1])
                    rows.append(dict(image=p.name, cls=CLASSES[gc], area=round(area, 1), state=state))
        by_size = {}
        for k, (lo, hi) in SIZE_BINS.items():
            sel = [r for r in rows if lo <= r['area'] < hi]
            by_size[k] = dict(objects=len(sel), recall=sum(r['state'] == 'detected' for r in sel) / max(1, len(sel)),
                              confused=sum(r['state'] == 'confused' for r in sel) / max(1, len(sel)))
        res = dict(model=name, test_images=len(test_images), mAP50_95=float(m.box.map), mAP50=float(m.box.map50),
                   per_class_mAP50_95=per_class, report_conf=REPORT_CONF, match_iou=MATCH_IOU, by_size=by_size,
                   recall_all=sum(r['state'] == 'detected' for r in rows) / max(1, len(rows)),
                   confused_all=sum(r['state'] == 'confused' for r in rows) / max(1, len(rows)))
        (dest / 'objects_test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        (dest / 'metrics.json').write_text(json.dumps(res, indent=1) + '\n')
        print(f"{name}: test mAP50-95 {res['mAP50_95']:.3f}, mAP50 {res['mAP50']:.3f}, recall {res['recall_all']:.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['check', 'prepare', 'train', 'evaluate'])
    ap.add_argument('--only', nargs='*', choices=list(MODELS))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'check': lambda: check(), 'prepare': prepare, 'train': lambda: train(a.only, a.smoke),
     'evaluate': lambda: evaluate(a.smoke)}[a.command]()
