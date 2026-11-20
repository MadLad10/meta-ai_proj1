"""Same-image vocabulary test (CPU): Grounding DINO and OWLv2 on the same 1,117 DUO crops as the GPU box-crop test,
with different class lists. Replicates job8_crop_naming.py's crop and scoring rules exactly, and also records whether
the winning box covers the object inside the crop.
    python analysis/vocab_cpu.py <model> <vocab> [--limit N]     model: gdino | owlv2;  vocab: duo4 | ruod10 | duo4rev
Writes analysis/vocab_cpu/<model>_<vocab>.jsonl (resumable, one line per crop)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'gpu_job' / 'outputs'; OUT = ROOT / 'analysis' / 'vocab_cpu'
MARGIN, MIN_SIDE = 0.10, 56                                  # as in job8_crop_naming.py
VOCABS = {'duo4': ['sea cucumber', 'sea urchin', 'scallop', 'starfish'],
          'duo4rev': ['starfish', 'scallop', 'sea urchin', 'sea cucumber'],
          'ruod10': ['sea cucumber', 'sea urchin', 'scallop', 'starfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']}
IDS = {'grounding-dino': 'IDEA-Research/grounding-dino-base', 'owlv2': 'google/owlv2-base-patch16-ensemble'}


def crop(img, bbox):
    x, y, w, h = bbox; mx, my = MARGIN * w, MARGIN * h
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my)); x1, y1 = min(img.width, int(round(x + w + mx))), min(img.height, int(round(y + h + my)))
    c = img.crop((x0, y0, x1, y1)); s = 1.0
    if min(c.size) < MIN_SIDE:
        s = MIN_SIDE / min(c.size); c = c.resize((max(MIN_SIDE, round(c.width * s)), max(MIN_SIDE, round(c.height * s))), 3)
    obj = ((x - x0) * c.width / (x1 - x0), (y - y0) * c.height / (y1 - y0), w * c.width / (x1 - x0), h * c.height / (y1 - y0))
    return c, obj


def iou(a, b):                                               # xywh
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy; return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model', choices=['gdino', 'owlv2']); ap.add_argument('vocab', choices=list(VOCABS))
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--per_class', type=int, default=0, help='stratified random sample per class (seed 17)'); a = ap.parse_args()
    import torch
    from PIL import Image
    from remote_zip import open_duo
    cache = O / 'duo_images'; cache.mkdir(exist_ok=True); zf = [None]
    def image(fn):                                           # DUO test image from the official Figshare zip, cached
        f = cache / fn
        if not f.exists():
            zf[0] = zf[0] or open_duo(); f.write_bytes(zf[0].read(f'DUO/DUO/images/test/{fn}'))
        return Image.open(f).convert('RGB')
    import job5_shim as job5
    torch.set_num_threads(a.threads)
    names = VOCABS[a.vocab]
    g = json.loads((O / 'gt' / 'duo_instances_test.json').read_text()); ann = {x['id']: x for x in g['annotations']}; ims = {im['id']: im for im in g['images']}
    gpu = {json.loads(l)['ann_id']: json.loads(l) for l in (O / 'job8' / 'grounding-dino_common' / 'predictions.jsonl').read_text().splitlines()}
    ids = sorted(gpu)[:a.limit] if a.limit else sorted(gpu)
    if a.per_class:
        import random
        ids = sorted(i for c in sorted({v['gt'] for v in gpu.values()}) for i in random.Random(17).sample(sorted(k for k, v in gpu.items() if v['gt'] == c), a.per_class))
    OUT.mkdir(exist_ok=True); dest = OUT / f'{a.model}_{a.vocab}{"_limit" + str(a.limit) if a.limit else ""}{"_pc" + str(a.per_class) if a.per_class else ""}.jsonl'
    done = {json.loads(l)['ann_id'] for l in dest.read_text().splitlines() if l.strip()} if dest.exists() else set()
    mid = IDS['grounding-dino' if a.model == 'gdino' else 'owlv2']
    if a.model == 'gdino':
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection as M
        proc = AutoProcessor.from_pretrained(mid); model = M.from_pretrained(mid).eval()
        caption = '. '.join(names) + '.'; enc = proc.tokenizer(caption, return_offsets_mapping=True)
        rows = job5.class_token_map(caption, names, enc['offset_mapping'])
    else:
        from transformers import Owlv2Processor as Pc, Owlv2ForObjectDetection as M
        proc = Pc.from_pretrained(mid); model = M.from_pretrained(mid).eval()
    with dest.open('a') as f:
        for n, aid in enumerate(ids):
            if aid in done:
                continue
            x = ann[aid]; img = image(ims[x['image_id']]['file_name']); c, obj = crop(img, x['bbox'])
            with torch.no_grad():
                if a.model == 'gdino':
                    inputs = proc(images=c, text=caption, return_tensors='pt'); out = model(**inputs)
                    pm = torch.zeros(len(names), out.logits.shape[-1])
                    for k, toks in enumerate(rows):
                        pm[k, toks] = 1.0 / len(toks)
                    prob = out.logits[0].float().sigmoid() @ pm.T                       # queries x classes
                    s = prob.max(0).values; k = int(s.argmax()); q = int(prob[:, k].argmax())
                    cx, cy, w, h = out.pred_boxes[0][q].tolist(); W, H = c.width, c.height
                else:
                    inputs = proc(text=[[f'a photo of a {nm}' for nm in names]], images=c, return_tensors='pt'); out = model(**inputs)
                    lg = out.logits[0].float().sigmoid(); s = lg.max(0).values; k = int(s.argmax()); q = int(lg[:, k].argmax())
                    cx, cy, w, h = out.pred_boxes[0][q].tolist(); W = H = max(c.width, c.height)   # OWLv2 pads to a square
                box = ((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H)
            f.write(json.dumps(dict(ann_id=aid, gt=gpu[aid]['gt'], pred=names[k], scores=dict(zip(names, map(float, s))), box_iou=iou(box, obj))) + '\n'); f.flush()
            if n % 100 == 0:
                print(a.model, a.vocab, n, '/', len(ids), flush=True)


if __name__ == '__main__':
    main()
