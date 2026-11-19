"""Job 5 (track B): zero-shot open-vocabulary detectors and VLMs on established underwater detection benchmarks.

Benchmarks: DUO test set (1,111 images; needs job4_duo.py prepare) and RUOD test set (needs job6_ruod.py prepare; VLMs
use a fixed 1,000-image subset because VLM inference is slow). No training and no threshold tuning: every model is
scored with the standard COCO protocol (pycocotools) and its errors are broken down with TIDE (Bolya et al., ECCV 2020).

    python job5_duo_zeroshot.py --limit 20                                  # smoke test, DUO, all models
    python job5_duo_zeroshot.py                                             # DUO, all models
    python job5_duo_zeroshot.py --dataset ruod                              # RUOD
    python job5_duo_zeroshot.py --names latin --only grounding-dino owlv2   # ablation: Latin class names
    python job5_duo_zeroshot.py --pixels low --only qwen2.5-vl-7b           # ablation: VLM image resolution (low / high)
    python job5_duo_zeroshot.py --prompt noempty --first 100 --only qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-8b   # ablation: no 'Answer []' sentence
    python job5_duo_zeroshot.py --gdino text --only grounding-dino          # comparison: Hugging Face phrase decoding
Grounding DINO is scored by default ('posmap') as in its own COCO evaluation: class scores from the token logits of each
class name, top 300 (query, class) pairs per image, so no box is lost to unmatched phrases (folder suffix _posmap).
Output: outputs/job5/<dataset>/<model>_<names>[_<pixels>]/{detections.json, metrics.json, record.json}
"""
import argparse, importlib.util, inspect, json, platform, random, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('job4', HERE / 'job4_duo.py'); job4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(job4)
OUT = HERE / 'outputs' / 'job5'
COMMON = {'holothurian': 'sea cucumber', 'echinus': 'sea urchin'}          # other class names are already common English
DETECTORS = {'grounding-dino': 'IDEA-Research/grounding-dino-base', 'owlv2': 'google/owlv2-base-patch16-ensemble'}
VLMS = {'qwen2.5-vl-3b': ('Qwen/Qwen2.5-VL-3B-Instruct', 'absolute'),      # boxes in pixels of the resized input
        'qwen2.5-vl-7b': ('Qwen/Qwen2.5-VL-7B-Instruct', 'absolute'),
        'qwen3-vl-8b': ('Qwen/Qwen3-VL-8B-Instruct', 'relative1000')}      # boxes on a 0-1000 grid of the image
PIXELS = {'low': 640 * 28 * 28, 'default': 1280 * 28 * 28, 'high': 2560 * 28 * 28}
MIN_SCORE = 0.01
GDINO_TOPK = 300                       # (query, class) pairs kept per image, as in Grounding DINO's COCO evaluation
RUOD_VLM_SUBSET, SUBSET_SEED = 1000, 17


def use_dataset(name):
    if name == 'ruod':
        job4.RAW = HERE / 'ruod_raw'; job4.DATA = HERE / 'ruod_yolo'


def prompt_names(categories, mode):
    names = [c['name'].strip().lower() for c in sorted(categories, key=lambda c: c['id'])]
    return {n: (COMMON.get(n, n) if mode == 'common' else n) for n in names}




def class_for(label, names):
    """Map a predicted phrase to a benchmark class. A phrase containing exactly one class's full prompt name maps to
    that class; otherwise a distinctive word (singular or plural) of exactly one name, e.g. Grounding DINO's fragment
    'cucumber'. Phrases naming two classes, or only shared words such as 'sea', map to None and are dropped."""
    label = str(label).lower()
    full = {cls for cls, name in names.items() if re.search(r'\b' + re.escape(name) + r'\b', label)}
    if len(full) == 1:
        return full.pop()
    if len(full) > 1:
        return None
    stem = lambda w: w[:-1] if w.endswith('s') and len(w) > 3 else w
    words = {stem(w) for w in re.findall(r'[a-z]+', label)}
    counts = {}
    for name in names.values():
        for w in {stem(x) for x in name.split()}:
            counts[w] = counts.get(w, 0) + 1
    hits = {cls for cls, name in names.items() for w in {stem(x) for x in name.split()}
            if w in words and counts[w] == 1 and w not in {'sea', 'a', 'an', 'the', 'of'}}
    return hits.pop() if len(hits) == 1 else None


def vlm_box(raw, mode, img_wh, resized_wh):
    """Convert a VLM box to pixels of the original image."""
    (W, H), (rw, rh) = img_wh, resized_wh
    x0, y0, x1, y1 = raw
    if mode == 'relative1000':
        return [x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000]
    return [x0 * W / rw, y0 * H / rh, x1 * W / rw, y1 * H / rh]


def parse_vlm(reply):
    """[(x0, y0, x1, y1), label] from a JSON-like list of {"bbox_2d": [...], "label": "..."} objects."""
    num = r'\s*(-?\d+(?:\.\d+)?)\s*'
    out = []
    for m in re.finditer(r'\{[^{}]*?\}', reply):
        obj = m.group(0)
        b = re.search(r'\[' + ','.join([num] * 4) + r'\]', obj); lab = re.search(r'"label"\s*:\s*"([^"]+)"', obj)
        if b and lab:
            out.append(([float(v) for v in b.groups()], lab.group(1)))
    return out


def class_token_map(caption, class_names, offsets):
    """Token positions of each class name in the caption (from the tokenizer's character offsets), in class order.
    Names are searched left to right, so 'fish' after 'starfish' finds its own word."""
    rows, pos = [], 0
    for nm in class_names:
        s = caption.index(nm, pos); e = s + len(nm); pos = e
        toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e]
        if not toks:
            raise ValueError(f'no tokens for class name {nm!r} in {caption!r}')
        rows.append(toks)
    return rows


def run_detector(name, images, names, device, gdino='posmap'):
    import torch
    if name == 'grounding-dino':
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection as M
        proc = AutoProcessor.from_pretrained(DETECTORS[name])
    else:
        from transformers import Owlv2Processor as P, Owlv2ForObjectDetection as M
        proc = P.from_pretrained(DETECTORS[name])
    model = M.from_pretrained(DETECTORS[name]).to(device).eval()
    classes = list(names); posmap = None
    for img in images:
        raw = None
        with torch.no_grad():
            if name == 'grounding-dino':
                caption = '. '.join(names[c] for c in classes) + '.'
                inputs = proc(images=img, text=caption, return_tensors='pt').to(device)
                out = model(**inputs)
                if gdino == 'posmap':
                    # Grounding DINO's own COCO evaluation: a class's score is the mean token probability over the tokens of its
                    # name (positive map), and the top GDINO_TOPK (query, class) pairs are kept. No text decoding, so no box
                    # can be left without a class.
                    if posmap is None:
                        enc = proc.tokenizer(caption, return_offsets_mapping=True)
                        if enc['input_ids'] != inputs.input_ids[0].tolist():
                            raise SystemExit('Grounding DINO: tokenizer ids differ from the processor ids; stop and report.')
                        rows = class_token_map(caption, [names[c] for c in classes], enc['offset_mapping'])
                        posmap = torch.zeros(len(classes), out.logits.shape[-1], device=out.logits.device)
                        for c, toks in enumerate(rows):
                            posmap[c, toks] = 1.0 / len(toks)
                        print('grounding-dino class tokens:', json.dumps({names[c]: proc.tokenizer.convert_ids_to_tokens(
                            [enc['input_ids'][t] for t in toks]) for c, toks in zip(classes, rows)}), flush=True)
                    prob = out.logits[0].float().sigmoid() @ posmap.T                 # (queries, classes)
                    top = torch.topk(prob.flatten(), min(GDINO_TOPK, prob.numel()))
                    q, k = top.indices // prob.shape[1], top.indices % prob.shape[1]
                    cx, cy, w, h = out.pred_boxes[0][q].float().unbind(-1)             # normalised centre / size
                    W, H = img.width, img.height
                    r = dict(boxes=torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1),
                             scores=top.values)
                    labels = [classes[int(i)] for i in k.tolist()]
                else:                    # the Hugging Face pipeline: decode a phrase per box, then match it to a class
                    post = proc.post_process_grounded_object_detection
                    kw = dict(text_threshold=0.1, target_sizes=[(img.height, img.width)])
                    kw['threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'] = 0.1
                    r = post(out, inputs.input_ids, **kw)[0]
                    raw = [str(l) for l in r.get('text_labels', r.get('labels'))]   # saved to replies.jsonl
                    labels = [class_for(l, names) for l in raw]
            else:
                inputs = proc(text=[[f'a photo of a {names[c]}' for c in classes]], images=img, return_tensors='pt').to(device)
                out = model(**inputs); side = max(img.height, img.width)        # OWLv2 pads to a square
                post = getattr(proc, 'post_process_object_detection', None) or proc.post_process_grounded_object_detection
                r = post(outputs=out, threshold=MIN_SCORE, target_sizes=[(side, side)])[0]
                labels = [classes[int(l)] for l in r['labels'].tolist()]
        yield [(b, float(s), l) for b, s, l in zip(r['boxes'].tolist(), r['scores'].tolist(), labels)], raw


def run_vlm(name, images, names, pixels, prompt_variant='default'):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    model_id, mode = VLMS[name]
    model = VLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
    proc = AutoProcessor.from_pretrained(model_id, max_pixels=PIXELS[pixels])
    listing = ', '.join(names.values())
    coords = 'on a 0-1000 grid of the image (0,0 top-left, 1000,1000 bottom-right)' if mode == 'relative1000' else 'in pixel coordinates of the image'
    prompt = (f'Detect every {listing} in this underwater image. Answer with JSON only: a list of objects, each '
              f'{{"bbox_2d": [x1, y1, x2, y2], "label": one of [{listing}]}}, with coordinates {coords}.' +
              (' Answer [] if there are none.' if prompt_variant == 'default' else ''))
    for img in images:
        msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        reply = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        g = inputs['image_grid_thw'][0].tolist(); resized = (int(g[2]) * 14, int(g[1]) * 14)
        dets = [(vlm_box(b, mode, (img.width, img.height), resized), 1.0, class_for(l, names)) for b, l in parse_vlm(reply)]
        yield dets, reply                                   # VLMs give no score: every detection scores 1.0


def answer_all(ims, make_gen, part, tag):
    """Run the model over the images; returns ({image_id: dict(found, reply, sec)}, number of images taken from an earlier run).
    With `part` (a path), every answer is appended to it at once, and images already in it are not asked again, so a rerun
    of an interrupted run continues where it stopped. A last line cut off by a crash is dropped."""
    done = {}
    if part is not None and part.exists():
        good = []
        for line in part.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line); done[r['image_id']] = r; good.append(line)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        part.write_text(''.join(l + '\n' for l in good), encoding='utf-8')
        print(f'{tag}: resuming, {len(done)} of {len(ims)} images already answered', flush=True)
    todo = [im for im in ims if im['id'] not in done]
    out = open(part, 'a', encoding='utf-8') if part is not None else None
    recs = dict(done); t0 = t_last = time.monotonic()
    try:
        for n, (im, (found, reply)) in enumerate(zip(todo, make_gen(todo) if todo else iter(()))):
            now = time.monotonic()
            r = dict(image_id=im['id'], found=[[list(b), float(s), c] for b, s, c in found], reply=reply, sec=now - t_last)
            t_last = now; recs[im['id']] = r
            if out is not None:
                out.write(json.dumps(r) + '\n'); out.flush()
            if n % 100 == 0:
                print(json.dumps(dict(model=tag, done=len(done) + n, of=len(ims), minutes=round((now - t0) / 60, 1))), flush=True)
    finally:
        if out is not None:
            out.close()
    missing = [im['id'] for im in ims if im['id'] not in recs]
    if missing:
        raise SystemExit(f'{tag}: {len(missing)} images got no answer; rerun the same command to continue.')
    return recs, len(done)


def evaluate(gt_path, det_path, cat_by_name):
    """COCO metrics (pycocotools), per-class AP, and the TIDE error breakdown."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dets = json.loads(Path(det_path).read_text())
    if not dets:
        return dict(AP=0.0, note='no detections')
    G = COCO(str(gt_path)); D = G.loadRes(str(det_path)); E = COCOeval(G, D, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    keys = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    res = dict(zip(keys, [float(v) for v in E.stats])); per = {}
    for name, cid in cat_by_name.items():
        e = COCOeval(G, D, 'bbox'); e.params.catIds = [cid]; e.evaluate(); e.accumulate(); e.summarize(); per[name] = float(e.stats[0])
    res['per_class_AP'] = per
    try:
        from tidecv import TIDE, datasets
        def boxpoly(x):            # TIDE's loader requires a 'segmentation' field; a polygon equal to the box changes nothing in box mode.
            # Always built from the box: DUO's own 'segmentation' is a flat list, not a list of polygons, and pycocotools rejects it.
            bx, by, bw, bh = x['bbox']; return dict(x, segmentation=[[bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]])
        g = json.loads(Path(gt_path).read_text()); g['annotations'] = [boxpoly(x) for x in g['annotations']]
        tg = Path(det_path).with_name('_tide_gt.json'); tg.write_text(json.dumps(g))
        td = Path(det_path).with_name('_tide_det.json'); td.write_text(json.dumps([boxpoly(x) for x in dets]))
        t = TIDE(); t.evaluate(datasets.COCO(str(tg)), datasets.COCOResult(str(td)), mode=TIDE.BOX)
        main = t.get_main_errors(); special = t.get_special_errors()
        res['tide_main_errors'] = {k: float(v) for k, v in next(iter(main.values())).items()}
        res['tide_special_errors'] = {k: float(v) for k, v in next(iter(special.values())).items()}
    except Exception as ex:                                  # TIDE is an extra; COCO metrics stand without it
        res['tide_error'] = repr(ex)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='duo', choices=['duo', 'ruod'])
    ap.add_argument('--only', nargs='*', choices=list(DETECTORS) + list(VLMS))
    ap.add_argument('--names', default='common', choices=['common', 'latin'])
    ap.add_argument('--pixels', default='default', choices=list(PIXELS))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--prompt', default='default', choices=['default', 'noempty'],
                    help="VLM prompt ablation: 'noempty' drops the sentence 'Answer [] if there are none.'")
    ap.add_argument('--first', type=int, default=0, help='run only the first N test images, as a real run (for ablations)')
    ap.add_argument('--gdino', default='posmap', choices=['posmap', 'text'],
                    help="Grounding DINO scoring: 'posmap' = its own COCO evaluation (primary); 'text' = Hugging Face phrase decoding")
    a = ap.parse_args()
    import torch, transformers
    from PIL import Image
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU visible to PyTorch; stopping.')
    use_dataset(a.dataset)
    files = job4.coco_files()
    if 'test' not in files:
        raise SystemExit(f'{a.dataset} test annotations not found: run the prepare step of Job 4 (DUO) or Job 6 (RUOD) first.')
    gt_full_path, d = files['test']
    cat_by_name = {c['name'].strip().lower(): c['id'] for c in d['categories']}
    names = prompt_names(d['categories'], a.names)
    test_dir = job4.DATA / 'images' / 'test'
    all_ims = [im for im in d['images'] if (test_dir / Path(im['file_name']).name).exists()]
    if len(all_ims) != len(d['images']):
        raise SystemExit(f'{len(d["images"]) - len(all_ims)} test images missing from {test_dir}: run prepare first.')
    env = dict(python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0), transformers=transformers.__version__)
    print(json.dumps(env))
    root = OUT / a.dataset; root.mkdir(parents=True, exist_ok=True)
    for name in list(DETECTORS) + list(VLMS):
        if a.only and name not in a.only:
            continue
        is_vlm = name in VLMS
        ims = all_ims
        if is_vlm and a.dataset == 'ruod' and not a.limit:
            ims = sorted(random.Random(SUBSET_SEED).sample(all_ims, min(RUOD_VLM_SUBSET, len(all_ims))), key=lambda im: im['id'])
        if a.limit or a.first:
            ims = all_ims[:a.limit or a.first]
        gt_path = gt_full_path
        if ims is not all_ims or a.limit:
            ids = {i['id'] for i in ims}
            gt_path = root / f'_gt_{len(ims)}{"_smoke" if a.limit else ""}.json'
            gt_path.write_text(json.dumps(dict(d, images=ims, annotations=[x for x in d['annotations'] if x['image_id'] in ids])))
        tag = f'{name}_{a.names}' + (f'_{a.pixels}' if is_vlm and a.pixels != 'default' else '') + \
              (f'_{a.prompt}' if is_vlm and a.prompt != 'default' else '') + (f'_first{a.first}' if a.first else '') + \
              ('_posmap' if name == 'grounding-dino' and a.gdino == 'posmap' else '')   # 'text' keeps the earlier folder name
        dest = root / (('_smoke_' if a.limit else '') + tag); dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'metrics.json').exists():
            print(f'{tag}: done, skipping'); continue
        t0 = time.monotonic(); dets = []; unknown = 0; outside = 0; replies = []
        def make_gen(todo):
            imgs = (Image.open(test_dir / Path(im['file_name']).name).convert('RGB') for im in todo)
            return run_vlm(name, imgs, names, a.pixels, a.prompt) if is_vlm else run_detector(name, imgs, names, 'cuda', a.gdino)
        # VLM runs are slow (up to 16 s per image): each answer is saved at once to partial.jsonl, and a rerun of the same
        # command continues where an interrupted run stopped. Images run in test-set order, so a stopped run covers the first K.
        recs, resumed = answer_all(ims, make_gen, dest / 'partial.jsonl' if is_vlm else None, tag)
        for im in ims:                                       # test-set order, so the result does not depend on where a run resumed
            found, reply = recs[im['id']]['found'], recs[im['id']]['reply']
            if reply is not None:
                replies.append(dict(image_id=im['id'], reply=reply))
            for (x0, y0, x1, y1), s, cls in found:
                if cls is None:
                    unknown += 1; continue
                if x1 < -2 or y1 < -2 or x0 > im['width'] + 2 or y0 > im['height'] + 2:
                    outside += 1
                x0, y0 = max(0.0, x0), max(0.0, y0); x1, y1 = min(float(im['width']), x1), min(float(im['height']), y1)
                if s >= MIN_SCORE and x1 > x0 and y1 > y0:
                    dets.append(dict(image_id=im['id'], category_id=cat_by_name[cls], bbox=[x0, y0, x1 - x0, y1 - y0], score=s))
        (dest / 'detections.json').write_text(json.dumps(dets))
        if replies:
            (dest / 'replies.jsonl').write_text('\n'.join(json.dumps(r) for r in replies) + '\n')
        m = evaluate(gt_path, dest / 'detections.json', cat_by_name)
        m.update(model=name, dataset=a.dataset, names=a.names, pixels=a.pixels if is_vlm else None, prompt=a.prompt if is_vlm else None,
                 gdino_scoring=a.gdino if name == 'grounding-dino' else None,
                 prompt_names=names, images=len(ims),
                 detections=len(dets), unmatched_labels=unknown, boxes_outside_image=outside, min_score=MIN_SCORE,
                 model_id=VLMS[name][0] if is_vlm else DETECTORS[name])
        (dest / 'metrics.json').write_text(json.dumps(m, indent=1) + '\n')
        secs = [recs[im['id']]['sec'] for im in ims]
        (dest / 'record.json').write_text(json.dumps(dict(wall_s=time.monotonic() - t0, model_s_sum=sum(secs),
                                                          model_s_median=sorted(secs)[len(secs) // 2], resumed_images=resumed, env=env), indent=1) + '\n')
        print(f"{tag}: AP {m.get('AP', 0):.3f}  AP50 {m.get('AP50', 0):.3f}  detections {len(dets)}  unmatched labels {unknown}  boxes outside image {outside}")
        if a.limit and is_vlm and replies:
            print('  first reply:', replies[0]['reply'][:300])
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
