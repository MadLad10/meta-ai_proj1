"""Check of the box geometry in vocab_cpu.py on a few crops: the winning Grounding DINO box computed there must equal
the box from the Hugging Face post-processing for the same query, and the drawn boxes are saved for a visual check.
    python analysis/vocab_boxcheck.py <out_dir> [ann_id ...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from PIL import Image, ImageDraw
import job5_shim as job5
from vocab_cpu import crop, iou, O, IDS
from remote_zip import open_duo
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection as M

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True); want = [int(a) for a in sys.argv[2:]] or [48, 84, 97, 1098]
names = ['sea cucumber', 'sea urchin', 'scallop', 'starfish']; torch.set_num_threads(10)
g = json.loads((O / 'gt' / 'duo_instances_test.json').read_text()); ann = {x['id']: x for x in g['annotations']}; ims = {im['id']: im for im in g['images']}
proc = AutoProcessor.from_pretrained(IDS['grounding-dino']); model = M.from_pretrained(IDS['grounding-dino']).eval()
caption = '. '.join(names) + '.'; rows = job5.class_token_map(caption, names, proc.tokenizer(caption, return_offsets_mapping=True)['offset_mapping'])
for aid in want:
    x = ann[aid]; f = O / 'duo_images' / ims[x['image_id']]['file_name']
    img = Image.open(f).convert('RGB') if f.exists() else Image.open(__import__('io').BytesIO(open_duo().read('DUO/DUO/images/test/' + f.name))).convert('RGB')
    c, obj = crop(img, x['bbox'])
    with torch.no_grad():
        inputs = proc(images=c, text=caption, return_tensors='pt'); out = model(**inputs)
    pm = torch.zeros(len(names), out.logits.shape[-1])
    for k, toks in enumerate(rows):
        pm[k, toks] = 1.0 / len(toks)
    prob = out.logits[0].float().sigmoid() @ pm.T; k = int(prob.max(0).values.argmax()); q = int(prob[:, k].argmax())
    cx, cy, w, h = out.pred_boxes[0][q].tolist(); W, H = c.width, c.height
    mine = ((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H)
    hf = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=-1.0, text_threshold=1.1, target_sizes=[(H, W)])[0]['boxes'][q].tolist()
    print(aid, x['bbox'], 'crop', c.size, 'pred', names[k], 'mine xyxy', [round(v, 1) for v in (mine[0], mine[1], mine[0] + mine[2], mine[1] + mine[3])],
          'hf xyxy', [round(v, 1) for v in hf], 'iou with object', round(iou(mine, obj), 3))
    d = ImageDraw.Draw(c); d.rectangle([obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3]], outline=(0, 255, 0), width=2)
    d.rectangle([mine[0], mine[1], mine[0] + mine[2], mine[1] + mine[3]], outline=(255, 0, 0), width=2); c.save(out_dir / f'crop_{aid}.png')
