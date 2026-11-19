"""Figure: DUO test images with what each model does with every ground-truth object (found and named / found but
misnamed / missed), using the same definition as the per-object measures (best-overlapping box among the image's 100
highest-scoring boxes, IoU >= 0.5). Images are read from DUO's official Figshare zip (CC BY 4.0) and cached locally.
Output: paper/fig_examples.pdf (text converted to outlines).
"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess import O, GT, RUNS, iou, by_image

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'; CACHE = O / 'duo_images'
EXAMPLES = [33, 77, 43]                                   # chosen with pick_examples.py (first 100 test images)
MODELS = [('Ground truth', None), ('YOLO11n (fine-tuned)', 'yolo11n'), ('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2'), ('Qwen3-VL-8B', 'qwen3_8b_100')]
SHORT = {1: 'cucumber', 2: 'urchin', 3: 'scallop', 4: 'starfish'}
CLS_COL = {1: '#E69F00', 2: '#56B4E9', 3: '#CC79A7', 4: '#F0E442'}
OK, BAD, MISS = '#00C853', '#FF1744', 'white'


def image(file_name):
    CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / file_name
    if not f.exists():
        from remote_zip import open_duo
        f.write_bytes(open_duo().read(f'DUO/DUO/images/test/{file_name}'))
    from PIL import Image
    return Image.open(f).convert('RGB')


def outcome(gts, dets, top=100):
    ds = sorted(dets, key=lambda x: -x['score'])[:top]
    M = iou(np.array([a['bbox'] for a in gts], float).reshape(-1, 4), np.array([x['bbox'] for x in ds], float).reshape(-1, 4))
    out = []
    for j in range(len(gts)):
        hit = np.where(M[j] >= 0.5)[0] if M.shape[1] else []
        out.append(None if not len(hit) else ds[hit[np.argmax(M[j, hit])]])
    return out


def view(gts, w, h):
    """Crop around the objects (12% margin), widened to 16:9 like the frames, inside the image."""
    x0 = min(a['bbox'][0] for a in gts); y0 = min(a['bbox'][1] for a in gts)
    x1 = max(a['bbox'][0] + a['bbox'][2] for a in gts); y1 = max(a['bbox'][1] + a['bbox'][3] for a in gts)
    mx, my = 0.12 * (x1 - x0) + 0.03 * w, 0.12 * (y1 - y0) + 0.03 * h
    x0, y0, x1, y1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < 16 / 9: d = (16 / 9 * ch - cw) / 2; x0 -= d; x1 += d
    else: d = (9 / 16 * cw - ch) / 2; y0 -= d; y1 += d
    if x1 - x0 > w: x0, x1 = 0, w
    if y1 - y0 > h: y0, y1 = 0, h
    sx = min(0, w - x1) + max(0, -x0); sy = min(0, h - y1) + max(0, -y0)
    return x0 + sx, y0 + sy, x1 + sx, y1 + sy


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    g = json.loads(GT.read_text()); ims = {im['id']: im for im in g['images']}
    gt = {}
    for a in g['annotations']:
        gt.setdefault(a['image_id'], []).append(a)
    D = {k: by_image([x for x in json.loads((RUNS[k] / 'detections.json').read_text()) if x['image_id'] in set(EXAMPLES)]) for _, k in MODELS if k}
    fig, axes = plt.subplots(len(EXAMPLES), len(MODELS), figsize=(7.0, 0.80 * len(EXAMPLES) + 0.40))
    for r, i in enumerate(EXAMPLES):
        im = ims[i]; gs = gt[i]; img = image(im['file_name']); x0, y0, x1, y1 = view(gs, im['width'], im['height'])
        lw = 1.0
        for c, (title, k) in enumerate(MODELS):
            ax = axes[r, c]; ax.imshow(img); ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_linewidth(0.4)
            if r == 0: ax.set_title(title, fontsize=7, pad=3)
            if c == 0:
                for a in gs:
                    bx, by, bw, bh = a['bbox']
                    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=CLS_COL[a['category_id']], lw=lw))
                continue
            placed = []                                   # label boxes already drawn (data units), to avoid stacking identical labels
            th = (5.5 / 72) / 0.78 * (y1 - y0)            # approximate label height in data units
            for a, d in zip(gs, outcome(gs, D[k].get(i, []))):
                bx, by, bw, bh = a['bbox']
                if d is None:
                    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=MISS, lw=0.8, ls=(0, (2, 1.5))))
                    continue
                px, py, pw, ph = d['bbox']; good = d['category_id'] == a['category_id']
                ax.add_patch(Rectangle((px, py), pw, ph, fill=False, ec=OK if good else BAD, lw=lw))
                if not good:
                    txt = SHORT[d['category_id']]; tw = 0.62 * th * len(txt)
                    ty = py if py - th > y0 else py + ph + th            # above the box, or below it at the top edge
                    if any(abs(px - qx) < tw and abs(ty - qy) < th for qx, qy in placed):
                        continue
                    placed.append((px, ty))
                    ax.text(max(px, x0), ty, txt, color='white', fontsize=5.5, va='bottom', ha='left',
                            bbox=dict(boxstyle='square,pad=0.12', fc=BAD, ec='none'))
    handles = [Patch(fc='none', ec=CLS_COL[c], label=f'true {SHORT[c]}') for c in (1, 2, 3, 4)] + [
        Line2D([], [], color=OK, lw=1.2, label='found, correct class'), Line2D([], [], color=BAD, lw=1.2, label='found, wrong class (label shown)'),
        Line2D([], [], color='#555555', lw=0.9, ls=(0, (2, 1.5)), label='missed')]
    fig.legend(handles=handles, ncol=7, loc='lower center', frameon=False, fontsize=6, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.003, right=0.997, top=0.915, bottom=0.085, wspace=0.025, hspace=0.05)
    raw = P / 'fig_examples.raw.pdf'; fig.savefig(raw, dpi=300); plt.close(fig)
    subprocess.run(['gs', '-q', '-o', str(P / 'fig_examples.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/printer', str(raw)], check=True)
    raw.unlink(); print('written', P / 'fig_examples.pdf')


if __name__ == '__main__':
    main()
