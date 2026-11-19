"""Figure: box-crop naming test. (a) Per-class accuracy of the five zero-shot models when given the true box, with 95%
bootstrap intervals and the 25% chance level. (b) One random crop per class (seed 17, shorter side >= 80 px) with each
model's answer. Inputs: analysis/crops.json, gpu_job/outputs/job8/*/predictions.jsonl, DUO images (Figshare, cached).
Output: paper/fig_crops.pdf (text converted to outlines)."""
import json, random, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_examples import image

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'; J = ROOT / 'gpu_job' / 'outputs' / 'job8'
C = json.loads((ROOT / 'analysis' / 'crops.json').read_text())
G = json.loads((ROOT / 'gpu_job' / 'outputs' / 'gt' / 'duo_instances_test.json').read_text())
CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']; NAME = dict(holothurian='sea cucumber', echinus='sea urchin', scallop='scallop', starfish='starfish')
MODELS = [('Grounding DINO', 'gdino', 'grounding-dino_common', '#0072B2'), ('OWLv2', 'owlv2', 'owlv2_common', '#E69F00'),
          ('Qwen2.5-VL-3B', 'qwen25_3b', 'qwen2.5-vl-3b_common', '#E8A9C9'), ('Qwen2.5-VL-7B', 'qwen25_7b', 'qwen2.5-vl-7b_common', '#CC79A7'),
          ('Qwen3-VL-8B', 'qwen3_8b', 'qwen3-vl-8b_common', '#6A3D9A')]
MIN_SIDE = 150
CLS_COL = {'holothurian': '#E69F00', 'echinus': '#56B4E9', 'scallop': '#CC79A7', 'starfish': '#F0E442'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    preds = {k: {p['ann_id']: p for p in map(json.loads, (J / d / 'predictions.jsonl').read_text().splitlines())} for _, k, d, _ in MODELS}
    ann = {a['id']: a for a in G['annotations']}; ims = {im['id']: im for im in G['images']}
    SHORTM = {'gdino': 'GDINO', 'owlv2': 'OWLv2', 'qwen25_3b': 'Q2.5-3B', 'qwen25_7b': 'Q2.5-7B', 'qwen3_8b': 'Q3-8B'}
    SHORTC = dict(holothurian='cucumber', echinus='urchin', scallop='scallop', starfish='starfish')
    picks = []
    for c in CLASSES:                                            # one random crop per class among the larger ones (display only)
        cand = sorted(i for i, p in preds['gdino'].items() if p['gt'] == c and min(ann[i]['bbox'][2:]) >= MIN_SIDE)
        picks.append(random.Random(17).choice(cand))
    fig = plt.figure(figsize=(7.0, 2.55))
    ax = fig.add_axes([0.065, 0.20, 0.40, 0.60]); w = 0.16
    for j, (lbl, k, _, col) in enumerate(MODELS):
        m = C['models'][k]['per_class']
        for i, c in enumerate(CLASSES):
            x = i + (j - 2) * w; v = m[c]
            ax.bar(x, 100 * v['acc'], w * 0.9, color=col, linewidth=0, label=lbl if i == 0 else None)
            ax.plot([x, x], [100 * v['ci'][0], 100 * v['ci'][1]], color='#333333', lw=0.6)
    ax.axhline(25, color='#555555', lw=0.7, ls=(0, (3, 2))); ax.text(-0.48, 26.5, 'chance', ha='left', va='bottom', fontsize=6, color='#555555')
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{NAME[c]}\n(n={C["per_class_n"][c]})' for c in CLASSES])
    ax.set_ylabel('% of crops named correctly'); ax.set_ylim(0, 100); ax.yaxis.grid(True, lw=0.3, color='#cccccc'); ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8, fontsize=6)
    fig.text(0.005, 0.955, '(a)', fontsize=7, weight='bold'); fig.text(0.505, 0.955, '(b)', fontsize=7, weight='bold')
    for n, aid in enumerate(picks):
        a = ann[aid]; im = image(ims[a['image_id']]['file_name']); x, y, bw, bh = a['bbox']; mx, my = 0.1 * bw, 0.1 * bh
        crop = im.crop((max(0, int(x - mx)), max(0, int(y - my)), min(im.width, int(x + bw + mx)), min(im.height, int(y + bh + my))))
        tc = CLASSES[a['category_id'] - 1]; left = 0.525 + n * 0.12
        iax = fig.add_axes([left, 0.50, 0.105, 0.38]); iax.imshow(crop, aspect='auto'); iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values(): sp.set_edgecolor(CLS_COL[tc]); sp.set_linewidth(1.6)
        iax.set_title(f'true: {SHORTC[tc]}', fontsize=6.5, weight='bold', pad=2)
        for r, (lbl, k, _, _) in enumerate(MODELS):
            pr = preds[k][aid]['pred']; ok = pr == tc
            fig.text(left, 0.45 - r * 0.068, f"{'✓' if ok else '✗'} {SHORTM[k]}: {SHORTC.get(pr, 'none')}", fontsize=5.6,
                     color='#1B7837' if ok else '#C51B7D', va='top')
    raw = P / 'fig_crops.raw.pdf'; fig.savefig(raw, dpi=300); plt.close(fig)
    subprocess.run(['gs', '-q', '-o', str(P / 'fig_crops.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/printer', str(raw)], check=True)
    raw.unlink(); print('written', P / 'fig_crops.pdf', 'crops', picks)


if __name__ == '__main__':
    main()
