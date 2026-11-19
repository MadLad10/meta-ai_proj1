"""Appendix figure: two VLM failure modes on DUO test images. Top: Qwen2.5-VL-7B answers [] with the default prompt and
returns boxes without the sentence 'Answer [] if there are none'. Bottom: Qwen3-VL-8B keeps listing boxes until the token
limit. Boxes are drawn in the class colors. Output: paper/fig_vlm_failures.pdf (text converted to outlines)."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_examples import image, CLS_COL, SHORT
from assess import O, GT

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'; Z = O / 'job5' / 'duo'


def dets(run, img):
    return [x for x in json.loads((Z / run / 'detections.json').read_text()) if x['image_id'] == img]


def replies(run, img):
    for l in (Z / run / 'replies.jsonl').read_text().splitlines():
        r = json.loads(l)
        if r['image_id'] == img:
            return r['reply']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    g = json.loads(GT.read_text()); ims = {im['id']: im for im in g['images']}
    gt = {}
    for a in g['annotations']:
        gt.setdefault(a['image_id'], []).append(a)
    r11 = replies('qwen2.5-vl-7b_common', 11).strip()
    q80 = dets('qwen3-vl-8b_common_first100', 80); rep80 = replies('qwen3-vl-8b_common_first100', 80)
    assert r11.strip('`').replace('json', '').strip() == '[]' and not rep80.rstrip().endswith(']'), 'unexpected replies'
    panels = [(11, f'(a) Qwen2.5-VL-7B: answers []', []), (11, '(b) same, prompt without the sentence', dets('qwen2.5-vl-7b_common_noempty_first100', 11)),
              (80, '(c) ground truth', gt[80]), (80, f'(d) Qwen3-VL-8B: reply cut off ({len(q80)} boxes)', q80)]
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.25))
    for ax, (i, title, boxes) in zip(axes.flat, panels):
        ax.imshow(image(ims[i]['file_name'])); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6, pad=2)
        for sp in ax.spines.values(): sp.set_linewidth(0.4)
        for bx in boxes:
            x, y, w, h = bx['bbox']; ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=CLS_COL[bx['category_id']], lw=0.8))
    fig.legend(handles=[Patch(fc='none', ec=CLS_COL[k], label=f'sea {SHORT[k]}' if k in (1, 2) else SHORT[k]) for k in (1, 2, 3, 4)],
               ncol=4, loc='lower center', frameon=False, fontsize=6, handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.11, wspace=0.03, hspace=0.2)
    raw = P / 'fig_vlm_failures.raw.pdf'; fig.savefig(raw, dpi=300); plt.close(fig)
    subprocess.run(['gs', '-q', '-o', str(P / 'fig_vlm_failures.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/printer', str(raw)], check=True)
    raw.unlink(); print('written; reply 11 =', json.dumps(r11)[:40], '| image 80 boxes', len(q80))


if __name__ == '__main__':
    main()
