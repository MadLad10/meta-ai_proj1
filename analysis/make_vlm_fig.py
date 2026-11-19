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
    rows = [(11, [('Ground truth', gt[11]), ('Qwen2.5-VL-7B, default prompt: answers []', []),
                  ('Qwen2.5-VL-7B, prompt without the sentence', dets('qwen2.5-vl-7b_common_noempty_first100', 11))]),
            (80, [('Ground truth', gt[80]), (f'Qwen3-VL-8B: reply cut off ({len(q80)} boxes)', q80), None])]
    assert r11.strip('`').replace('json', '').strip() == '[]' and not rep80.rstrip().endswith(']'), 'unexpected replies'
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 2.75))
    for r, (i, panels) in enumerate(rows):
        img = image(ims[i]['file_name'])
        for c in range(3):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if panels[c] is None:
                ax.axis('off'); continue
            title, boxes = panels[c]
            ax.imshow(img); ax.set_title(title, fontsize=6.3, pad=2)
            for b in boxes:
                x, y, w, h = b['bbox']; ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=CLS_COL[b['category_id']], lw=0.9))
    axes[1, 2].legend(handles=[Patch(fc='none', ec=CLS_COL[k], label=f'sea {SHORT[k]}' if k in (1, 2) else SHORT[k]) for k in (1, 2, 3, 4)],
                      ncol=2, loc='center', frameon=False, fontsize=6.5, title='box class', title_fontsize=6.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.18)
    raw = P / 'fig_vlm_failures.raw.pdf'; fig.savefig(raw, dpi=300); plt.close(fig)
    subprocess.run(['gs', '-q', '-o', str(P / 'fig_vlm_failures.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', '-dPDFSETTINGS=/printer', str(raw)], check=True)
    raw.unlink(); print('written; reply 11 =', json.dumps(r11)[:40], '| image 80 boxes', len(q80))


if __name__ == '__main__':
    main()
