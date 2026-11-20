"""Figure: box-crop naming accuracy per class, both benchmarks (from analysis/crops.json). Output: paper/fig_heatmap.pdf."""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / 'paper'
C = json.loads((ROOT / 'analysis' / 'crops.json').read_text())
MODELS = [('Grounding DINO', 'gdino'), ('OWLv2', 'owlv2'), ('Qwen2.5-VL-3B', 'qwen25_3b'), ('Qwen2.5-VL-7B', 'qwen25_7b'), ('Qwen3-VL-8B', 'qwen3_8b')]
SEA = [('holothurian', 'sea cuc.'), ('echinus', 'urchin'), ('scallop', 'scallop'), ('starfish', 'starfish')]
OTHER = [('fish', 'fish'), ('corals', 'coral'), ('diver', 'diver'), ('cuttlefish', 'cuttlef.'), ('turtle', 'turtle'), ('jellyfish', 'jellyf.')]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    cols = [('DUO', c, n) for c, n in SEA] + [('RUOD', c, n) for c, n in SEA + OTHER]
    M = np.array([[(C['models'] if b == 'DUO' else C['ruod']['models'])[k]['per_class'][c]['acc'] for b, c, _ in cols] for _, k in MODELS])
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    xs = [i + (0.5 if i >= 4 else 0) for i in range(len(cols))]                         # gap between the benchmarks
    for r in range(M.shape[0]):
        for j, x in enumerate(xs):
            v = M[r, j]
            ax.add_patch(plt.Rectangle((x - 0.48, r - 0.46), 0.96, 0.92, color=plt.cm.Blues(0.08 + 0.85 * v), lw=0))
            ax.text(x, r, f'{100 * v:.0f}', ha='center', va='center', fontsize=6.3, color='white' if v > 0.55 else '#1a1a1a')
    D = C['default_labels']                                                          # outline each model's default answer
    for r, (_, k) in enumerate(MODELS):
        for b in ('duo', 'ruod'):
            lab = D[f'{b}:{k}']['label']
            for j, (bb, c, _) in enumerate(cols):
                if bb.lower() == b and c == lab:
                    ax.add_patch(plt.Rectangle((xs[j] - 0.46, r - 0.42), 0.92, 0.84, fill=False, ec='#D55E00', lw=1.3))
    ax.set_xlim(-0.6, xs[-1] + 0.6); ax.set_ylim(len(MODELS) - 0.5, -1.35)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m for m, _ in MODELS])
    ax.set_xticks(xs); ax.set_xticklabels([n for _, _, n in cols], rotation=0)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(np.mean(xs[:4]), -1.05, 'DUO (chance 25%)', ha='center', fontsize=6.8, weight='bold')
    ax.text(np.mean(xs[4:8]), -1.05, 'RUOD: seafloor classes', ha='center', fontsize=6.8, weight='bold')
    ax.text(np.mean(xs[8:]), -1.05, 'RUOD: other classes (chance 10%)', ha='center', fontsize=6.8, weight='bold')
    for x in (xs[4] - 0.5 - 0.25, xs[8] - 0.5):
        ax.plot([x, x], [-0.5, len(MODELS) - 0.5], color='#555555', lw=0.8, ls=(0, (2, 1.5)) if x > xs[5] else '-')
    fig.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.12)
    raw = P / 'fig_heatmap.raw.pdf'; fig.savefig(raw); plt.close(fig)
    subprocess.run(['gs', '-q', '-o', str(P / 'fig_heatmap.pdf'), '-dNoOutputFonts', '-sDEVICE=pdfwrite', str(raw)], check=True)
    raw.unlink(); print('written')


if __name__ == '__main__':
    main()
