"""Box-crop naming test (Job 8): per-class accuracy with bootstrap intervals over crops (500 draws, seed 20261119,
resampled within each class), confusion counts, and paired differences. Writes analysis/crops.json."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; J = ROOT / 'gpu_job' / 'outputs' / 'job8'
CLASSES = ['holothurian', 'echinus', 'scallop', 'starfish']
RUNS = {'gdino': 'grounding-dino_common', 'gdino_latin': 'grounding-dino_latin', 'owlv2': 'owlv2_common', 'owlv2_latin': 'owlv2_latin',
        'qwen25_3b': 'qwen2.5-vl-3b_common', 'qwen25_7b': 'qwen2.5-vl-7b_common', 'qwen3_8b': 'qwen3-vl-8b_common'}


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    P = {k: {p['ann_id']: p for p in map(json.loads, (J / v / 'predictions.jsonl').read_text().splitlines())} for k, v in RUNS.items()}
    ids = sorted(next(iter(P.values())))
    assert all(sorted(p) == ids for p in P.values())
    gt = {i: P['gdino'][i]['gt'] for i in ids}
    by = {c: np.array([i for i in ids if gt[i] == c]) for c in CLASSES}
    rng = np.random.default_rng(20261119)
    draws = [{c: rng.choice(by[c], len(by[c])) for c in CLASSES} for _ in range(500)]   # resample crops within each class
    correct = {k: {i: P[k][i]['pred'] == gt[i] for i in ids} for k in P}
    out = dict(crops=len(ids), per_class_n={c: int(len(by[c])) for c in CLASSES}, models={}, differences={})
    boot = {}
    for k in P:
        pc = {c: float(np.mean([correct[k][i] for i in by[c]])) for c in CLASSES}
        b = np.array([[np.mean([correct[k][i] for i in d[c]]) for c in CLASSES] for d in draws])
        boot[k] = b
        conf = {c: {e: int(sum(P[k][i]['pred'] == e for i in by[c])) for e in CLASSES + [None]} for c in CLASSES}
        out['models'][k] = dict(mean_class_accuracy=float(np.mean(list(pc.values()))), mean_ci=ci(b.mean(1)),
                                per_class={c: dict(acc=pc[c], ci=ci(b[:, n])) for n, c in enumerate(CLASSES)},
                                confusion={c: {str(e): v for e, v in conf[c].items()} for c in CLASSES},
                                most_common_label=max(CLASSES, key=lambda e: sum(P[k][i]['pred'] == e for i in ids)),
                                share_most_common=float(max(sum(P[k][i]['pred'] == e for i in ids) for e in CLASSES) / len(ids)))
    for a, b_, lab in [('qwen3_8b', 'gdino', 'Qwen3-VL-8B minus Grounding DINO'), ('qwen3_8b', 'owlv2', 'Qwen3-VL-8B minus OWLv2'),
                       ('qwen3_8b', 'qwen25_7b', 'Qwen3-VL-8B minus Qwen2.5-VL-7B')]:
        out['differences'][lab] = dict(diff=out['models'][a]['mean_class_accuracy'] - out['models'][b_]['mean_class_accuracy'],
                                       ci=ci(boot[a].mean(1) - boot[b_].mean(1)))
    (Path(__file__).resolve().parent / 'crops.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, m in out['models'].items():
        print(f"{k:12s} mean {m['mean_class_accuracy']:.3f} {[round(x, 3) for x in m['mean_ci']]} | " +
              ' '.join(f"{c[:4]} {v['acc']:.2f}" for c, v in m['per_class'].items()) + f" | most common label {m['most_common_label']} ({m['share_most_common']:.0%})")
    for k, v in out['differences'].items():
        print(f"{k}: {v['diff']:+.3f} {[round(x, 3) for x in v['ci']]}")


if __name__ == '__main__':
    main()
