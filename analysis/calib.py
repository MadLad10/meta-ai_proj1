"""Score-calibration control for the box-crop test (open-vocabulary detectors): does a per-class score offset explain the
default label? Each class's crop scores are standardized over all crops of the benchmark (z-score), then the class with
the highest standardized score is taken. Reports seafloor and other-class accuracy, raw vs calibrated, with bootstrap
intervals over crops (500 draws within class, seed 20261119). Writes analysis/calib.json."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; J = ROOT / 'gpu_job' / 'outputs' / 'job8'
SEA = ['holothurian', 'echinus', 'scallop', 'starfish']
NAME = dict(holothurian='sea cucumber', echinus='sea urchin')


def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def run(path):
    P = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cls = list(P[0]['out']); S = np.array([[p['out'][c] for c in cls] for p in P]); gt = np.array([p['gt'] for p in P])
    key = {NAME.get(c, c): c for c in set(gt)}; lab = np.array([key.get(c, c) for c in cls])     # score keys are class ids
    raw = lab[S.argmax(1)]; Z = (S - S.mean(0)) / (S.std(0) + 1e-9); cal = lab[Z.argmax(1)]
    classes = sorted(set(gt)); by = {c: np.where(gt == c)[0] for c in classes}
    rng = np.random.default_rng(20261119); W = [{c: rng.choice(by[c], len(by[c])) for c in classes} for _ in range(500)]
    out = {}
    for name, pred in (('raw', raw), ('calibrated', cal)):
        cor = pred == gt
        sea = [c for c in classes if c in SEA]; oth = [c for c in classes if c not in SEA]
        f = lambda idx_by, cs: np.mean([cor[idx_by[c]].mean() for c in cs]) if cs else float('nan')
        boot_sea = [f(w, sea) for w in W]; boot_oth = [f(w, oth) for w in W] if oth else None
        vals, cnt = np.unique(pred[np.isin(gt, SEA)], return_counts=True)
        out[name] = dict(seafloor=float(f(by, sea)), seafloor_ci=ci(boot_sea), other=float(f(by, oth)) if oth else None,
                         other_ci=ci(boot_oth) if oth else None, top_answer_seafloor=str(vals[cnt.argmax()]), top_share_seafloor=float(cnt.max() / cnt.sum()))
    return out


def main():
    res = {}
    for bench, job in (('duo', J), ('ruod', J / 'ruod')):
        for model in ('grounding-dino_common', 'owlv2_common'):
            res[f'{bench}:{model.split("_")[0]}'] = run(job / model / 'predictions.jsonl')
    (ROOT / 'analysis' / 'calib.json').write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res.items():
        r, c = v['raw'], v['calibrated']
        print(f"{k:22s} seafloor raw {r['seafloor']:.3f} -> calibrated {c['seafloor']:.3f} {[round(x, 3) for x in c['seafloor_ci']]}"
              + (f" | other raw {r['other']:.3f} -> {c['other']:.3f}" if r['other'] is not None else '') + f" | top answer {r['top_answer_seafloor']} {r['top_share_seafloor']:.0%} -> {c['top_answer_seafloor']} {c['top_share_seafloor']:.0%}")


if __name__ == '__main__':
    main()
