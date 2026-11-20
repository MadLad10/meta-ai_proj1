"""RUOD summary from analysis/cache/ruod_*.npz: AP with paired bootstrap intervals (500 draws over test images, seed
20261119), per-class AP, the four seafloor classes vs the other six, found/named shares, labels of found seafloor
objects, the matched box budget on the 1,000-image VLM subset, and fine-tuned AP without the test images that also
appear in the training set. Writes analysis/ruod.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_bootstrap import Matched, draws
from assess import ci

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / 'gpu_job' / 'outputs'; C = ROOT / 'analysis' / 'cache'
FT = ['yolov8n', 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'rtdetr-l']; OVD = ['gdino', 'owlv2']; VLM = ['qwen25_3b', 'qwen25_7b', 'qwen3_8b']


class Arr:                                                     # Matched-compatible view of a cached model
    def __init__(self, z):
        self.img_ids = list(z['img_ids']); self.cats = sorted({int(k[1:].split('_')[0]) for k in z.files if k.startswith('c') and '_' in k})
        self.per_cat = [dict(npig=z[f'c{c}_npig'], S=z[f'c{c}_S'], IMG=z[f'c{c}_IMG'], TP=z[f'c{c}_TP'], FP=z[f'c{c}_FP']) for c in self.cats]
    ap = Matched.ap


def main():
    g = json.loads((O / 'gt' / 'ruod_instances_test.json').read_text())
    names = {c['id']: c['name'] for c in g['categories']}; cats = sorted(names)
    SEA = [c for c in cats if names[c] in ('holothurian', 'echinus', 'scallop', 'starfish')]; OTH = [c for c in cats if c not in SEA]
    Z = {k: np.load(C / f'ruod_{k}.npz') for k in FT + OVD + VLM}; A = {k: Arr(z) for k, z in Z.items()}
    all_ids = A['yolo11n'].img_ids; sub = A['qwen3_8b'].img_ids; pos = {i: n for n, i in enumerate(all_ids)}
    sub_idx = np.array([pos[i] for i in sub]); Wf = draws(len(all_ids)); Ws = draws(len(sub))
    dup = set(json.loads((O / 'gt' / 'ruod_test_duplicates.json').read_text())['image_ids'])
    def w_sub(k, d=None):                                      # weights on k's own images for subset draw d (None: all ones)
        base = np.ones(len(sub)) if d is None else Ws[d]
        if k in VLM: return base
        w = np.zeros(len(all_ids)); w[sub_idx] = base; return w
    out = dict(classes={str(c): names[c] for c in cats}, seafloor=[names[c] for c in SEA], protocol='paired bootstrap over test images, 500 draws, seed 20261119')
    sea_mask = np.array([c in SEA for c in A['yolo11n'].cats])
    def summary(vals_point, vals_boot):
        return dict(ap=float(vals_point), ci=ci(vals_boot))
    out['models'] = {}; boot_full, boot_sub = {}, {}
    for k in FT + OVD + VLM:
        a = A[k]; full = k not in VLM
        pc_point = a.ap(None if full else None, per_class=True)
        bf = np.array([a.ap(w, per_class=True) for w in (Wf if full else [w_sub(k, d) for d in range(len(Ws))])])
        bs = np.array([a.ap(w_sub(k, d), per_class=True) for d in range(len(Ws))]) if full else bf
        pcs = np.array(pc_point); boot_full[k] = bf; boot_sub[k] = bs
        rec = dict(images='full' if full else 'subset', ap=summary(np.nanmean(pcs), np.nanmean(bf, 1)),
                   ap_subset=summary(np.nanmean(a.ap(w_sub(k), per_class=True)), np.nanmean(bs, 1)),
                   seafloor_ap=summary(np.nanmean(pcs[sea_mask]), np.nanmean(bf[:, sea_mask], 1)),
                   other_ap=summary(np.nanmean(pcs[~sea_mask]), np.nanmean(bf[:, ~sea_mask], 1)),
                   per_class={names[c]: summary(pcs[n], bf[:, n]) for n, c in enumerate(a.cats)})
        if k in FT:
            wd = np.array([0.0 if i in dup else 1.0 for i in all_ids])
            rec['ap_dedup'] = summary(a.ap(wd), [a.ap(w * wd) for w in Wf])
        F = Z[k]['fn_top100_iou50']                           # images x classes x [objects, found, named]
        Wfn = Wf if full else Ws
        def share(sel, col):
            tot = F[:, sel, :].sum(1); bs_ = np.array([w @ tot for w in Wfn])
            return dict(share=float(tot[:, col].sum() / tot[:, 0].sum()), ci=ci(bs_[:, col] / bs_[:, 0]))
        seai = [cats.index(c) for c in SEA]; othi = [cats.index(c) for c in OTH]
        rec['found_named'] = dict(seafloor=dict(found=share(seai, 1), named=share(seai, 2)), other=dict(found=share(othi, 1), named=share(othi, 2)),
                                  all=dict(found=share(list(range(len(cats))), 1), named=share(list(range(len(cats))), 2)),
                                  per_class={names[c]: dict(objects=int(F[:, n, 0].sum()), found=float(F[:, n, 1].sum() / max(F[:, n, 0].sum(), 1)),
                                                            named=float(F[:, n, 2].sum() / max(F[:, n, 0].sum(), 1))) for n, c in enumerate(cats)})
        L = Z[k]['labels']; found_sea = L[seai].sum(1); lab_tot = L[seai].sum(0)
        top = int(np.argmax(lab_tot)); rec['found_seafloor_labels'] = dict(most_common=names[cats[top]], share=float(lab_tot[top] / max(found_sea.sum(), 1)),
                                                                            by_label={names[cats[e]]: int(v) for e, v in enumerate(lab_tot)})
        if 'budget' in Z[k].files:
            B = Z[k]['budget'].astype(float); bb = np.array([w @ B for w in Ws])
            P = lambda t: t[..., 0] / np.maximum(t[..., 0] + t[..., 1], 1); R = lambda t: t[..., 0] / np.maximum(t[..., 2], 1)
            rec['budget'] = dict(precision=float(P(B.sum(0))), recall=float(R(B.sum(0))), precision_ci=ci(P(bb)), recall_ci=ci(R(bb)),
                                 boxes_per_image=float(B[:, :2].sum() / len(sub)), _bs=bb)
        out['models'][k] = rec
    # differences
    def d_full(a, b): return dict(diff=out['models'][a]['ap']['ap'] - out['models'][b]['ap']['ap'], ci=ci(np.nanmean(boot_full[a], 1) - np.nanmean(boot_full[b], 1)))
    def d_sub(a, b): return dict(diff=out['models'][a]['ap_subset']['ap'] - out['models'][b]['ap_subset']['ap'], ci=ci(np.nanmean(boot_sub[a], 1) - np.nanmean(boot_sub[b], 1)))
    weakest = min(FT, key=lambda k: out['models'][k]['ap']['ap']); best = max(OVD, key=lambda k: out['models'][k]['ap']['ap'])
    out['differences'] = {'weakest fine-tuned minus best zero-shot detector': dict(models=[weakest, best], **d_full(weakest, best)),
                          'OWLv2 minus Grounding DINO': d_full('owlv2', 'gdino'),
                          'Grounding DINO minus Qwen3-VL-8B (subset)': d_sub('gdino', 'qwen3_8b'), 'OWLv2 minus Qwen3-VL-8B (subset)': d_sub('owlv2', 'qwen3_8b'),
                          'Qwen3-VL-8B minus Qwen2.5-VL-7B (subset)': d_sub('qwen3_8b', 'qwen25_7b')}
    # seafloor drop for each model: other minus seafloor AP
    out['seafloor_gap'] = {k: dict(diff=out['models'][k]['other_ap']['ap'] - out['models'][k]['seafloor_ap']['ap'],
                                   ci=ci(np.nanmean(boot_full[k][:, ~sea_mask], 1) - np.nanmean(boot_full[k][:, sea_mask], 1))) for k in FT + OVD + VLM}
    q = out['models']['qwen3_8b']['budget']
    for k in ['gdino', 'owlv2', 'yolo11n']:
        b = out['models'][k]['budget']
        for m, f in (('precision', lambda t: t[:, 0] / np.maximum(t[:, 0] + t[:, 1], 1)), ('recall', lambda t: t[:, 0] / np.maximum(t[:, 2], 1))):
            out['differences'][f'budget Qwen3-VL-8B minus {k} {m}'] = dict(diff=q[m] - b[m], ci=ci(f(q['_bs']) - f(b['_bs'])))
    for k in out['models'].values():
        k.get('budget', {}).pop('_bs', None)
    (ROOT / 'analysis' / 'ruod.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, r in out['models'].items():
        fn = r['found_named']
        print(f"{k:10s} AP {r['ap']['ap']:.3f} {[round(x, 3) for x in r['ap']['ci']]} sub {r['ap_subset']['ap']:.3f} | sea {r['seafloor_ap']['ap']:.3f} other {r['other_ap']['ap']:.3f}"
              f" | found/named sea {fn['seafloor']['found']['share']:.2f}/{fn['seafloor']['named']['share']:.2f} other {fn['other']['found']['share']:.2f}/{fn['other']['named']['share']:.2f}"
              f" | sea labels -> {r['found_seafloor_labels']['most_common']} {r['found_seafloor_labels']['share']:.0%}" + (f" | dedup {r['ap_dedup']['ap']:.3f}" if 'ap_dedup' in r else '')
              + (f" | budget P {r['budget']['precision']:.3f} R {r['budget']['recall']:.3f}" if 'budget' in r else ''))
    for k, v in out['differences'].items():
        print(f"  {k}: {v['diff']:+.3f} {[round(x, 3) for x in v['ci']]}")


if __name__ == '__main__':
    main()
