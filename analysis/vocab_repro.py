"""Reproduction check for the CPU vocabulary test: vocab_cpu.py with the original four class names (first 20 DUO crops)
must give the GPU box-crop test's answers and scores (gpu_job/outputs/job8). Prints the number of crops, the share with
the same answer and the largest absolute score difference. Writes analysis/vocab_repro.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; J = ROOT / 'gpu_job' / 'outputs' / 'job8'
ID = {'sea cucumber': 'holothurian', 'sea urchin': 'echinus', 'scallop': 'scallop', 'starfish': 'starfish'}
res = {}
for model, run in (('gdino', 'grounding-dino_common'), ('owlv2', 'owlv2_common')):
    gpu = {json.loads(l)['ann_id']: json.loads(l) for l in (J / run / 'predictions.jsonl').read_text().splitlines()}
    cpu = [json.loads(l) for l in (ROOT / 'analysis' / 'vocab_cpu' / f'{model}_duo4_limit20.jsonl').read_text().splitlines() if l.strip()]
    diff = max(abs(r['scores'][k] - gpu[r['ann_id']]['out'][ID[k]]) for r in cpu for k in r['scores'])
    same = sum(ID[r['pred']] == gpu[r['ann_id']]['pred'] for r in cpu)
    res[model] = dict(crops=len(cpu), same_answer=same, max_abs_score_diff=diff)
    print(model, res[model])
(ROOT / 'analysis' / 'vocab_repro.json').write_text(json.dumps(res, indent=1) + '\n')
