"""Status of the GPU jobs (Job 4 DUO fine-tuning, Job 5 zero-shot, Job 7 COCO scoring, Job 6 RUOD). Run from this folder."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
print('JOB 4: DUO (real underwater benchmark), six architectures')
prep = OUT / 'job4' / 'prepare.json'
if prep.exists():
    p = json.loads(prep.read_text()); print(f"  prepared: images {p['images']}, licence note: {p['licence_note'][:60]}")
else:
    print('  not prepared')
for name in ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolov8n', 'rtdetr-l'):
    d = OUT / 'job4' / name
    if (d / 'metrics.json').exists():
        m = json.loads((d / 'metrics.json').read_text())
        print(f"  {name:10s} test mAP50-95 {m['mAP50_95']:.3f}  mAP50 {m['mAP50']:.3f}  recall {m['recall_all']:.3f}  "
              f"small-object recall {m['by_size']['small']['recall']:.3f}")
    elif (d / 'record.json').exists():
        print(f'  {name:10s} trained, not evaluated')
    else:
        print(f'  {name:10s} not trained')

print('JOB 5: zero-shot detectors and VLMs (COCO AP, TIDE)')
for d in sorted((OUT / 'job5').glob('*/*/metrics.json')) if (OUT / 'job5').exists() else []:
    m = json.loads(d.read_text()); t = m.get('tide_main_errors') or {}
    print(f"  {d.parent.parent.name}/{d.parent.name:34s} AP {m.get('AP', 0):.3f}  AP50 {m.get('AP50', 0):.3f}  AP_small {m.get('AP_small', 0):.3f}  "
          f"detections {m.get('detections', 0)}  unmatched {m.get('unmatched_labels', 0)}  TIDE miss {t.get('Miss', float('nan')):.1f} cls {t.get('Cls', float('nan')):.1f} loc {t.get('Loc', float('nan')):.1f}")

print('JOB 7: fine-tuned detectors, COCO protocol')
for d in sorted((OUT / 'job7').glob('*/*/metrics.json')) if (OUT / 'job7').exists() else []:
    m = json.loads(d.read_text())
    print(f"  {d.parent.parent.name}/{d.parent.name:12s} AP {m.get('AP', 0):.3f}  AP50 {m.get('AP50', 0):.3f}  AP_small {m.get('AP_small', 0):.3f}")

print('JOB 6: RUOD (optional), six architectures')
for name in ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolov8n', 'rtdetr-l'):
    d = OUT / 'job6' / name
    if (d / 'metrics.json').exists():
        m = json.loads((d / 'metrics.json').read_text())
        print(f"  {name:10s} test mAP50-95 {m['mAP50_95']:.3f}  mAP50 {m['mAP50']:.3f}")
    elif (d / 'record.json').exists():
        print(f'  {name:10s} trained, not evaluated')
