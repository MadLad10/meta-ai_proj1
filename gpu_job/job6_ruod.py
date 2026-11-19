"""Job 6 (track B, optional): the Job 4 procedure on RUOD, a second established underwater detection benchmark.

RUOD (Fu et al., "Rethinking General Underwater Object Detection", Neurocomputing 2023): 14,000 images, 10 classes.
Download it into gpu_job/ruod_raw/ following EXPERIMENT_PLAN.md (section D-B3). This script reuses job4_duo.py
unchanged except for the folders, the expected split and the class list, which is read from the annotation file.

    python job6_ruod.py check | prepare | train [--smoke] | evaluate [--smoke]
"""
import importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('job4', HERE / 'job4_duo.py'); job4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(job4)
job4.RAW = HERE / 'ruod_raw'
job4.DATA = HERE / 'ruod_yolo'
job4.OUT = HERE / 'outputs' / 'job6'
job4.EXPECTED = {'train': 9800, 'test': 4200}     # RUOD paper split; if check reports other counts, stop and tell the user


def classes_from_annotations():
    files = job4.coco_files()
    if 'train' not in files:
        return None
    return [c['name'].strip().lower() for c in sorted(files['train'][1]['categories'], key=lambda c: c['id'])]


if __name__ == '__main__':
    if job4.RAW.exists():
        names = classes_from_annotations()
        if names:
            job4.CLASSES = names
            print('RUOD classes from the annotation file:', names)
    sys.argv[0] = 'job6_ruod.py'
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['check', 'prepare', 'train', 'evaluate'])
    ap.add_argument('--only', nargs='*', choices=list(job4.MODELS)); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'check': job4.check, 'prepare': job4.prepare, 'train': lambda: job4.train(a.only, a.smoke),
     'evaluate': lambda: job4.evaluate(a.smoke)}[a.command]()
