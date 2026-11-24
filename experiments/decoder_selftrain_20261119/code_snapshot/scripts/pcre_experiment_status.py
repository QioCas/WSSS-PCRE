"""Inspect a PCRE queue without changing any files."""
import argparse
import json
import os
from pathlib import Path


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # The running trainer may still be writing its last JSONL line.
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_dir', nargs='?')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.job_dir:
        directory = Path(args.job_dir)
    else:
        jobs = sorted((root / 'experiments').glob('pcre_sat_*'))
        if not jobs:
            parser.error('No experiment queues found')
        directory = jobs[-1]
    result = {'job_dir': str(directory.resolve())}
    events = records(directory / 'queue_events.jsonl')
    result['latest_queue_event'] = events[-1] if events else None
    if (directory / 'supervisor.pid').exists():
        pid = int((directory / 'supervisor.pid').read_text())
        result['supervisor_pid'] = pid
        try:
            os.kill(pid, 0)
            result['supervisor_alive'] = True
        except ProcessLookupError:
            result['supervisor_alive'] = False
    result['runs'] = {}
    for mode in ('smoke_baseline', 'smoke_adaptive', 'adaptive', 'baseline', 'adaptive_no_warmup'):
        run = directory / mode
        if not run.exists():
            continue
        training = records(run / 'training_metrics.jsonl')
        validation = records(run / 'validation_metrics.jsonl')
        checkpoints = records(run / 'checkpoints.jsonl')
        scored = [row for row in validation if row['segmentation']['miou'] is not None]
        best = max(scored, key=lambda row: row['segmentation']['miou']) if scored else None
        result['runs'][mode] = {
            'latest_training': training[-1] if training else None,
            'latest_validation': validation[-1] if validation else None,
            'best_segmentation_miou_percent': best['segmentation']['miou'] * 100 if best else None,
            'best_model_path': best['model_path'] if best else None,
            'latest_checkpoint': checkpoints[-1] if checkpoints else None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
