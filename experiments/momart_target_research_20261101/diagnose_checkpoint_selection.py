"""Exploratory same-dev epoch history audit, never an alternative model score.

Only the maximum-F1 state was saved. Other epochs' logs cannot establish
calibrated recall, subgroup/sequence gates, or reconstruct their missing weights.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    records={}
    for path in sorted(a.study.glob('*_training.json')):
        history=json.loads(path.read_text())['history'];best=max(history,key=lambda row:row['dev']['f1']);b=best['dev']
        candidates=[row for row in history if row['dev']['correct']>=b['correct'] and row['dev']['wrong_acquire']<=a.max_acquire_errors and row['dev']['wrong_carry']<=a.max_carry_errors]
        candidate=max(candidates,key=lambda row:(row['dev']['correct'],row['dev']['f1'])) if candidates else None
        def brief(row):
            return dict(epoch=row['epoch'],**{key:row['dev'][key] for key in ['correct','selected','wrong_acquire','wrong_carry','f1']}) if row else None
        records[path.stem]=dict(best=brief(best),candidate=brief(candidate),candidate_count=len(candidates),
            different_epoch_candidate=bool(candidate and candidate['epoch']!=best['epoch']),history_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report=dict(protocol='exploratory_checkpoint_priority_audit_v1',criteria=dict(correct='at least same model maximum-F1 raw correct count',max_acquire_errors=a.max_acquire_errors,max_carry_errors=a.max_carry_errors),
        models=len(records),different_epoch_candidates=sum(row['different_epoch_candidate'] for row in records.values()),records=records,
        caveat='Same-development posthoc diagnostic only. Raw probabilities before availability offset/rejection. Different epoch weights unavailable; no subgroup/role/sequence metrics or new model result. Do not substitute this diagnostic for unchanged full gates or fresh confirmation.')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(models=report['models'],different_epoch_candidates=report['different_epoch_candidates']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-acquire-errors',type=int,default=2);p.add_argument('--max-carry-errors',type=int,default=0);main(p.parse_args())
