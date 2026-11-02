"""Freeze the growth winner and original control before confirmation preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    selected=json.loads(a.selection.read_text());control=json.loads(a.control.read_text());audit=json.loads(a.training_audit.read_text())
    value=selected['result']
    if not value['joint_growth_goal_met'] or not all(g['joint_goal_met'] for g in value['subgroup_gates'].values()):
        raise ValueError('Growth winner did not meet both registered development subgroup gates')
    corpus=json.loads((a.holdout/'manifest.json').read_text())
    reserved={ep['episode_id'] for ep in corpus['episodes']}
    used={ep for episodes in audit['episodes'].values() for ep in episodes}
    if used&reserved:raise ValueError('Prospective confirmation overlaps examined model data')
    for forbidden in ['annotations_all.json','candidate_manifest.json']:
        if (a.holdout/forbidden).exists():raise ValueError('Expected untouched, unprocessed confirmation corpus')
    payload=dict(selected,holdout='confirm5',holdout_evaluated=False,holdout_original_manifest_sha256=digest(a.holdout/'manifest.json'),
        source_selection_sha256=digest(a.selection),training_audit_sha256=digest(a.training_audit),
        control_selection_sha256=digest(a.control),control_name=control['name'],control_calibration=control['result']['calibration'],
        control_policy='Original frozen_for_confirm4 selection, not the growth-dev recalibrated control',
        purpose='Prospective selector confirmation; source-nonvalid episodes may have been seen by upstream frozen visual/action models')
    a.output.mkdir(parents=True);(a.output/'selected.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(winner=payload['name'],reserved_episodes=len(reserved),source_partitions=sorted({ep['source_partition'] for ep in corpus['episodes']}),output=str(a.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['selection','control','training-audit','holdout','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
