"""Freeze the growth winner and original control before confirmation preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_retention_reference(value,registered,retention):
    groups=value.get('current_retention',{})
    if not {'dev','old_dev','new_dev'}<=groups.keys() or not all(v['joint_goal_met'] for v in groups.values()):
        raise ValueError('Current-candidate retention did not pass all registered development groups')
    keys=['threshold','margin','class_thresholds','class_margins']
    cal=lambda x:{k:x[k] for k in keys if k in x}
    if registered['component']!=retention['result']['component'] or cal(registered['calibration'])!=cal(retention['result']['calibration']):
        raise ValueError('Retention comparator differs from the complete development configuration')


def component_weights(component):
    if component['kind']=='parent':return [Path(component['screen'])/(component['name']+'.pt')]
    if component['kind']=='availability':return component_weights(component['parent'])
    if component['kind']=='blend':return [p for member in component['members'] for p in component_weights(member)]
    raise ValueError('Unsupported weight freeze component: '+component['kind'])


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
    payload=dict(selected,holdout=a.holdout.name.removesuffix('_corpus'),holdout_evaluated=False,holdout_original_manifest_sha256=digest(a.holdout/'manifest.json'),
        source_selection_sha256=digest(a.selection),training_audit_sha256=digest(a.training_audit),
        control_selection_sha256=digest(a.control),control_name=control['name'],control_calibration=control['result']['calibration'],
        control_policy='Original frozen_for_confirm4 selection, not the growth-dev recalibrated control',
        purpose='Prospective selector confirmation; source-nonvalid episodes may have been seen by upstream frozen visual/action models')
    if selected.get('current_retention_required'):
        if a.retention is None:raise ValueError('Registered current retention requires the exact comparator selection')
        retention=json.loads(a.retention.read_text());registry=json.loads((a.selection.parent/'registry.json').read_text())
        validate_retention_reference(value,registry['blend_frozen_configuration'],retention)
        payload['retention_reference']=dict(name=retention['name'],component=retention['result']['component'],calibration=retention['result']['calibration'])
        payload['retention_selection_sha256']=digest(a.retention)
        paths=set(component_weights(value['component'])+component_weights(control['result']['component'])+component_weights(retention['result']['component']))
        payload['frozen_weight_sha256']={str(path):digest(path) for path in sorted(paths)}
        import torch
        from history_contract import public_history_contract
        histories={}
        for path in sorted(paths):
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint.get('spec',{}).get('kind')!='grid_gru':continue
            history=Path(checkpoint['spec']['main_history_path']);metadata=json.loads(history.with_suffix('.json').read_text())
            histories[str(history)]=dict(archive_sha256=digest(history),metadata_sha256=digest(history.with_suffix('.json')),contract=public_history_contract(metadata))
        if histories:
            contract=next(iter(histories.values()))['contract']
            if not all(value['contract']==contract for value in histories.values()):raise ValueError('Recurrent members use different public sensor contracts')
            payload['frozen_main_history_contract']=contract;payload['frozen_training_histories']=histories
    a.output.mkdir(parents=True);(a.output/'selected.json').write_text(json.dumps(payload,indent=2))
    (a.output/'holdout_manifest.json').write_bytes((a.holdout/'manifest.json').read_bytes())
    print(json.dumps(dict(winner=payload['name'],reserved_episodes=len(reserved),source_partitions=sorted({ep['source_partition'] for ep in corpus['episodes']}),output=str(a.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['selection','control','training-audit','holdout','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--retention',type=Path);main(p.parse_args())
