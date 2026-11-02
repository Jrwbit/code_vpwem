"""Export a relocatable public-feature selector, not a perception/action policy.

Only the neural public base/geometry/spatial family used by current controls and
blends is supported. No dataset, annotations, evaluation rows or cached features
are copied. The bundle remains explicitly experimental regardless of scores.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export_bundle(selection,root,output):
    if output.exists():raise FileExistsError(output)
    selected=json.loads(selection.read_text());tasks=json.loads((root/'momart_target_selector_20261101/features.json').read_text())['tasks']
    artifacts={}
    def relocate(component):
        c=copy.deepcopy(component);kind=c['kind']
        if kind=='blend':c['members']=[relocate(v) for v in c['members']]
        elif kind=='availability':
            if c['model'] is not None:raise ValueError('Learned availability models are outside this bundle contract')
            c['parent']=relocate(c['parent'])
        elif kind=='parent':
            if c['name']=='previous_pointer':raise ValueError('Legacy previous_pointer is outside this neural bundle contract')
            path=Path(c['screen'])/(c['name']+'.pt')
            if not path.is_absolute():path=root/path
            ckpt=torch.load(path,map_location='cpu',weights_only=False);spec=ckpt['spec']
            if 'model' in ckpt or spec['kind'] not in ['linear','mlp','query','gru'] or spec['view'] not in ['base','geometry','spatial']:
                raise ValueError('Unsupported model/input family')
            if any(spec.get(k) for k in ['policy_features_path','candidate_history','history_signals']):
                raise ValueError('Additional public feature channels require their own explicit bundle contract')
            digest=sha(path);artifacts[digest]=path;c['name']=digest;c['screen']='weights'
        else:raise ValueError('Unsupported bundle component: '+kind)
        return c
    component=relocate(selected['result']['component'])
    # Validate the entire tree before creating output or copying artifacts.
    output.mkdir(parents=True);(output/'weights').mkdir();api=output/'api';api.mkdir()
    for digest,path in artifacts.items():shutil.copyfile(path,output/'weights'/(digest+'.pt'))
    for name in ['runtime.py','models.py','availability_features.py']:
        shutil.copyfile(Path(__file__).parent/name,api/name)
    metadata=output/'momart_target_selector_20261101';metadata.mkdir()
    (metadata/'features.json').write_text(json.dumps(dict(tasks=tasks),indent=2))
    calibration=selected['result']['calibration']
    (output/'selected.json').write_text(json.dumps(dict(name=selected['name'],status='research_candidate_not_promoted',
        result=dict(component=component,calibration={k:calibration.get(k,0.) for k in ['threshold','margin']})),indent=2))
    (output/'README.md').write_text('''# Public-feature single-box selector (research candidate)

This package is not a detector, image encoder, action policy or validated robot
deployment. It selects one normalized xyxy box or valid=false from public
features prepared by the documented MoMaRT frontend. No labels/scene GT/future
actions may enter these inputs. Do not infer closed-loop success from this bundle.

Requires Python, NumPy and PyTorch; source package supplies runtime/models and
availability factorization. Add BUNDLE/api to PYTHONPATH, then:

    from runtime import FrameFeatures, SingleBoxSelector
    selector = SingleBoxSelector(BUNDLE / "selected.json", BUNDLE)
    result = selector.select(frame, episode_id, sensor_step)

`frame` is FrameFeatures: candidates[8,139], context[280], valid[8], task from
features.json; geometry models also need geometry_candidates[8,16] and
geometry_context[14]. Main visual/context features must use the original frozen
encoder/normalization. Context includes current and t-8 observations, not future
data. Temporal state resets by episode; repeat steps are idempotent; steps must
increase. The task vocabulary and feature schema remain the validated four-task
contract. Whole-bundle paths are relative and the directory may be moved.

manifest.json records hashes of all copied code, weights and configuration.
''')
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    manifest=dict(protocol='public_feature_selector_bundle_v1',status='research_candidate_not_promoted',
        original_selection_sha256=sha(selection),tasks=tasks,files=files,
        scope='Selector and public feature API only; no perception/action policy or datasets included')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['selection','root','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();m=export_bundle(a.selection,a.root,a.output);print(json.dumps(dict(files=len(m['files']),output=str(a.output),status=m['status'])))
