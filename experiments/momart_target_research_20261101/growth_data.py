"""Combine enlarged train/dev while preserving every earlier episode assignment."""
import numpy as np
from data import load_group,attach_geometry,combine,partitions
from expanded import load_expanded_data


COMMON=['rows','tasks','candidates','context','valid','y','roles','positive','visible','weights','geometry_candidates','geometry_context']


def load_growth_data(root):
    research=root/'momart_target_research_20261101';old=load_expanded_data(root)
    new=attach_geometry(load_group(research/'growth_corpus',research/'growth_features.npz',research/'growth_phases.json','old'),research/'geometry_growth.npz')
    new['rows']=[dict(row,cohort='growth') for row in new['rows']]
    old_episodes={row['episode_id'] for row in old['rows']};new_episodes={row['episode_id'] for row in new['rows']}
    if old_episodes&new_episodes:raise ValueError('Growth episodes overlap examined training/development/regression')
    import json
    holdout=json.loads((research/'confirm5_corpus/manifest.json').read_text())
    if (old_episodes|new_episodes)&{ep['episode_id'] for ep in holdout['episodes']}:raise ValueError('Growth overlaps reserved confirmation5')
    return combine([{key:group[key] for key in COMMON} for group in [old,new]])


def growth_partitions(data):
    split=partitions(data)
    is_new=np.array([row.get('cohort')=='growth' for row in data['rows']])
    split['old_train']=split['train'][~is_new[split['train']]]
    split['old_dev']=split['dev'][~is_new[split['dev']]]
    split['new_dev']=split['dev'][is_new[split['dev']]]
    return split
