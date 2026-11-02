"""Paired episode-cluster bootstrap; frames within a trajectory are not independent."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(a):
    source=json.loads(a.input.read_text())['results'];reference=a.reference;episodes=sorted(source[reference]['episodes'])
    assert episodes==sorted(source['primary']['episodes'])
    arrays={k:np.array([[source[reference if k=='baseline' else k]['episodes'][e][v] for v in ['correct','selected','visible','recall']] for e in episodes],float) for k in ['baseline','primary']}
    def measures(values):
        return np.r_[values[:,0].sum()/max(values[:,1].sum(),1),values[:,0].sum()/max(values[:,2].sum(),1),values[:,3].mean()]
    rng=np.random.default_rng(20261105);deltas=[]
    for _ in range(a.repeats):
        index=rng.integers(0,len(episodes),len(episodes));deltas.append(measures(arrays['primary'][index])-measures(arrays['baseline'][index]))
    delta=measures(arrays['primary'])-measures(arrays['baseline']);ci=np.quantile(deltas,[.025,.975],axis=0)
    result=dict(episodes=episodes,seed=20261105,repeats=a.repeats,unit='paired episode clusters; primary accepted vs '+reference+' accepted',
        differences={key:dict(point=float(delta[i]),percentile95=[float(ci[0,i]),float(ci[1,i])]) for i,key in enumerate(['precision','pooled_recall','episode_macro_recall'])},
        limitation=f'{len(episodes)} clusters only; uncertainty interval describes this proxy-labelled corpus, not policy success')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=10000);p.add_argument('--reference',choices=['baseline','control'],default='baseline');main(p.parse_args())
