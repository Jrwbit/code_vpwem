"""Standalone evidence figure, generated only from saved confirmation metrics."""
import argparse
import json
from pathlib import Path
from matplotlib.patches import Patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    result=json.loads(a.result.read_text())['results'];has_sequence='sequence' in result['primary']
    fig,axes=plt.subplots(2,2,figsize=(12,8.5)) if has_sequence else plt.subplots(1,3,figsize=(14,4.2))
    axes=np.asarray(axes).reshape(-1)
    colors=['#777777','#287bb5'];names=['Frozen previous pointer','Dense mask + spatial ExtraTrees' if result['primary']['name'].startswith('mask_') else result['primary']['name']]
    for j,(key,label) in enumerate(zip(['baseline','primary'],names)):
        r=result[key]['accepted'];x=np.arange(3)
        axes[0].bar(x+(j-.5)*.35,[r[k]*100 for k in ['precision','recall','coverage']],.35,label=label,color=colors[j])
        values=[result[key]['roles'][str(role)]['recall']*100 for role in range(5)]
        axes[1].plot(np.arange(5),values,'o-',label=label,color=colors[j])
        axes[2].bar(np.arange(3)+(j-.5)*.35,[r[k] for k in ['wrong_class','wrong_acquire','wrong_carry']],.35,color=colors[j])
    axes[0].set_xticks(range(3),['Precision','Recall','Coverage']);axes[0].set_ylim(0,105);axes[0].set_ylabel('%');axes[0].axhline(90,color='red',linestyle=':',linewidth=1)
    axes[1].set_xticks(range(5),['Acquire\nbowl','Door\nhandle','Rack','Drawer\nhandle','Carry\nbowl']);axes[1].set_ylim(0,105);axes[1].set_ylabel('Correct box recall (%)')
    axes[2].set_xticks(range(3),['Wrong\nclass','Wrong class\nacquiring','Wrong class\ncarrying']);axes[2].set_ylabel('Frames')
    axes[0].set_title(f"{len(result['primary']['episodes'])} episodes: {result['primary']['accepted']['samples']} labelled frames");axes[1].set_title('Per-role correct-box recall');axes[2].set_title('Wrong-class errors by phase')
    if has_sequence:
        for j,key in enumerate(['baseline','primary']):
            seq=result[key]['sequence'];axes[3].bar(np.arange(4)+(j-.5)*.35,[seq[k] for k in ['stable_class_switches','stable_correct_to_null','stale_previous_class_at_change','correct_box_at_change']],.35,color=colors[j])
        axes[3].set_xticks(range(4),['Class chatter','Correct to null','Stale after\nchange','Correct at\nchange']);axes[3].set_ylabel('Adjacent frame pairs');axes[3].set_title('Subject-switch proxy diagnostics')
    fig.legend([Patch(color=color) for color in colors],names,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
