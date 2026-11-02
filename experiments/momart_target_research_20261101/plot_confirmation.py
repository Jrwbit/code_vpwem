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
    result=json.loads(a.result.read_text())['results'];fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    colors=['#777777','#287bb5'];names=['Frozen previous pointer','Dev-selected spatial query']
    for j,(key,label) in enumerate(zip(['baseline','primary'],names)):
        r=result[key]['accepted'];x=np.arange(3)
        axes[0].bar(x+(j-.5)*.35,[r[k]*100 for k in ['precision','recall','coverage']],.35,label=label,color=colors[j])
        values=[result[key]['roles'][str(role)]['recall']*100 for role in range(5)]
        axes[1].plot(np.arange(5),values,'o-',label=label,color=colors[j])
        axes[2].bar(np.arange(3)+(j-.5)*.35,[r[k] for k in ['wrong_class','wrong_acquire','wrong_carry']],.35,color=colors[j])
    axes[0].set_xticks(range(3),['Precision','Recall','Coverage']);axes[0].set_ylim(0,105);axes[0].set_ylabel('%');axes[0].axhline(90,color='red',linestyle=':',linewidth=1)
    axes[1].set_xticks(range(5),['Acquire\nbowl','Door\nhandle','Rack','Drawer\nhandle','Carry\nbowl']);axes[1].set_ylim(0,105);axes[1].set_ylabel('Correct box recall (%)')
    axes[2].set_xticks(range(3),['Wrong\nclass','Wrong class\nacquiring','Wrong class\ncarrying']);axes[2].set_ylabel('Frames')
    axes[0].set_title('8 frozen episodes: 415 labelled frames');axes[1].set_title('Per-role coverage exposes rack failures');axes[2].set_title('Errors decreased; goal not reached')
    fig.legend([Patch(color=color) for color in colors],names,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
