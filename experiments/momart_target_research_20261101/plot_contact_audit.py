"""Contact/holding evidence figure from saved audit records, without relabelling."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    audit=json.loads(a.audit.read_text());records=audit['records'];x=np.array([r['step'] for r in records])
    phase={r['step']:r for r in json.loads(a.phases.read_text())['records'] if r['episode_id']==a.episode}
    held=np.array([phase[t]['held_now'] if t in phase else False for t in x])
    rack_key=next(key for key in records[0]['rack_joints'] if key.startswith(str(audit['rack_links'][0])+':'))
    rack=[r['rack_joints'][rack_key]['position'] for r in records];height=[r['bowl_relative_rack'][2] for r in records]
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax in axes:ax.fill_between(x,0,1,where=held,transform=ax.get_xaxis_transform(),step='mid',alpha=.13,color='orange',label='Weak proxy: held=True')
    left=axes[0].plot(x,rack,'o-',color='#287bb5',label='Rack extension (m)');axes[0].set_ylabel('Rack joint (m)');axes[0].set_ylim(0,records[0]['rack_joints'][rack_key]['upper']*1.08)
    twin=axes[0].twinx();right=twin.plot(x,height,'s-',color='#56983c',label='Bowl height in rack frame (m)');twin.set_ylabel('Bowl relative height (m)')
    axes[0].legend(left+right,[h.get_label() for h in left+right],loc='upper left',fontsize=8)
    for name,label,color in [('r_gripper_finger_link','Right finger to bowl','#c13c35'),('l_gripper_finger_link','Left finger to bowl','#287bb5')]:
        axes[1].plot(x,[1000*r['hand_distances'][name]['bowl_distance'] for r in records],'o-',label=label,color=color)
    axes[1].axhline(5,color='black',linestyle=':',label='5 mm geometry tolerance');axes[1].set_ylim(-3,85);axes[1].set_ylabel('Closest distance (mm)');axes[1].legend(fontsize=8)
    supports=[sum(p['distance']<=.002 and p['normal'][2]>=.3 for p in r['bowl_rack_contacts']) for r in records]
    axes[2].step(x,supports,where='mid',color='#287bb5',label='Upward bowl/rack contacts');axes[2].set_ylabel('Geometric contact points');axes[2].legend(fontsize=8);axes[2].set_xlabel('Recorded sensor step (20 Hz)')
    for ax in axes:
        ax.axvline(464,color='#7e5ba6',linestyle='--',alpha=.7);ax.axvline(560,color='#56983c',linestyle='--',alpha=.7)
    axes[0].set_title('Rack/bowl coupling can satisfy a weak holding rule\nOrange: old held label; purple line: acquisition proxy starts; green: bilateral bowl contact',fontsize=11)
    fig.suptitle(a.episode,fontsize=12);fig.tight_layout(rect=[0,.025,1,.96]);fig.text(.5,.005,'State-restored geometry only: no force-history reconstruction or human-intent ground truth.',ha='center',fontsize=8)
    fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['audit','phases','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episode',required=True);main(p.parse_args())
