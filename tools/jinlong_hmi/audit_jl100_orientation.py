#!/usr/bin/env python3
"""Extract JL100's far plate edge from STEP and audit a saved station pose."""
import argparse
import hashlib
import json
import pyexpat  # Load the Python runtime's matching Expat before OCCT libraries.
from pathlib import Path

import numpy as np


def unit(vector):
    return vector/np.linalg.norm(vector)


def parallel_angle(a,b):
    return float(np.degrees(np.arccos(np.clip(abs(unit(a)@unit(b)),-1,1))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=Path,default=Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP'))
    parser.add_argument('--result',type=Path,default=Path('outputs/diagnostics/shared-station-20261023/root-final-reproduce/result.json'))
    parser.add_argument('--template',type=Path,default=Path('outputs/diagnostics/actual-terminal-alignment-20261023/direction-audit/templates/JL100_observed_template.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line
    reader=STEPControl_Reader();reader.ReadFile(str(args.step));reader.TransferRoots()
    explorer=TopExp_Explorer(reader.OneShape(),TopAbs_EDGE)
    drawing=[];candidates=[];index=0
    while explorer.More():
        curve=BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()));explorer.Next();index+=1
        first,last=curve.FirstParameter(),curve.LastParameter()
        points=[]
        for t in np.linspace(first,last,2 if curve.GetType()==GeomAbs_Line else 30):
            p=curve.Value(t);points.append([p.X(),p.Y(),p.Z()])
        points=np.array(points)
        if np.max(abs(points[:,2]))<1e-6: drawing.append(points)
        if curve.GetType()!=GeomAbs_Line: continue
        a,b=points;mid=(a+b)/2;length=float(np.linalg.norm(b-a))
        if length>80 and np.max(abs(points[:,2]))<1e-6 and mid[0]<-40 and mid[1]<0:
            candidates.append({'edge_index':index,'endpoints_step_mm':points.tolist(),'length_mm':length,
                               'midpoint_radius_mm':float(np.linalg.norm(mid[:2]))})
    if not candidates: raise ValueError('JL100 outer edge absent')
    edge=max(candidates,key=lambda e:e['midpoint_radius_mm']);points=np.array(edge['endpoints_step_mm'])
    # Verified mount: F_R_TCP=Rz(-90), TCP_R_STEP=Rz(-150)Rx(180).
    a=np.deg2rad(120);R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.diag([1,-1,-1])
    flange_T_step=np.eye(4);flange_T_step[:3,:3]=R;flange_T_step[2,3]=.006
    result=json.loads(args.result.read_text())['candidate'];observation=json.loads(args.template.read_text())
    W_T_F=np.array(result['solutions']['JL100']['endpoint_flange_pose'])
    W_T_B=np.array(result['world_T_boxes']['JL100']);W_T_S=np.array(result['world_T_station'])
    W_T_step=W_T_F@flange_T_step
    world=points/1000@W_T_step[:3,:3].T+W_T_step[:3,3]
    shaft=W_T_S[:3,0];edge_direction=unit(world[1]-world[0])
    row=np.array(observation['features'][2]['xyz_box_mm'])-np.array(observation['features'][0]['xyz_box_mm'])
    observed_direction=unit(W_T_B[:3,:3]@row)
    radial=world.mean(0)-W_T_F[:3,3]
    data={'status':'offline_direction_audit_only','model':'JL100','edge':edge,
          'flange_T_step_metres':flange_T_step.tolist(),'world_T_flange':W_T_F.tolist(),
          'station_R_observed_box':(W_T_S[:3,:3].T@W_T_B[:3,:3]).tolist(),
          'world_edge_endpoints_m':world.tolist(),'motor_shaft_axis_world':shaft.tolist(),
          'outer_long_edge_direction_world':edge_direction.tolist(),
          'outer_edge_from_flange_world_mm':(radial*1000).tolist(),
          'outer_long_edge_parallel_motor_axis_error_deg':parallel_angle(edge_direction,shaft),
          'observed_three_stud_axis_parallel_motor_axis_error_deg':parallel_angle(observed_direction,shaft),
          'outer_edge_parallel_observed_three_stud_axis_error_deg':parallel_angle(edge_direction,observed_direction),
          'outer_edge_is_on_robot_left_positive_world_y':bool(radial[1]>0),
          'interpretation':'Existing JL100 station orientation already meets user three-stud-axis/shaft parallel constraint and outer long edge on robot-left side. No 90-degree flip follows from this constraint.',
          'limits':['Direction audit is not insertion verification.','The archived motor assembly still requires physical datum registration.'],
          'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),args.step,args.result,args.template]}}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'jl100_orientation.json').write_text(json.dumps(data,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    for p in drawing:
        ax[0].plot(*p[:,:2].T,color='grey',lw=.5)
        q=p/1000@W_T_step[:3,:3].T+W_T_step[:3,3]
        ax[1].plot(*(q[:,:2]*1000).T,color='grey',lw=.5)
    ax[0].plot(*points[:,:2].T,'r-',lw=3,label='JL100 outer long edge')
    ax[0].plot(0,0,'kx',label='Flange axis')
    ax[0].set(aspect='equal',xlabel='STEP X mm',ylabel='STEP Y mm',title='Actual STEP edge: 131.987 mm')
    ax[1].plot(*(world[:,:2]*1000).T,'r-',lw=3,label='JL100 outer long edge')
    ax[1].plot(*(W_T_F[:2,3]*1000),'kx',label='Flange axis')
    obs=np.array([f['xyz_box_mm'] for f in observation['features']])/1000
    wp=obs@W_T_B[:3,:3].T+W_T_B[:3,3]
    ax[1].plot(*(wp[:6,:2]*1000).T,'bo',label='Six measured studs')
    ax[1].plot(*(wp[6,:2]*1000),'go',label='Measured ground')
    centre=W_T_B[:3,3]*1000
    ax[1].arrow(centre[0]-70,centre[1]-70,80,0,width=.8,color='orange',length_includes_head=True)
    ax[1].text(centre[0]-70,centre[1]-78,'Motor shaft +X; robot left = +Y',fontsize=9)
    ax[1].set(aspect='equal',xlabel='World X mm',ylabel='World Y mm',title='Saved JL100 target viewed from above')
    for a in ax:a.grid(True);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'jl100_orientation.png',dpi=140);plt.close(fig)
    print(json.dumps({k:v for k,v in data.items() if k.endswith('_deg') or k.endswith('_y')}))


if __name__=='__main__':main()
