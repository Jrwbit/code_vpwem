#!/usr/bin/env python3
"""Revalidate a historical three-fixture route in a proposed side cell.

All trajectories are checked again, approach is regenerated and stops 30 mm
before mating, and the output is explicitly not executable. No network access.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from search_fixture_layout import FIXTURES, set_product


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim-root',type=Path,required=True)
    ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--turn-deg',type=float,default=90)
    ap.add_argument('--camera-lift-mm',type=float,default=120)
    ap.add_argument('--render',action='store_true')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.sim_root.resolve()))
    from sim.helios_insert import swap as sw,corridor as cr,run as presentation
    from sim.helios_insert.scene import Robot,build,WAIST,LEFT,HEAD,RIGHT
    presentation._FONT=('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',)
    doc=json.loads(args.scene.read_text());original=copy.deepcopy(doc)
    source=json.loads(args.plan.read_text())
    doc['tool_mesh']=str(args.sim_root/'sim/helios_insert/assets/flange_v2_hulls.json')
    doc.pop('viz_boxes',None)
    probe=Robot(build(doc));old_w=np.array(source['waist'])
    probe.set(WAIST,old_w);probe.fk();old_base=sw.arm_base_T(probe)
    w=old_w.copy();w[3]+=np.radians(args.turn_deg)
    for joint,value in zip(WAIST,w):
        if not probe.lim[joint][0]<=value<=probe.lim[joint][1]:raise ValueError('Waist outside model limits')
    probe.set(WAIST,w);probe.fk();delta=sw.arm_base_T(probe)@np.linalg.inv(old_base)
    for ob in doc['obstacles']:
        if 'T' in ob:
            ob['T']=(delta@ob['T']).tolist()
        else:
            low,high=np.array(ob['z_from_floor'])+doc['floor_z']
            T=np.eye(4);T[:3,3]=[*ob['center_xy'],(low+high)/2]
            ob['T']=(delta@T).tolist();ob['size']=[*ob['size_xy'],high-low]
            # An AABB for presentation-only zones; exact tilted slab is used
            # for all collision tests through ob['T'] and ob['size'].
            half=np.abs(delta[:3,:3])@(.5*np.array(ob['size']))
            center=np.array(ob['T'])[:3,3]
            ob['center_xy']=center[:2].tolist();ob['size_xy']=(2*half[:2]).tolist()
            ob['z_from_floor']=[float(center[2]-half[2]-doc['floor_z']),float(center[2]+half[2]-doc['floor_z'])]
    doc['box_T_base_link']=(delta@doc['box_T_base_link']).tolist()
    for t in doc['targets']:t['T']=(delta@t['T']).tolist()
    camera=delta@np.array(original['camera_T_base_link']);camera[2,3]+=args.camera_lift_mm/1000
    cr.apply_camera(doc,camera,center='camera')
    # Current right arm with its board stays in the collision model.
    doc['static_pose']={'head':[.06139625209,-.3190680039],'right_arm':[0.]*7}
    ctx=cr.Ctx(copy.deepcopy(doc),viz=False);robot=ctx.robot
    robot.set(WAIST,w)
    robot.set(HEAD,doc['static_pose']['head']);robot.set(RIGHT,doc['static_pose']['right_arm'])
    targets={t['id']:np.array(t['T']) for t in doc['targets']}
    arm_segments=[s for s in source['segments'] if s['group']=='left_arm']
    home=np.array(arm_segments[0]['start']);current=home.copy();segments=[]
    lo,hi=sw._bounds(robot,LEFT,.02)
    for old in arm_segments:
        name=old['name'];fid=next((f for f in FIXTURES if f in name),None)
        if '插入' in name or '退出' in name:continue
        set_product(robot,fid if '接近' in name else None)
        if '接近' in name:
            robot.set(LEFT,current);robot.fk();Ta=robot.tcp_pose('FLANGE').copy()
            Tb=targets[fid].copy();Tb[:3,3]-=.03*Tb[:3,2]
            if Rotation.from_matrix(Tb[:3,:3]@Ta[:3,:3].T).magnitude()>np.radians(1):
                raise sw.PathError(fid+' historical hover orientation does not match fixture target')
            rows=[np.r_[w,current]]
            for fraction in np.linspace(0,1,17)[1:]:
                T=Ta.copy();T[:3,3]=Ta[:3,3]+fraction*(Tb[:3,3]-Ta[:3,3])
                def residual(q):
                    robot.set(LEFT,q);robot.fk();actual=robot.tcp_pose('FLANGE')
                    return np.r_[T[:3,3]-actual[:3,3],.3*Rotation.from_matrix(T[:3,:3]@actual[:3,:3].T).as_rotvec()]
                fit=least_squares(residual,current,bounds=(lo,hi),max_nfev=100)
                error=residual(fit.x)
                if np.linalg.norm(error[:3])>.0005 or np.linalg.norm(error[3:])>.0009:raise sw.PathError(fid+' approach IK residual')
                sw.check_joint_line(robot,doc,rows[-1],np.r_[w,fit.x],'FLANGE')
                current=fit.x;rows.append(np.r_[w,current])
            forward=sw._waypoint_segment(fid+' approach to 30 mm (no contact)','left_arm',rows,'FLANGE')
            backward=sw._waypoint_segment(fid+' retract to hover','left_arm',list(reversed(rows)),'FLANGE')
            for seg in (forward,backward):seg.update(active_product=fid,motor=fid)
            segments.extend([forward,backward]);current=rows[0][4:].copy()
        else:
            rows=[np.r_[w,current]]+[np.r_[w,step['target']] for step in old['steps']]
            for a,b in zip(rows,rows[1:]):sw.check_joint_line(robot,doc,a,b,'FLANGE')
            segments.append(sw._waypoint_segment(name,'left_arm',rows,'FLANGE'));current=rows[-1][4:].copy()
        print('PASS',name,flush=True)
    if np.max(np.abs(current-home))>.001:raise sw.PathError('Cycle did not return home')
    stats=[];frames=[]
    start={'waist':w.tolist(),'left_arm':home.tolist(),**doc['static_pose']};frame_start=copy.deepcopy(start)
    for seg in segments:
        set_product(robot,seg.get('active_product'))
        stats.append(cr.seg_stats(ctx,seg))
        frames+=sw.replay_frames(robot,doc,[seg],frame_start)
        frame_start['left_arm']=seg['waypoints'][-1][4:].tolist()
    set_product(robot)
    if min(s['clear_min_mm'] for s in stats)<7.9:raise sw.PathError('Replay clearance failed')
    plan=sw.export_plan(robot,doc,segments,start,{'waist':w,'sols':{}},True,args.out/'cycle_preview.json')
    plan.update(executable=False,execution_authorized=False,note='Hypothetical relocated cell. Regenerated noncontact approach only. Never execute on physical robot.')
    (args.out/'cycle_preview.json').write_text(json.dumps(plan,indent=2))
    report={'simulation_passed':True,'execution_authorized':False,'controller_writes':False,
            'box_xyz_base_link_m':np.array(doc['box_T_base_link'])[:3,3].tolist(),
            'camera_xyz_base_link_m':camera[:3,3].tolist(),'waist_model_deg':np.degrees(w).tolist(),
            'floor_z':doc['floor_z'],'torso_turn_deg':args.turn_deg,'camera_lift_mm':args.camera_lift_mm,
            'nearest_mm':30,'segments':stats,'fixture_order':['JL100','JM160','JL132'],
            'joint_travel_deg':sum(float(np.degrees(np.abs(b[4:]-a[4:]).max())) for s in segments for a,b in zip(s['waypoints'],s['waypoints'][1:])),
            'assumptions':'Historical nominal tool/box geometry; tilted motor/table stand following torso transform; camera raised relative to cell. No supports/cables/vision field of view or contact/electrical inspection validated. Start is fixed work-waist home, not live robot pose.'}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.out/'scene_used.json').write_text(json.dumps(doc,indent=2))
    sw._set_start(robot,start)
    cr.shot(ctx,args.out/'layout.png','Three-tool left-side layout (simulation only)', ['Fixed waist; approach stops 30 mm outside box','Nominal geometry; no robot commands'])
    if args.render:sw.render(robot,doc,frames,args.out/'three_tool_cycle.mp4','Three-tool left-side study (no contact)', ['JL100 -> JM160 -> JL132 -> home','Fixed work waist; camera raised 120 mm; no real execution'])
    print('PASSED',args.out,flush=True)


if __name__=='__main__':main()
