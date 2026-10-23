#!/usr/bin/env python3
"""Offline local box/tool envelope study; no robot reachability assertion.

Needle tips remain at the registered mating pose while plate/bosses withdraw.
The active cylinders use the longest requested extension, a conservative union
for every shorter variant. Thus an active-cylinder failure needs a separate
length-specific follow-up; a pass also covers each shorter cylinder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM = Path('/home/wjr/.local/share/jinlong-gpu-planner/ipc/sim-repo')
sys.path.insert(0, str(SIM))
from sim.helios_insert import fixtures as fx
from cycle_validation import ClearancePolicy, _mating_exemption


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, default=ROOT/'outputs/diagnostics/full-solve-20261023/contact_fit/fit_result.json')
    p.add_argument('--box', type=Path, default=ROOT/'outputs/diagnostics/full-solve-20261023/geometry/JM160_collision.json')
    p.add_argument('--tool', type=Path, default=ROOT/'outputs/diagnostics/full-solve-20261023/tool_geometry/tool_collision.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/full-solve-20261023/contact_fit/extension_sweep.json')
    p.add_argument('--max-extension-mm', type=float, default=60.)
    args = p.parse_args()
    fit, box, cad = [json.loads(path.read_text()) for path in [args.fit, args.box, args.tool]]
    site = json.loads((ROOT/'outputs/diagnostics/front-fixed-waist-20261023/input.json').read_text())['site']
    C = np.asarray(site['tcp_T_objects']['JM160']); S = np.eye(4); S[:3, 3] = np.asarray(site['box_center_adjustments_mm']['JM160'])/1000
    B_F = S @ np.linalg.inv(C) @ np.linalg.inv(site['flange_T_tcp'])
    fx.FLANGE_T_TCP_DEG = -90.
    spec = mujoco.MjSpec(); spec.memory = 128*1024*1024
    ground = next(t for t in box['measured_terminals'] if t['name'] == 'term_JM160_gnd')
    z_move = ground['a'][2] + .002 - np.asarray(fit['tool']['ground_slot_tip_box_mm'])[:, 2].mean()/1000
    angle = np.radians(fit['least_squares']['yaw_deg']); R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    shift = np.r_[np.asarray(fit['least_squares']['translation_mm'])/1000, z_move]
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.ravel())
    moving = spec.worldbody.add_body(name='movable_plate_bosses', pos=shift, quat=quat)
    moving.add_joint(name='withdrawal', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0,0,-1], range=[0,.2], limited=True)
    active = spec.worldbody.add_body(name='active_needles')
    active.add_freejoint()
    def mesh(body, name, vertices, faces=None):
        kwargs = {'name': name, 'uservert': np.asarray(vertices).ravel()}
        if faces is not None: kwargs['userface'] = np.asarray(faces, int).ravel()
        spec.add_mesh(**kwargs)
        g=body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
        g.contype, g.conaffinity, g.margin = 4, 0, .01
    def cylinder(body, name, a, b, radius):
        g=body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        g.fromto=np.r_[a,b];g.size=[radius,0,0];g.contype,g.conaffinity,g.margin=4,0,.01
    old=json.loads((SIM/'sim/helios_insert/assets/flange_v2_hulls.json').read_text())
    turn=fx.rz(30.)
    for i, vertices in enumerate(old['hulls']):
        v=np.asarray(vertices)@turn.T;v=v@B_F[:3,:3].T+B_F[:3,3]
        mesh(moving, f'tool_disc_{i}', v)
    linear=np.asarray(fit['tool']['box_from_step_rotation'])
    # This equivalent transform includes the same nominal -2mm X correction.
    for part in cad['parts']:
        v=(np.asarray(part['vertices_step_mm'])-[0,0,6.])@linear.T/1000 + np.linalg.inv(C)[:3,3]+S[:3,3]
        if part['replaces']=='tool_boss_JM160':v += [0,.005,0]
        mesh(moving,part['name'],v,part['faces'])
    for name,a,b,radius in fx.tool_primitives():
        if name.startswith(('tool_boss_', 'tool_spacer_')) or 'JM160' in name:continue
        cylinder(moving,name,B_F[:3,:3]@a+B_F[:3,3],B_F[:3,:3]@b+B_F[:3,3],radius)
    tips=np.asarray(fit['tool']['socket_tip_box_mm'])/1000
    for i, tip in enumerate(tips):
        b=R@tip+shift;a=b-[0,0,.032+args.max_extension_mm/1000]
        cylinder(active,f'tool_sock_JM160_{i+1}',a,b,.005)
    tip=np.asarray(fit['tool']['ground_slot_tip_box_mm']).mean(0)/1000
    tip += [-fit['least_squares']['slot_shift_mm']/1000,.005,0]
    b=R@tip+shift;a=b-[0,0,.022+args.max_extension_mm/1000]
    cylinder(active,'tool_pin_JM160',a,b,.0025)
    for i,o in enumerate(box['boxes']):
        g=spec.worldbody.add_geom(name=f'box_scan_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=o['center'],size=o['half_size'])
        g.contype,g.conaffinity,g.margin=2,4,.01
    for term in box['measured_terminals']:
        a,b=np.array(term['a']),np.array(term['b'])
        if term['name'].startswith('term_JM160_stud_'):
            k=int(term['name'].split('_')[-1])-1
            a[:2]=np.array(fit['refined_stud_caps'][k]['xyz_box_mm'][:2])/1000
            b[:2]=a[:2]
        g=spec.worldbody.add_geom(name=term['name'],type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        g.fromto=np.r_[a,b];g.size=[term['radius'],0,0];g.contype,g.conaffinity,g.margin=2,4,.01
    model=spec.compile();data=mujoco.MjData(model)
    robot=SimpleNamespace(m=model,d=data,_gname=lambda i:model.geom(i).name)
    ctx=SimpleNamespace(robot=robot,cfg={'box_T_base_link':np.eye(4)},tcp='FLANGE')
    policy=ClearancePolicy(); names=tuple(model.geom(i).name for i in range(model.ngeom));addr=model.joint('withdrawal').qposadr[0]
    rows=[]
    for extension in np.arange(0,args.max_extension_mm+.01,1.):
        data.qpos[addr]=extension/1000;mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
        categories={k:(.01,None) for k in ('moving','active','moving_observed','moving_unknown','active_observed','active_unknown')}
        for contact in data.contact[:data.ncon]:
            ns=(names[contact.geom1],names[contact.geom2])
            if _mating_exemption(ctx,contact,ns,'JM160','insert',policy):continue
            toolg=contact.geom1 if ns[0].startswith('tool_') else contact.geom2
            group='moving' if model.geom_bodyid[toolg]==model.body('movable_plate_bosses').id else 'active'
            if contact.dist<categories[group][0]:categories[group]=(float(contact.dist),list(ns))
            obstacle=next((name for name in ns if name.startswith('box_scan_')),None)
            unknown=bool(obstacle and box['boxes'][int(obstacle.split('_')[-1])]['unknown'])
            detail=group+('_unknown' if unknown else '_observed')
            if contact.dist<categories[detail][0]:categories[detail]=(float(contact.dist),list(ns))
        warnings={str(k):int(w.number) for k,w in enumerate(data.warning) if w.number}
        rows.append({'extension_mm':float(extension),'moving_clearance_mm':categories['moving'][0]*1000,'moving_pair':categories['moving'][1],
                     'active_union_clearance_mm':categories['active'][0]*1000,'active_pair':categories['active'][1],
                     'classified_clearance':{k:{'mm':v[0]*1000,'pair':v[1]} for k,v in categories.items()},
                     'warnings':warnings,'local_3mm_pass':bool(not warnings and min(categories['moving'][0],categories['active'][0])>=.003-1e-6)})
    first=next((row for row in rows if row['local_3mm_pass']),None)
    sources=[args.box,args.fit,args.tool,ROOT/'outputs/diagnostics/front-fixed-waist-20261023/input.json',SIM/'sim/helios_insert/assets/flange_v2_hulls.json',Path(__file__)]
    result={'status':'local_geometry_design_candidate_only','source_box':str(args.box),'source_fit':str(args.fit),
            'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'target_extra_depth_mm':z_move*1000,'target_depth_source':'scan collision ground P5 surface + 2mm compression',
            'active_cylinder_union_extension_mm':args.max_extension_mm,'first_1mm_grid_passing_extension':first,
            'first_moving_assembly_3mm_extension':next((r for r in rows if r['moving_clearance_mm']>=3),None),
            'first_observed_moving_assembly_3mm_extension':next((r for r in rows if r['classified_clearance']['moving_observed']['mm']>=3),None),
            'rows':rows,
            'limitations':['No arm, camera, table or trajectory in this local invariant geometry check.',
                           'Active needles are conservatively unioned across all tested lengths; failed active clearance is not an individual-length certificate.',
                           'Scan unknown columns and z quantization remain conservative; an obstruction can require better evidence rather than mechanical removal.',
                           'Needle strength, mounting, manufacturing and electrical engagement are not validated.']}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'first_pass':first,'zero':rows[0],'last':rows[-1]},indent=2))


if __name__=='__main__':main()
