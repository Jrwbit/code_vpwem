#!/usr/bin/env python3
"""Offline layout study: move the complete camera/motor/table cell, test all tools.

Uses historical geometry, never connects to a robot, and never grants execution.
The cycle deliberately stops 30 mm outside the nominal mating pose (no contact).
"""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FIXTURES = ('JL100', 'JL132', 'JM160')


def set_product(robot, fixture=None):
    """A box has one product's terminals. Keep every tool and box collision.

    Switching uses the conservative union of all terminal layouts. This does
    not enable the upstream insertion exceptions for disc/box or mating pairs.
    """
    if not hasattr(robot, '_layout_terminal_masks'):
        robot._layout_terminal_masks = [(g, int(robot.m.geom_contype[g]), int(robot.m.geom_conaffinity[g]))
            for g in range(robot.m.ngeom) if robot.m.geom(g).name.startswith('term_')]
    for g, contype, conaffinity in robot._layout_terminal_masks:
        active = fixture is None or robot.m.geom(g).name.startswith('term_'+fixture+'_')
        robot.m.geom_contype[g] = contype if active else 0
        robot.m.geom_conaffinity[g] = conaffinity if active else 0


def historical_connect(sw, robot, doc, A, B, historical):
    """Lazy roadmap of historical joint seeds, with every new edge rechecked."""
    import heapq
    if np.max(np.abs(A-B))<1e-9:return [A,B]
    nodes = [A,B] + [np.r_[A[:4],q] for q in historical]
    # Historical trajectories can have repeated endpoints.
    unique=[]
    for q in nodes:
        if not any(np.max(np.abs(q-other))<1e-6 for other in unique):unique.append(q)
    nodes=unique
    if len(nodes)<2:raise sw.PathError('degenerate switching endpoints')
    queue=[(0.,0)];cost={0:0.};parent={};edges={}
    while queue:
        value,i=heapq.heappop(queue)
        if value>cost[i]+1e-9:continue
        if i==1:
            indices=[1]
            while indices[-1]!=0:indices.append(parent[indices[-1]])
            return [nodes[k] for k in reversed(indices)]
        neighbours=sorted((j for j in range(len(nodes)) if j!=i),key=lambda j:np.max(np.abs(nodes[j]-nodes[i])))
        for j in neighbours:
            distance=float(np.max(np.abs(nodes[j]-nodes[i])))
            if value+distance>=cost.get(j,float('inf')):continue
            key=tuple(sorted((i,j)))
            if key not in edges:
                try:
                    sw.check_joint_line(robot,doc,nodes[i],nodes[j],'FLANGE');edges[key]=True
                except sw.PathError:edges[key]=False
            if edges[key]:
                cost[j]=value+distance;parent[j]=i;heapq.heappush(queue,(cost[j],j))
    raise sw.PathError('no path through rechecked historical roadmap')


def flange_target(box, site, fixture, stand_off):
    offset = np.array(site['tcp_T_objects'][fixture], float)
    offset[2, 3] += site.get('fixture_z_adjustments_mm', {}).get(fixture, 0) / 1000
    shift, z = np.eye(4), np.eye(4)
    shift[:3, 3] = np.array(site.get('box_center_adjustments_mm', {}).get(fixture, [0, 0, 0])) / 1000
    z[2, 3] = site.get('tcp_z_adjustment_mm', 0) / 1000
    target = box @ shift @ np.linalg.inv(offset) @ np.linalg.inv(np.array(site['flange_T_tcp']) @ z)
    target[:3, 3] -= box[:3, 2] * stand_off
    return target


def relocate(scene, xyz, yaw):
    """Rigidly move the whole cell, including table and camera, in world axes."""
    doc = copy.deepcopy(scene)
    old = np.array(doc['box_T_base_link'])
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    box = turn @ old
    box[:3, 3] = xyz
    delta = box @ np.linalg.inv(old)
    for ob in doc['obstacles']:
        if 'T' in ob:
            ob['T'] = (delta @ np.array(ob['T'])).tolist()
        elif ob['name'] == 'table':
            low, high = np.array(ob['z_from_floor']) + doc['floor_z']
            center = delta @ [*ob['center_xy'], (low + high) / 2, 1]
            ob['T'] = turn.copy().tolist()
            ob['T'] = np.array(ob['T'])
            ob['T'][:3, 3] = center[:3]
            ob['T'] = ob['T'].tolist()
            ob['size'] = [*ob['size_xy'], high - low]
            # Upstream presentation zones use an axis-aligned table envelope;
            # collision geometry itself retains the exact rotated slab above.
            ob['center_xy'] = center[:2].tolist()
            ob['size_xy'] = (np.abs(turn[:2,:2]) @ np.array(ob['size_xy'])).tolist()
            ob['z_from_floor'] = [float(center[2]-(high-low)/2-doc['floor_z']),
                                  float(center[2]+(high-low)/2-doc['floor_z'])]
    doc['box_T_base_link'] = box.tolist()
    doc.pop('viz_boxes', None)
    return doc, delta @ np.array(scene['camera_T_base_link'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim-root', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--scene', type=Path, required=True, help='World-anchored scene from prior hover study')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--work-waist-deg', type=float, nargs=4,
                    help='Fixed simulated work posture, model degrees; not applied to robot')
    ap.add_argument('--lower-elbow', action='store_true', help='Use existing redundant-arm elbow descent after IK')
    ap.add_argument('--high-grid', action='store_true', help='Test elevated side cell and additional yaw angles')
    ap.add_argument('--refine-clearance', action='store_true', help='Refine redundancy for wrist/elbow/obstacle clearance')
    ap.add_argument('--compact-stand', action='store_true', help='Propose 500 x 500 mm support under motor instead of moving the existing wide table')
    ap.add_argument('--seed-plan', type=Path, help='Historical joint angles as IK seeds only, all new paths rechecked')
    ap.add_argument('--rigid-left', action='store_true', help='Search around a 90 degree left rotation of the original cell')
    ap.add_argument('--allow-detours', action='store_true', help='Allow checked intermediate joint positions when direct switching fails')
    ap.add_argument('--adaptive-hover', action='store_true', help='Try 60..150 mm tool-specific retreat distances')
    ap.add_argument('--camera-lift-mm', type=float, default=0, help='Explicit proposed change of camera height relative to motor')
    ap.add_argument('--prefer-high-hover', action='store_true', help='Try high retreat first to allow turning above the motor')
    args = ap.parse_args()
    if args.limit < 0 or (args.rigid_left and args.limit == 0) or not np.isfinite(args.camera_lift_mm):
        ap.error('invalid search count or camera lift')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.sim_root.resolve()))
    from sim.helios_insert import corridor as cr, swap as sw
    from sim.helios_insert.scene import WAIST, LEFT, RIGHT, HEAD, Robot, build
    from sim.helios_insert import run as presentation
    presentation._FONT = ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',)
    bundle = json.loads(args.bundle.read_text())
    scene = json.loads(args.scene.read_text())
    scene['tool_mesh'] = str(args.sim_root / 'sim/helios_insert/assets/flange_v2_hulls.json')
    joints = bundle['state']['real']['joints']
    waist = np.array([joints[n] for n in WAIST])
    if args.work_waist_deg is not None:
        waist = np.radians(args.work_waist_deg)
    start_arm = np.array([joints[n] for n in LEFT])
    scene['static_pose'] = {'head': [joints[n] for n in HEAD], 'right_arm': [joints[n] for n in RIGHT]}
    if args.compact_stand:
        # Explicit alternative hardware, not removal of an obstacle. Preserve
        # table top and thickness, place its centre under the motor.
        table = next(ob for ob in scene['obstacles'] if ob['name']=='table')
        if 'T' in table:
            table['T'][0][3], table['T'][1][3] = np.array(scene['box_T_base_link'])[:2,3]
            table['size'][:2] = [.5,.5]
        else:
            table['center_xy'] = np.array(scene['box_T_base_link'])[:2,3].tolist()
            table['size_xy'] = [.5,.5]
    site = bundle['site']
    rng = np.random.default_rng(20261017)
    results, viable = [], []
    # +X forward, +Y robot left, +Z up in base_link, not camera coordinates.
    candidates = [('chest_baseline', np.array(scene['box_T_base_link'])[:3, 3], 0)]
    grid = list(itertools.product((.20, .35), (.45, .60, .75), (.80, .90), (0, 90)))
    if args.high_grid:
        grid = list(itertools.product((.20, .35), (.45, .60), (1.0, 1.1), (90, 180, -90)))
    candidates += [(f'left_{i:02}', np.array([x, y, z]), yaw) for i, (x, y, z, yaw) in enumerate(grid[:args.limit])]
    if args.rigid_left:
        origin = np.array(scene['box_T_base_link'])[:3,3]
        # Waist J4 is the torso turn, while J1..J3 bend/translate its axis.
        # Account for that offset; a world-origin rotation alone places the
        # cell incorrectly relative to the turned shoulder.
        reference_robot = Robot(build(scene))
        reference_waist = waist.copy(); reference_waist[3] -= np.pi/2
        reference_robot.set(WAIST,reference_waist);reference_robot.fk()
        old_shoulder=reference_robot.d.xpos[reference_robot.m.body('AR5-5_07L_link2').id].copy()
        reference_robot.set(WAIST,waist);reference_robot.fk()
        new_shoulder=reference_robot.d.xpos[reference_robot.m.body('AR5-5_07L_link2').id].copy()
        turn=Rotation.from_euler('z',90,degrees=True)
        center = new_shoulder+turn.apply(origin-old_shoulder)
        candidates = [('left_rigid_'+str(i),center+delta,yaw) for i,(delta,yaw) in enumerate(
            itertools.product(([0,0,0],[0,.08,0],[.08,0,0],[-.08,0,0],[0,0,.08],[0,0,-.08]),(90,75,105)))][:args.limit]
    previous = []
    historical = []
    if args.seed_plan:
        seed_plan = json.loads(args.seed_plan.read_text())
        historical = [np.array(step['target']) for s in seed_plan['segments'] if s['group']=='left_arm' for step in s['steps']]
    diagnostics = []
    for name, xyz, yaw in candidates:
        doc, camera = relocate(scene, xyz, yaw)
        camera[2,3] += args.camera_lift_mm/1000
        cr.apply_camera(doc, camera, center='camera')
        box = np.array(doc['box_T_base_link'])
        doc['targets'] = [{'id': f, 'tcp': 'FLANGE', 'T': flange_target(box, site, f, 0).tolist()} for f in FIXTURES]
        robot = Robot(build(doc))
        for joint, value in zip(WAIST,waist):
            if not robot.lim[joint][0] <= value <= robot.lim[joint][1]:
                raise ValueError(f'Simulated waist outside model limit: {joint}')
        robot.set(WAIST, waist); robot.set(HEAD, doc['static_pose']['head']); robot.set(RIGHT, doc['static_pose']['right_arm'])
        lo, hi = sw._bounds(robot, LEFT, .055)
        def solve(T, seeds):
            def residual(q):
                robot.set(LEFT, q); robot.fk()
                actual = robot.tcp_pose('FLANGE')
                return np.r_[T[:3, 3] - actual[:3, 3], .3 * Rotation.from_matrix(T[:3, :3] @ actual[:3, :3].T).as_rotvec()]
            found = []
            for seed in seeds:
                fit = least_squares(residual, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=100)
                if args.lower_elbow and np.linalg.norm(residual(fit.x)) < .001:
                    _, lowered = sw.ik_low_elbow(robot, T, 'FLANGE', fit.x, iters=60)
                    # Original solver uses a small-angle rotation residual. The
                    # independent full SO(3) residual remains mandatory here.
                    fit = least_squares(residual, np.clip(lowered, lo+1e-6, hi-1e-6), bounds=(lo,hi), max_nfev=80)
                if args.refine_clearance and np.linalg.norm(residual(fit.x)) < .01:
                    def merit(q):
                        e = residual(q)
                        quality = sw.quality(robot, 'FLANGE')
                        return np.r_[e, .05*max(quality['elbow']+.08,0),
                                     .04*max(quality['wrist']-.8,0), .3*max(.025-quality['clear'],0)]
                    fit = least_squares(merit,fit.x,bounds=(lo,hi),max_nfev=100)
                    fit = least_squares(residual,fit.x,bounds=(lo,hi),max_nfev=80)
                e = residual(fit.x)
                quality = sw.quality(robot, 'FLANGE')
                diagnostics.append({'position_error_mm':float(np.linalg.norm(e[:3])*1000),
                                    'angle_error_deg':float(np.degrees(np.linalg.norm(e[3:])/.3)),
                                    **quality})
                if np.linalg.norm(e[:3]) < .0005 and np.linalg.norm(e[3:]) < .0009 and sw.legal(quality, doc):
                    found.append((fit.x.copy(), quality))
                    if len(found) >= 2: break
            return sorted(found, key=lambda item: item[1]['wrist'] + max(item[1]['elbow'],0))
        report = {'name': name, 'box_xyz_m': xyz.tolist(), 'cell_yaw_deg': yaw,
                  'camera_xyz_m': camera[:3, 3].tolist(), 'waist_model_deg': np.degrees(waist).tolist(),
                  'all_hover_reachable': False, 'cycle_passed': False, 'failures': []}
        hover_solutions = []
        hover_distances = []
        diagnostics = []
        for fid in FIXTURES:
            seeds = ([hover_solutions[-1][0]] if hover_solutions else []) + previous[:3] + historical + [start_arm] + [rng.uniform(lo, hi) for _ in range(4)]
            sols = []
            distances = ((.15,.12,.11,.08,.06) if args.prefer_high_hover else (.08,.11,.12,.06,.15)) if args.adaptive_hover else (.110,)
            for hover in distances:
                sols = solve(flange_target(box, site, fid, hover), seeds)
                if sols: break
            if not sols:
                report['failures'].append('no legal hover: ' + fid); break
            hover_solutions.append(sols[0])
            hover_distances.append(hover)
        if len(hover_solutions) == 3:
            report['all_hover_reachable'] = True
            report['hover_q_rad'] = [q.tolist() for q,_ in hover_solutions]
            report['hover_mm'] = dict(zip(FIXTURES,[d*1000 for d in hover_distances]))
            previous = [q for q, _ in hover_solutions]
            segments = []
            try:
                # Full cyclic switching, including last -> first. Each swept joint
                # segment is checked at <=0.5 deg, plus original elbow/wrist gates.
                for i, fid in enumerate(FIXTURES):
                    set_product(robot,fid)
                    qh = hover_solutions[i][0]
                    Th = flange_target(box, site, fid, hover_distances[i])
                    Tp = flange_target(box, site, fid, .030)
                    rows = [np.r_[waist, qh]]
                    qcur = qh.copy()
                    for fraction in np.linspace(0, 1, 9)[1:]:
                        T = Th.copy(); T[:3, 3] = Th[:3, 3] + fraction * (Tp[:3, 3] - Th[:3, 3])
                        sols = solve(T, [qcur])
                        if not sols:
                            report['failed_approach'] = {'fixture':fid,'stand_off_mm':float((hover_distances[i]+fraction*(.03-hover_distances[i]))*1000),
                                                         'last_ik':diagnostics[-1]}
                            raise sw.PathError(fid + ' approach IK/clearance failed')
                        qnext = sols[0][0]
                        A, B = rows[-1], np.r_[waist, qnext]
                        sw.check_joint_line(robot, doc, A, B, 'FLANGE')
                        # Bound actual interpolated TCP deviation from desired line.
                        for alpha in np.linspace(0, 1, 7):
                            robot.set(LEFT, qcur + alpha * (qnext - qcur)); robot.fk()
                            actual = robot.tcp_pose('FLANGE')
                            expected_p = Th[:3, 3] + ((fraction-.125) + .125*alpha) * (Tp[:3, 3]-Th[:3, 3])
                            if np.linalg.norm(actual[:3, 3]-expected_p) > .003 or Rotation.from_matrix(T[:3,:3] @ actual[:3,:3].T).magnitude() > np.radians(1):
                                raise sw.PathError(fid + ' interpolation deviates from normal')
                        rows.append(B); qcur = qnext
                    segments.append(sw._waypoint_segment(fid + f' approach {hover_distances[i]*1000:.0f} to 30 mm (no contact)', 'left_arm', rows, 'FLANGE'))
                    segments.append(sw._waypoint_segment(fid + f' retract 30 to {hover_distances[i]*1000:.0f} mm', 'left_arm', list(reversed(rows)), 'FLANGE'))
                    for seg in segments[-2:]:
                        seg['active_product'] = fid
                        seg['motor'] = fid
                    set_product(robot)
                    qnext = hover_solutions[(i+1)%3][0]
                    A, B = np.r_[waist,qh], np.r_[waist,qnext]
                    try:
                        sw.check_joint_line(robot, doc, A, B, 'FLANGE')
                        switch_rows = [A,B]
                    except sw.PathError:
                        if not args.allow_detours: raise
                        try:
                            switch_rows = historical_connect(sw,robot,doc,A,B,historical)
                        except sw.PathError:
                            switch_rows = sw.connect(robot,doc,A,B,'FLANGE',rng,n_samples=80)
                    segments.append(sw._waypoint_segment(fid + ' switch to ' + FIXTURES[(i+1)%3], 'left_arm', switch_rows, 'FLANGE'))
                report['cycle_passed'] = True
                report['joint_travel_deg'] = sum(float(np.degrees(np.abs(b[4:]-a[4:]).max())) for seg in segments for a,b in zip(seg['waypoints'],seg['waypoints'][1:]))
                report['switch_joint_travel_deg'] = sum(float(np.degrees(np.abs(b[4:]-a[4:]).max())) for seg in segments if ' switch ' in seg['name'] for a,b in zip(seg['waypoints'],seg['waypoints'][1:]))
                # Kinematic speed proxy only, no acceleration/IO/detection time.
                report['hover_mm'] = dict(zip(FIXTURES,[d*1000 for d in hover_distances]))
                report['speed_proxy_seconds'] = report['switch_joint_travel_deg']/20 + sum(2*(h-.03)*1000/50 for h in hover_distances)
                report['hover_q_rad'] = [q.tolist() for q,_ in hover_solutions]
                viable.append((report['speed_proxy_seconds'], doc, segments, report))
            except sw.PathError as error:
                report['failures'].append(str(error))
        results.append(report)
        report['best_failed_or_solved_ik'] = sorted(diagnostics, key=lambda d:d['position_error_mm']+d['angle_error_deg'])[:3]
        (args.out/'search.json').write_text(json.dumps(results, indent=2))
        print(name, xyz, yaw, report['all_hover_reachable'], report['cycle_passed'], report['failures'], flush=True)
    summary = {'controller_writes': False, 'execution_authorized': False,
               'camera_lift_mm': args.camera_lift_mm,
               'compact_stand_500mm': args.compact_stand,
               'waist_fixed_for_entire_cycle': True, 'hover_search_mm': [int(h*1000) for h in distances], 'nearest_mm':30,
               'scope':'Three tools at same hypothetical box; noncontact approach/retract and cyclic switching. No electrical inspection/contact simulated.',
               'assumptions':'Nominal tool geometry; entire table/camera/motor rigidly relocated. Supports, cables and vision visibility not verified. Speed proxy assumes joints20deg/s, approach50mm/s; excludes acceleration and dwell.',
               'tested':len(results),'passed':len(viable),'results':results}
    if viable:
        _, doc, segments, best = min(viable, key=lambda item:item[0])
        ctx = cr.Ctx(copy.deepcopy(doc), viz=False)
        best['segment_stats'] = []
        for s in segments:
            set_product(ctx.robot,s.get('active_product'))
            best['segment_stats'].append(cr.seg_stats(ctx,s))
        set_product(ctx.robot)
        summary['best'] = best
        (args.out/'best_scene.json').write_text(json.dumps(doc,indent=2))
        start = {'waist':waist.tolist(),'left_arm':best['hover_q_rad'][0],**doc['static_pose']}
        sw.export_plan(ctx.robot, doc, segments, start, {'waist':waist,'sols':{}}, True, args.out/'cycle_preview.json')
        plan=json.loads((args.out/'cycle_preview.json').read_text());plan['execution_authorized']=False;plan['executable']=False
        plan['note']='Hypothetical relocated cell, noncontact 110/30 mm cycle only; never execute on current physical layout'
        (args.out/'cycle_preview.json').write_text(json.dumps(plan,indent=2))
        sw._set_start(ctx.robot,start)
        cr.shot(ctx,args.out/'best_layout.png','Left-side cell layout study (simulation only)',[str(np.round(np.array(best['box_xyz_m'])*1000))+' mm; fixed waist; all three tools','No contact, no real robot execution'])
        if args.render:
            frames=[]
            current_start=copy.deepcopy(start)
            for seg in segments:
                set_product(ctx.robot,seg.get('active_product'))
                frames += sw.replay_frames(ctx.robot,doc,[seg],current_start)
                current_start['left_arm'] = seg['waypoints'][-1][4:].tolist()
            set_product(ctx.robot)
            sw.render(ctx.robot,doc,frames,args.out/'three_tool_cycle.mp4','Left-side three-tool cycle (simulation only)', ['Tool-specific hover -> 30 mm -> hover; no contact','All three tools, fixed waist; supports/cables not modelled'])
    (args.out/'report.json').write_text(json.dumps(summary,indent=2))
    print('Completed:',args.out,'passed',len(viable),flush=True)


if __name__ == '__main__':
    main()
