#!/usr/bin/env python3
"""Offline, pair-specific checks for a fixed-waist inspection cycle.

This does not operate hardware or change the installed planner.  Local clearances
are study parameters, not measured manufacturing tolerances.  Collision coverage
is the loaded MuJoCo model's masks; all samples and interpolation are checked,
not a mathematical continuous-collision certificate.  Terminal geometry must be
independently registered before these checks imply physical mating feasibility.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import numpy as np


LOCAL_PHASES = frozenset(('approach', 'insert', 'retract'))
PHASES = LOCAL_PHASES | {'transit', 'corridor'}


@dataclass(frozen=True)
class ClearancePolicy:
    normal_m: float = .008
    local_box_m: float = .003
    local_terminal_m: float = .003
    transition_m: float = .030
    generation_buffer_m: float = .002
    sample_deg: float = .25
    minimum_intervals: int = 2
    max_ground_compression_m: float = .002
    socket_bore_radius_m: float = .0035
    mating_angle_deg: float = 1.0
    numerical_tolerance_m: float = 1e-6

    def __post_init__(self):
        positive = (self.normal_m, self.transition_m, self.sample_deg,
                    self.socket_bore_radius_m, self.mating_angle_deg)
        nonnegative = (self.local_box_m, self.local_terminal_m,
                       self.generation_buffer_m, self.max_ground_compression_m,
                       self.numerical_tolerance_m)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError('positive, finite clearance and sampling parameters required')
        if not all(math.isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError('local margins/compression cannot be negative or nonfinite')
        if max(self.local_box_m, self.local_terminal_m) > self.normal_m:
            raise ValueError('local margins cannot exceed ordinary clearance')
        if self.minimum_intervals < 2:
            raise ValueError('every edge must include its midpoint')


def _imports():
    import mujoco
    from sim.helios_insert import corridor, plan, scene, swap
    return mujoco, corridor, plan, scene, swap


def target_pose(ctx, fid):
    for t in ctx.cfg['targets']:
        if t['id'] == fid:
            if 'T' not in t:
                return next(t.T for t in _imports()[3].targets(ctx.cfg) if t.id == fid)
            return np.asarray(t['T'], float)
    raise ValueError(f'no final target for {fid}')


def _angle_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def _normal(ctx):
    return np.asarray(ctx.cfg['box_T_base_link'], float)[:3, 2]


def _fraction(ctx, fid, phase, policy):
    if phase not in PHASES:
        raise ValueError(f'unknown phase: {phase}')
    if phase not in LOCAL_PHASES:
        return 1.0
    T = ctx.robot.tcp_pose(ctx.tcp)
    remaining = float((target_pose(ctx, fid)[:3, 3] - T[:3, 3]) @ _normal(ctx))
    return float(np.clip(remaining / policy.transition_m, 0, 1))


def pair_clearance(names, phase, fraction, policy):
    """Required effective separation, before the existing tool/body padding.

    Only tool against local workpiece geometry receives the local taper.  Arm,
    camera, table, and robot/body pairs keep ordinary separation at every phase.
    """
    if phase not in PHASES:
        raise ValueError(f'unknown phase: {phase}')
    if phase not in LOCAL_PHASES or not any(n.startswith('tool_') and not n.startswith('tool_board') for n in names):
        return policy.normal_m
    if any(n.startswith('term_') for n in names):
        local = policy.local_terminal_m
    elif any(n.startswith('box_') for n in names):
        local = policy.local_box_m
    else:
        return policy.normal_m
    return local + float(np.clip(fraction, 0, 1)) * (policy.normal_m - local)


def _inactive_terminal(names, fid):
    return any(n.startswith('term_') and not n.startswith(f'term_{fid}_') for n in names)


def _cylinder_ends(robot, gid, normal):
    """Actual loaded geometry, never fixture-generated reference terminals."""
    import mujoco
    if robot.m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        return None
    center = robot.d.geom_xpos[gid]
    axis = robot.d.geom_xmat[gid].reshape(3, 3)[:, 2]
    if axis @ normal < 0:
        axis = -axis
    half = robot.m.geom_size[gid, 1]
    return center - half * axis, center + half * axis, axis


def _mating_exemption(ctx, c, names, fid, phase, policy):
    """Only hollow sockets and an explicitly bounded spring compression.

    Tool plate / box, and solid ground-pin boss / screw are never exempted.
    """
    if phase not in LOCAL_PHASES:
        return False
    pairs = {(f'tool_sock_{fid}_{k}', f'term_{fid}_stud_{k}') for k in (1, 2, 3)}
    ground = (f'tool_pin_{fid}', f'term_{fid}_gnd')
    pairs.add(ground)
    if names not in pairs and names[::-1] not in pairs:
        return False
    tool_first = names[0].startswith('tool_')
    ga, gb = (c.geom1, c.geom2) if tool_first else (c.geom2, c.geom1)
    robot = ctx.robot
    normal = _normal(ctx)
    a, b = _cylinder_ends(robot, ga, normal), _cylinder_ends(robot, gb, normal)
    if a is None or b is None:
        return False
    abottom, atip, axis = a
    btop, _, baxis = b
    alignment = math.degrees(math.acos(float(np.clip(axis @ baxis, -1, 1))))
    d = atip - btop
    lateral = float(np.linalg.norm(d - (d @ baxis) * baxis))
    engage = float(d @ baxis)
    if alignment > policy.mating_angle_deg:
        return False
    canonical = names if tool_first else names[::-1]
    if canonical == ground:
        # Cylindrical approximation: require the pin footprint to fit the screw
        # head; the allowed overlap represents bounded spring compression only.
        return (lateral <= max(0., robot.m.geom_size[gb, 0] - robot.m.geom_size[ga, 0])
                and engage <= policy.max_ground_compression_m + policy.numerical_tolerance_m)
    radial_room = policy.socket_bore_radius_m - robot.m.geom_size[gb, 0]
    # Include angular drift across the overlap, so an inclined stud cannot use
    # centreline agreement at the socket tip to pass through the socket wall.
    occupied_radius = lateral + max(0., engage) * math.sin(math.radians(alignment))
    return (radial_room >= 0 and occupied_radius <= radial_room + policy.numerical_tolerance_m
            and engage <= np.linalg.norm(atip - abottom) + policy.numerical_tolerance_m)


def _collision_warnings(robot):
    import mujoco
    categories = ('mjWARN_CONTACTFULL', 'mjWARN_CNSTRFULL', 'mjWARN_BADQPOS')
    return {name: int(robot.d.warning[int(getattr(mujoco.mjtWarning, name))].number)
            for name in categories
            if robot.d.warning[int(getattr(mujoco.mjtWarning, name))].number}


def _collision_cache(ctx, fid):
    r = ctx.robot
    previous = getattr(r, '_cycle_collision_cache', None)
    if previous is not None and previous['model'] is r.m and previous['fid'] == fid:
        return previous
    names = tuple(r._gname(g) for g in range(r.m.ngeom))
    tool = np.array([n.startswith('tool_') for n in names])
    cache = {
        'model': r.m, 'fid': fid, 'names': names, 'tool': tool,
        'local_tool': np.array([n.startswith('tool_') and not n.startswith('tool_board') for n in names]),
        'box': np.array([n.startswith('box_') for n in names]),
        'terminal': np.array([n.startswith('term_') for n in names]),
        'inactive': np.array([n.startswith('term_') and not n.startswith(f'term_{fid}_') for n in names]),
        'contype': np.array(r.m.geom_contype, copy=True),
        'body_padding_cache': {},
    }
    from sim.helios_insert.scene import Robot, TOOL_BODY_PAD
    cache['standard_body_padding'] = getattr(r.body_pad, '__func__', None) is Robot.body_pad
    cache['body_padding_m'] = TOOL_BODY_PAD
    ids = {name: i for i, name in enumerate(names)}
    pairs = [(f'tool_sock_{fid}_{k}', f'term_{fid}_stud_{k}') for k in (1, 2, 3)]
    pairs.append((f'tool_pin_{fid}', f'term_{fid}_gnd'))
    cache['mating_codes'] = np.array([a * r.m.ngeom + b
        for x, y in pairs if x in ids and y in ids
        for a, b in ((ids[x], ids[y]), (ids[y], ids[x]))], dtype=np.int64)
    r._cycle_collision_cache = cache
    return cache


def _contact_arrays(ctx, fid, phase, policy, generation=False):
    """Bulk distance/margin classification; only candidate mates need Python FK geometry.

    No distance-based culling: every reported contact contributes to the exact
    minimum and violation set. Cache entries contain geometry identity only,
    never distances, positions, phase thresholds, or mating decisions.
    """
    mujoco = _imports()[0]
    r = ctx.robot
    r.fk(); mujoco.mj_collision(r.m, r.d)
    cache = _collision_cache(ctx, fid)
    geoms = r.d.contact.geom
    g1, g2 = geoms[:, 0], geoms[:, 1]
    codes = g1.astype(np.int64) * r.m.ngeom + g2
    keep = ~(cache['inactive'][g1] | cache['inactive'][g2])
    if phase in LOCAL_PHASES and len(cache['mating_codes']):
        candidates = np.flatnonzero(keep & np.isin(codes, cache['mating_codes']))
        for i in candidates:
            names = cache['names'][g1[i]], cache['names'][g2[i]]
            if _mating_exemption(ctx, r.d.contact[int(i)], names, fid, phase, policy):
                keep[i] = False
    indices = np.flatnonzero(keep)
    a, b = g1[indices], g2[indices]
    fraction = _fraction(ctx, fid, phase, policy)
    thresholds = np.full(len(indices), policy.normal_m)
    if phase in LOCAL_PHASES:
        tool = cache['local_tool'][a] | cache['local_tool'][b]
        box = tool & (cache['box'][a] | cache['box'][b])
        terminal = tool & (cache['terminal'][a] | cache['terminal'][b])
        thresholds[box] = policy.local_box_m + fraction * (policy.normal_m - policy.local_box_m)
        thresholds[terminal] = policy.local_terminal_m + fraction * (policy.normal_m - policy.local_terminal_m)
    if generation:
        thresholds += policy.generation_buffer_m * (fraction if phase in LOCAL_PHASES else 1.)
    if cache['standard_body_padding']:
        one_tool = cache['tool'][a] ^ cache['tool'][b]
        other = np.where(cache['tool'][a], b, a)
        pads = np.where(one_tool & ((cache['contype'][other] & 1) != 0), cache['body_padding_m'], 0.)
    else:
        # Preserve custom Robot implementations rather than silently assuming
        # the standard scene's tool/body-padding definition.
        saved = cache['body_padding_cache']
        for code, i in zip(codes[indices], indices):
            if int(code) not in saved:
                saved[int(code)] = r.body_pad(r.d.contact[int(i)])
        pads = np.array([saved[int(code)] for code in codes[indices]])
    clearances = np.asarray(r.d.contact.dist)[indices] - pads
    return indices, clearances, thresholds, pads, cache['names']


def _contact_rows(ctx, fid, phase, policy, generation=False):
    indices, clears, thresholds, pads, names = _contact_arrays(ctx, fid, phase, policy, generation)
    for i, clear, threshold, pad in zip(indices, clears, thresholds, pads):
        c = ctx.robot.d.contact[int(i)]
        yield c, (names[c.geom1], names[c.geom2]), float(clear), float(threshold), float(pad)


def check_segment(ctx, a, b, phase, fid, policy=None, Ta=None, Tb=None,
                  dev_mm=None, angle_deg=1.0, waist=None):
    """Densely check one actual joint-interpolated edge, including its midpoint.

    Ta/Tb describe the full Cartesian path, not necessarily this edge. The caller
    checks path endpoint completion separately. No start-state violation escape.
    """
    _, cr, plan, sc, sw = _imports()
    policy = policy or ClearancePolicy()
    if phase not in PHASES:
        raise ValueError(f'unknown phase: {phase}')
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (11,) or b.shape != (11,) or not np.isfinite([a, b]).all():
        raise ValueError('11 finite model-radian waist+arm values required')
    waist = a[:4] if waist is None else np.asarray(waist, float)
    if waist.shape != (4,) or not np.isfinite(waist).all():
        raise ValueError('four finite fixed-waist angles required')
    if (Ta is None) != (Tb is None):
        raise ValueError('both Cartesian path endpoints are required')
    if Ta is not None:
        from scipy.spatial.transform import Rotation
        Ta, Tb = np.asarray(Ta, float), np.asarray(Tb, float)
        if Ta.shape != (4, 4) or Tb.shape != (4, 4) or not np.isfinite([Ta, Tb]).all():
            raise ValueError('finite 4x4 path endpoints required')
        delta = Tb[:3, 3] - Ta[:3, 3]
        delta2 = float(delta @ delta)
        rotation_delta = Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()
        if dev_mm is None:
            dev_mm = .5 if phase == 'insert' else 3.0
    intervals = max(policy.minimum_intervals, int(math.ceil(float(np.max(np.abs(b - a))) / math.radians(policy.sample_deg))))
    out = {'ok': True, 'phase': phase, 'fixture': fid, 'samples': intervals + 1,
           'min_clearance_mm': None, 'min_slack_mm': None, 'worst_pair': None,
           'min_joint_margin_deg': math.inf, 'max_wrist_ellipse': 0.,
           'max_elbow_rel_m': -math.inf, 'max_waist_change_deg': 0.,
           'max_path_deviation_mm': 0., 'max_orientation_error_deg': 0.,
           'max_orientation_step_deg': 0., 'failures': []}
    min_clear, min_slack = math.inf, math.inf
    prevR = None
    for i in range(intervals + 1):
        q = a + (i / intervals) * (b - a)
        r = ctx.robot
        r.set(sc.WAIST + sc.LEFT, q); r.fk()
        T = r.tcp_pose(ctx.tcp)
        fail = []
        waist_error = float(np.max(np.abs(q[:4] - waist)))
        out['max_waist_change_deg'] = max(out['max_waist_change_deg'], math.degrees(waist_error))
        if waist_error > 1e-10:
            fail.append('waist_moved')
        arm_margin = r.limit_margin(sc.LEFT)
        waist_margin = r.limit_margin(sc.WAIST)
        out['min_joint_margin_deg'] = min(out['min_joint_margin_deg'], math.degrees(arm_margin))
        if arm_margin < plan.MIN_LIMIT - 1e-10 or waist_margin < -1e-10:
            fail.append('joint_limit')
        wrist, elbow = sw.wrist_ellipse(q[4:]), sw.elbow_rel(r)
        out['max_wrist_ellipse'] = max(out['max_wrist_ellipse'], wrist)
        out['max_elbow_rel_m'] = max(out['max_elbow_rel_m'], elbow)
        if wrist > sw.WRIST_ELLIPSE_MAX + 1e-10:
            fail.append('coupled_wrist_limit')
        if elbow > ctx.cfg['elbow']['max_rel_m'] + 1e-10:
            fail.append('elbow_height')
        if phase == 'transit':
            violation = cr.adjust_violation(r, ctx.zones, ctx.geoms, ctx.tcp)
            if violation:
                fail.append('adjust_zone: ' + violation)
        if prevR is not None:
            out['max_orientation_step_deg'] = max(out['max_orientation_step_deg'], _angle_deg(prevR, T[:3, :3]))
        prevR = T[:3, :3].copy()
        if Ta is not None:
            u = float(np.clip(((T[:3, 3] - Ta[:3, 3]) @ delta) / delta2, 0, 1)) if delta2 > 1e-20 else i / intervals
            deviation = float(np.linalg.norm(T[:3, 3] - (Ta[:3, 3] + u * delta))) * 1000
            desiredR = Ta[:3, :3] @ Rotation.from_rotvec(u * rotation_delta).as_matrix()
            angle = _angle_deg(desiredR, T[:3, :3])
            out['max_path_deviation_mm'] = max(out['max_path_deviation_mm'], deviation)
            out['max_orientation_error_deg'] = max(out['max_orientation_error_deg'], angle)
            if deviation > dev_mm + 1e-6:
                fail.append('cartesian_path_deviation')
            if angle > angle_deg + 1e-8:
                fail.append('cartesian_orientation')
        indices, clears, thresholds, _, names = _contact_arrays(ctx, fid, phase, policy)
        if len(indices):
            slacks = clears - thresholds
            worst = int(np.argmin(slacks))
            min_clear = min(min_clear, float(np.min(clears)))
            if slacks[worst] < min_slack:
                min_slack = float(slacks[worst])
                c = r.d.contact[int(indices[worst])]
                out['worst_pair'] = [names[c.geom1], names[c.geom2]]
                out['worst_pair_clearance_mm'] = float(clears[worst]) * 1000
                out['worst_pair_required_mm'] = float(thresholds[worst]) * 1000
            if np.any(slacks < -policy.numerical_tolerance_m):
                fail.append('collision_clearance')
                if np.any(clears < -policy.numerical_tolerance_m):
                    fail.append('solid_penetration')
        warnings = _collision_warnings(r)
        if warnings:
            fail.append('mujoco_collision_coverage_invalid')
            out['mujoco_warnings'] = warnings
        if fail:
            out['ok'] = False
            if len(out['failures']) < 8:
                out['failures'].append({'sample': i, 'fraction': i / intervals, 'reasons': sorted(set(fail))})
    out['min_clearance_mm'] = None if math.isinf(min_clear) else min_clear * 1000
    out['min_slack_mm'] = None if math.isinf(min_slack) else min_slack * 1000
    return out


def validate_cycle(ctx, segments, fid, waist, policy=None):
    """Validate every interpolated segment and seam, with Cartesian completion.

    'ok' is sampled kinematic/collision validity under the supplied model. It is
    not electrical or physical commissioning acceptance. A closed cycle with a
    nominal final target visit is required, even if all individual edges pass.
    """
    policy = policy or ClearancePolicy()
    sc = _imports()[3]
    out = {'ok': True, 'fixture': fid, 'policy': asdict(policy), 'segments': [],
           'failures': [], 'samples': 0, 'physical_acceptance': False,
           'coverage': 'loaded-model collision masks, sampled joint interpolation'}
    previous, first = None, None
    reached = False
    for index, s in enumerate(segments):
        wps = np.asarray(s['waypoints'], float)
        if wps.ndim != 2 or wps.shape[1] != 11 or len(wps) < 2 or not np.isfinite(wps).all():
            raise ValueError('each segment needs at least two finite waist+arm waypoints')
        if first is None:
            first = wps[0]
        if previous is not None and np.max(np.abs(previous - wps[0])) > 1e-8:
            out['failures'].append(f'segment_{index}_discontinuous_seam')
        previous = wps[-1]
        phase = s.get('phase', 'insert' if s.get('contact') else ('transit' if s.get('role') == 'adjust' else 'approach'))
        linear = s.get('kind') == 'linear' or 'Ta' in s or 'Tb' in s
        if linear and ('Ta' not in s or 'Tb' not in s):
            raise ValueError('Cartesian segment must declare original Ta and Tb')
        row = {'name': s.get('name', str(index)), 'phase': phase, 'ok': True, 'edges': []}
        for a, b in zip(wps, wps[1:]):
            edge = check_segment(ctx, a, b, phase, fid, policy,
                                 Ta=s.get('Ta') if linear else None, Tb=s.get('Tb') if linear else None,
                                 dev_mm=s.get('dev_mm'), angle_deg=s.get('angle_deg', 1.), waist=waist)
            row['edges'].append(edge); row['ok'] &= edge['ok']; out['samples'] += edge['samples']
        if linear:
            for label, q, expected in [('start', wps[0], s['Ta']), ('end', wps[-1], s['Tb'])]:
                ctx.robot.set(sc.WAIST + sc.LEFT, q); ctx.robot.fk()
                T = ctx.robot.tcp_pose(ctx.tcp); expected = np.asarray(expected)
                dp = float(np.linalg.norm(T[:3, 3] - expected[:3, 3])) * 1000
                da = _angle_deg(T[:3, :3], expected[:3, :3])
                row[f'{label}_position_error_mm'] = dp; row[f'{label}_orientation_error_deg'] = da
                if dp > s.get('endpoint_mm', .5) or da > s.get('angle_deg', 1.):
                    row['ok'] = False
                    out['failures'].append(f'segment_{index}_{label}_target_missed')
        # A target visit must be an actual waypoint, not just an intention label.
        for q in wps:
            ctx.robot.set(sc.WAIST + sc.LEFT, q); ctx.robot.fk()
            T = ctx.robot.tcp_pose(ctx.tcp); final = target_pose(ctx, fid)
            reached |= bool(np.linalg.norm(T[:3, 3] - final[:3, 3]) <= .0005 and _angle_deg(T[:3, :3], final[:3, :3]) <= 1.)
        out['segments'].append(row)
        out['ok'] &= row['ok']
    if first is None:
        out['failures'].append('empty_cycle')
    elif np.max(np.abs(first - previous)) > 1e-8:
        out['failures'].append('cycle_did_not_return_to_start')
    if not reached:
        out['failures'].append('final_target_not_reached')
    out['ok'] &= not out['failures']
    out['nominal_target_reached'] = reached
    return out


def pair_contact_constraints(ctx, fid, phase, policy=None):
    """QP inequality generation from the same pair acceptance policy."""
    mujoco, _, _, sc, _ = _imports()
    policy = policy or ClearancePolicy()
    r, m, d = ctx.robot, ctx.robot.m, ctx.robot.d
    cols = [r.dadr[n] for n in sc.LEFT]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    rows, rhs = [], []
    for c, _, clear, threshold, _ in _contact_rows(ctx, fid, phase, policy, generation=True):
        normal = np.asarray(c.frame[:3])
        dv = np.zeros(m.nv)
        for gid, sign in ((c.geom1, -1.), (c.geom2, 1.)):
            body = int(m.geom_bodyid[gid])
            if body == 0:
                continue
            point = np.asarray(c.pos) + sign * normal * c.dist / 2
            mujoco.mj_jac(m, d, jp, jr, point, body)
            dv += sign * (normal @ jp)
        row = -dv[cols]
        if np.any(row):
            rows.append(row); rhs.append(.5 * (clear - threshold))
    warnings = _collision_warnings(r)
    if warnings:
        raise RuntimeError(f'MuJoCo collision coverage invalid; rebuild with sufficient arena: {warnings}')
    return (np.asarray(rows), np.asarray(rhs)) if rows else (None, None)


@contextmanager
def temporary_offline_policy(ctx, fid, policy=None):
    """Temporary, single-process adapters for the existing offline generator.

    Does not write sim sources or installed configuration.  This adapter is not
    thread safe (the existing generator itself has module-global configuration).
    Always run validate_cycle on the returned actual waypoints as the final gate.
    """
    _, cr, plan, _, sw = _imports()
    policy = policy or ClearancePolicy()
    original = cr.check_line, cr._chunk_ok, plan.contact_constraints, sw.contact_constraints
    def line(context, a, b, adjust=True, allow_start=False, contact_motor=None):
        phase = 'insert' if contact_motor else ('transit' if adjust else 'corridor')
        result = check_segment(context, a, b, phase, contact_motor or fid, policy)
        if not result['ok']:
            raise sw.PathError(f'pair policy rejected: {result}')
        return [result]
    def constraints(robot, group, **kwargs):
        if robot is not ctx.robot or tuple(group) != tuple(_imports()[3].LEFT):
            return original[2](robot, group, **kwargs)
        phase = 'insert' if getattr(robot, 'active_motor', None) else 'transit'
        return pair_contact_constraints(ctx, fid, phase, policy)
    def chunk(context, a, b, motor, floor, allow_start=False):
        # Never inherit the legacy global floor (minimum clearance minus 1 mm).
        # Every pair retains its own phase-specific acceptance threshold.
        phase = 'insert' if motor else 'corridor'
        return check_segment(context, a, b, phase, motor or fid, policy)['ok']
    cr.check_line, cr._chunk_ok, plan.contact_constraints, sw.contact_constraints = line, chunk, constraints, constraints
    try:
        yield
    finally:
        cr.check_line, cr._chunk_ok, plan.contact_constraints, sw.contact_constraints = original
