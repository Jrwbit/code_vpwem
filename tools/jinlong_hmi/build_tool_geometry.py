#!/usr/bin/env python3
"""Replace five approximate tool cylinders with the archived STEP's exterior.

Only offline MjSpec objects are modified. The original plate-sector hulls remain;
real STEP bosses overlap their shallow root instead of leaving a missing strip.
Their internal mounting bores are conservatively filled by each convex envelope.
Socket and spring-pin lengths remain the original photo estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path('/home/wjr/code/jinlong-motor-arm-control/docs/tooling/flange_v2.STEP')
DEFAULT_OUTPUT = ROOT / 'outputs/diagnostics/full-solve-20261023/tool_geometry'
CENTRES_STEP_MM = {
    'tool_boss_JL100': [-57.8, -78.4],
    'tool_boss_JL132': [-45.8, 87.5],
    'tool_boss_JM160': [93.05, -128.8],
    'tool_spacer_1': [74.7, -14.6],
    'tool_spacer_2': [103.7, -6.8],
}


def _mesh(shape):
    import trimesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(shape, .01, False, .10, True).Perform()
    vertices, faces = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current()); location = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, location)
        if tri:
            offset = len(vertices)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(location.Transformation())
                vertices.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                order = (a, c, b) if face.Orientation() == TopAbs_REVERSED else (a, b, c)
                faces.append([offset + k - 1 for k in order])
        explorer.Next()
    return trimesh.Trimesh(vertices, faces, process=True)


def build(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    import trimesh
    from scipy.spatial import ConvexHull
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.gp import gp_Pnt
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepCheck import BRepCheck_Analyzer

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(source))
    if int(status) != 1:
        raise RuntimeError(f'STEP read failed: {status}')
    reader.TransferRoots(); source_shape = reader.OneShape()
    # Isolate solid protrusions on the fixture side. The central shallow hub is
    # already represented by the plate mesh and is selected out by its centre.
    cutter = BRepPrimAPI_MakeBox(gp_Pnt(-300, -300, -100), gp_Pnt(300, 300, -.001)).Shape()
    common = BRepAlgoAPI_Common(source_shape, cutter)
    common.Build()
    if not common.IsDone():
        raise RuntimeError('STEP boss extraction failed')
    explorer = TopExp_Explorer(common.Shape(), TopAbs_SOLID)
    parts, excluded = {}, []
    while explorer.More():
        solid = explorer.Current(); explorer.Next()
        prop = GProp_GProps(); BRepGProp.VolumeProperties_s(solid, prop)
        p = prop.CentreOfMass(); centre = np.array([p.X(), p.Y(), p.Z()])
        if np.linalg.norm(centre[:2]) < 60:
            excluded.append({'reason': 'central hub retained in existing plate hulls', 'centroid_step_mm': centre.tolist()})
            continue
        name, distance = min(((n, float(np.linalg.norm(centre[:2] - xy))) for n, xy in CENTRES_STEP_MM.items()), key=lambda item: item[1])
        if distance > 1 or name in parts:
            raise RuntimeError(f'Unrecognised or duplicated protrusion: {centre}, {name}, distance {distance}')
        mesh = _mesh(solid)
        hull = mesh.convex_hull
        name_file = name + '_step_mm'
        mesh.export(output / (name_file + '.stl'))
        hull.export(output / (name_file + '_hull.stl'))
        parts[name] = {
            'replaces': name, 'name': name + '_cad',
            'vertices_step_mm': np.asarray(hull.vertices).tolist(),
            'faces': np.asarray(hull.faces).tolist(),
            'bounds_step_mm': mesh.bounds.tolist(),
            'centroid_step_mm': centre.tolist(),
            'occt_valid': bool(BRepCheck_Analyzer(solid).IsValid()),
            'source_solid_volume_mm3': prop.Mass(),
            'convex_envelope_volume_mm3': float(hull.volume),
            'source_mesh_watertight': bool(mesh.is_watertight),
            'envelope_minus_solid_mm3': float(hull.volume - prop.Mass()),
        }
        if not parts[name]['occt_valid'] or not mesh.is_watertight or not hull.is_watertight:
            raise RuntimeError(f'Invalid protrusion mesh: {name}')
        # Convexification must enclose all sampled CAD surface vertices. It can
        # only fill concavities/holes, never carve material away for clearance.
        equations = ConvexHull(hull.vertices).equations
        max_outside = float(np.max(mesh.vertices @ equations[:, :3].T + equations[:, 3]))
        parts[name]['max_tessellated_surface_outside_envelope_mm'] = max_outside
        if max_outside > 1e-6:
            raise RuntimeError(f'Convex envelope excludes CAD vertices: {name}')
    if set(parts) != set(CENTRES_STEP_MM):
        raise RuntimeError(f'Expected five bosses/spacers, got {list(parts)}')
    doc = {
        'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'status': 'offline_STEP_exterior_candidate',
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'units': 'STEP millimetres; transformed to flange metres when compiling',
        'fixture_side_cut_z_max_mm': -.001,
        'tessellation_linear_deflection_mm': .01,
        'tessellation_angular_deflection_rad': .10,
        'parts': list(parts.values()), 'excluded': excluded,
        'plate_hulls': 'Retain rotated original sector hulls; they omit bosses below STEP z=-3.6mm.',
        'unchanged_estimates': {'socket_length_mm': 32., 'socket_radius_mm': 5.,
                                'ground_pin_length_mm': {'JL100': 12., 'JL132': 20., 'JM160': 22.},
                                'ground_pin_radius_mm': 2.5},
        'limitations': [
            'Internal mounting bores/slots are filled by conservative separate convex envelopes.',
            'Tessellation approximates curved CAD surfaces; it is not a measured production-part tolerance.',
            'Boss roots overlap existing plate hulls to preserve the entire shallow region.',
            'Socket lengths and ground-pin lengths remain photo-derived estimates, not independently measured.',
            'Source tool STEP is vendor-provided; as-built changes and cables are not established here.',
        ],
    }
    path = output / 'tool_collision.json'
    path.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'output': str(path), 'parts': [{k: p[k] for k in ('name', 'bounds_step_mm', 'source_solid_volume_mm3', 'convex_envelope_volume_mm3')} for p in doc['parts']]}), flush=True)
    return doc


def modify_scene(scene, flange_T_tcp, collision=DEFAULT_OUTPUT / 'tool_collision.json'):
    path = Path(collision)
    if not path.exists():
        raise FileNotFoundError(path)
    T = np.asarray(flange_T_tcp, float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('A finite homogeneous flange_T_tcp is required')
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(T[:3, :3]), 1.):
        raise ValueError('flange_T_tcp rotation must be proper and orthonormal')
    scene['tool_cad_collision'] = str(path.resolve())
    scene['tool_cad_flange_T_tcp'] = T.tolist()
    return scene


def modify_spec(spec, scene):
    import mujoco
    if 'tool_cad_collision' not in scene:
        return spec
    doc = json.loads(Path(scene['tool_cad_collision']).read_text())
    by_name = {g.name: g for g in spec.geoms}
    expected = set(CENTRES_STEP_MM)
    if not expected <= set(by_name):
        raise ValueError(f'Missing original tool primitives: {expected - set(by_name)}')
    body = next(b for b in spec.bodies if b.name == 'AR5-5_07L_tcp')
    mount = np.asarray(scene['tool_cad_flange_T_tcp'], float)
    angle = np.radians(-150.)
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tcp_from_step = Rz @ np.diag([1., -1., -1.])
    for part in doc['parts']:
        spec.delete(by_name[part['replaces']])
        vertices = np.asarray(part['vertices_step_mm'], float)
        tcp_m = (vertices - [0, 0, 6.]) @ tcp_from_step.T / 1000.
        flange_m = tcp_m @ mount[:3, :3].T + mount[:3, 3]
        name = part['name']
        spec.add_mesh(name=name, uservert=flange_m.flatten(), userface=np.asarray(part['faces'], int).flatten())
        g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
        g.contype, g.conaffinity = 4, 0
        g.margin = .06  # original TOOL_MARGIN reporting horizon, not inflated solid
        g.rgba = [.80, .68, .38, 1]
    return apply_fixture_revision(spec, scene)




def apply_fixture_revision(spec, scene):
    """Apply an explicit proposed fixture revision; never mutate source STEP.

    Vectors use each fixture's saved nominal box frame, before target correction.
    Extending a cylinder keeps its mount end fixed and moves only its distal end.
    This is collision geometry for a design candidate, not a manufacturing model.
    """
    revision = scene.get('tool_fixture_revision')
    if not revision:
        return spec
    if revision.get('design_only') is not True:
        raise ValueError('A tooling revision must be explicitly marked design_only')
    geoms = {g.name: g for g in spec.geoms}
    references = scene['tool_fixture_reference_box_T_flange']
    for fid, change in revision['fixtures'].items():
        R = np.asarray(references[fid], float)[:3, :3].T
        for key in ('socket_extension_mm', 'ground_pin_extension_mm'):
            value = float(change.get(key, 0.))
            if not np.isfinite(value) or value < 0:
                raise ValueError('Only finite nonnegative tooling extensions are supported')
        for k in (1, 2, 3):
            g = geoms[f'tool_sock_{fid}_{k}']
            endpoints = np.asarray(g.fromto, float).reshape(2, 3).copy()
            u = endpoints[1]-endpoints[0]; u /= np.linalg.norm(u)
            endpoints[1] += u * change.get('socket_extension_mm', 0.) / 1000
            g.fromto = endpoints.ravel()
        pin = geoms[f'tool_pin_{fid}']
        endpoints = np.asarray(pin.fromto, float).reshape(2, 3).copy()
        u = endpoints[1]-endpoints[0]; u /= np.linalg.norm(u)
        endpoints += R @ np.asarray(change.get('ground_pin_shift_box_mm', [0., 0., 0.])) / 1000
        endpoints[1] += u * change.get('ground_pin_extension_mm', 0.) / 1000
        pin.fromto = endpoints.ravel()
        boss = geoms[f'tool_boss_{fid}_cad']
        boss.pos = np.asarray(boss.pos) + R @ np.asarray(change.get('ground_boss_shift_box_mm', [0., 0., 0.])) / 1000
    return spec


def plot_comparison(doc, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), constrained_layout=True)
    t = np.linspace(0, 2 * np.pi, 300)
    for ax, fid in zip(axes, ('JL100', 'JL132', 'JM160')):
        name = 'tool_boss_' + fid
        part = next(p for p in doc['parts'] if p['replaces'] == name)
        p = np.asarray(part['vertices_step_mm'])[:, :2] - CENTRES_STEP_MM[name]
        h = ConvexHull(p); border = p[h.vertices]; border = np.vstack([border, border[0]])
        ax.fill(border[:, 0], border[:, 1], color='#95bbde', alpha=.65, label='STEP convex envelope')
        ax.plot(border[:, 0], border[:, 1], color='#185c94', lw=2)
        ax.plot(12 * np.cos(t), 12 * np.sin(t), '--', color='#b63c37', lw=1.6, label='Previous radius 12 mm')
        ax.set(aspect='equal', xlim=(-16, 16), ylim=(-16, 16), xlabel='STEP X from old centre (mm)', ylabel='STEP Y (mm)', title=fid)
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Ground bosses: actual CAD exterior is direction-dependent', fontsize=14)
    fig.savefig(Path(output) / 'boss_proxy_comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.plot:
        # Load Python's plotting/XML libraries before OCCT's shared libraries.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot
    doc = build(args.source, args.output)
    if args.plot:
        plot_comparison(doc, args.output)


if __name__ == '__main__':
    main()
