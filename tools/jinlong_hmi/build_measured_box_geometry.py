#!/usr/bin/env python3
"""Build offline, model-specific box collision meshes from archived CAD.

The output is a *CAD candidate*, not an assertion that the 160 STEP matches the
scanned part.  Its manifest records that discrepancy.  Run the builder with
cadquery-ocp, trimesh, numpy and coacd installed; the planner-side two functions
need only numpy and mujoco.  No network, camera or robot commands are issued.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/diagnostics/full-solve-20261023/geometry'
SOURCES = {
    'JM160': (ROOT / 'docs/reference/jinlong_terminal_boxes/cad/JL160MB070.step', 1),
    'JL100': (ROOT / 'docs/reference/jinlong_terminal_boxes/cad/JL100B3.step', 1),
    'JL132': (Path('/home/wjr/code/r435/台州项目/建模/JL-132M-B3.step'), 2),
}


def box_from_source(fid):
    """CAD mm -> box mm. Origin cover-hole centre on rim, +Z into cavity.

    The negative Y side is identified by the unique centreline ground hole,
    not by a renderer's camera orientation.  160 has a 0.09368 degree rotation
    already present in the STEP's cover-hole array.
    """
    if fid == 'JL100':
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        origin = np.array([0, 144, 153.5])
    elif fid == 'JL132':
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
        origin = np.array([180.5, 0, 184])
    elif fid == 'JM160':
        s, c = 0.00163503279390698, 0.999998663332988
        R = np.array([[-s, 0, c], [-c, 0, -s], [0, -1, 0]])
        origin = np.array([0.0898532869825, 227.11782421961, 55.047488510225])
    else:
        raise ValueError(fid)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ origin
    return T


def extract(fid, path, solid_index):
    import trimesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRepCheck import BRepCheck_Analyzer

    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids = []
    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    solid = solids[solid_index]
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    BRepMesh_IncrementalMesh(solid, 0.10, False, 0.15, True).Perform()
    vertices, faces = [], []
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation:
            offset = len(vertices)
            for i in range(1, triangulation.NbNodes() + 1):
                p = triangulation.Node(i).Transformed(location.Transformation())
                vertices.append([p.X(), p.Y(), p.Z()])
            for i in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(i).Get()
                order = [a, c, b] if face.Orientation() == TopAbs_REVERSED else [a, b, c]
                faces.append([offset + k - 1 for k in order])
        explorer.Next()
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.apply_transform(box_from_source(fid))
    return mesh, valid


def build(fid, output):
    import coacd
    path, solid_index = SOURCES[fid]
    mesh, valid = extract(fid, path, solid_index)
    mesh.export(output / (fid + '_box_mm.stl'))
    # Concavity tolerance is 0.5 mm, with real_metric expressed in input units.
    # The manifest and sampling checks preserve this approximation explicitly.
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=0.5, real_metric=True, max_convex_hull=-1,
        preprocess_mode='auto', preprocess_resolution=80,
        resolution=2000, mcts_nodes=12, mcts_iterations=40,
        mcts_max_depth=3, seed=20261023, merge=True,
    )
    doc = {
        'model': fid, 'status': 'offline_CAD_candidate_not_site_acceptance',
        'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'solid_index': solid_index, 'occt_solid_valid': valid,
        'source_mesh_watertight': bool(mesh.is_watertight),
        'box_from_source_mm': box_from_source(fid).tolist(),
        'bounds_box_mm': mesh.bounds.tolist(),
        'units': 'metres', 'decomposition_requested_threshold_mm': 0.5,
        'decomposition_hull_limit': None,
        'frame': 'cover-hole-array centre on top rim; negative Y ground side; positive Z into box',
        'parts': [{'vertices': (np.asarray(v) / 1000).tolist(),
                   'faces': np.asarray(f).tolist()} for v, f in parts],
        'limitations': [
            'Box base only: terminal board, wires and screw heads are separate geometry.',
            'Convex decomposition approximates CAD; not a guaranteed exact surface.',
            'No sensor-to-CAD registration or hardware motion is implied.',
        ],
    }
    if fid == 'JM160':
        doc['limitations'].append(
            '160 CAD differs from manufacturer PDF (152 square vs 152x162, internal holes 78 vs102); '
            'CAD ground-hole axis xy is (0,-57.907)mm; its axial reference point is not a measured cap surface. '
            'Tool ground is (-29.034,-65.014)mm '
            'including saved -2mm X shift, and scanned cap depth is28.45mm. '
            'Do not label this geometry as the scanned 160 part.')
    (output / (fid + '_collision.json')).write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({k: doc[k] for k in ('model', 'occt_solid_valid', 'source_mesh_watertight', 'bounds_box_mm')} |
                     {'parts': len(parts)}), flush=True)
    return doc


def modify_scene(scene, fid, geometry_dir=DEFAULT_OUTPUT):
    """Remove the old generic four-wall model and pin CAD identity to config."""
    path = Path(geometry_dir) / (fid + '_collision.json')
    if not path.exists():
        raise FileNotFoundError(path)
    scene['obstacles'] = [o for o in scene['obstacles'] if not o['name'].startswith('box_')]
    scene['model_specific_box_collision'] = str(path.resolve())
    if json.loads(path.read_text()).get('measured_terminals'):
        scene['terminals'] = False
    return scene


def audit_decomposition(fid, output):
    """Reproducibly compare solid membership, including the open cavity axis."""
    import trimesh
    from scipy.spatial import ConvexHull

    doc = json.loads((output / (fid + '_collision.json')).read_text())
    mesh = trimesh.load(output / (fid + '_box_mm.stl'))
    points = np.random.default_rng(20261023).uniform(mesh.bounds[0], mesh.bounds[1], (6000, 3))
    points = np.vstack([points, [[0, 0, z] for z in range(1, 41)]])
    cad = mesh.contains(points)
    approximate = np.zeros(len(points), bool)
    for part in doc['parts']:
        vertices = np.asarray(part['vertices']) * 1000
        selected = np.where(np.all((points >= vertices.min(0) - 1e-6) &
                                   (points <= vertices.max(0) + 1e-6), axis=1))[0]
        if len(selected):
            planes = ConvexHull(vertices).equations
            approximate[selected] |= np.all(points[selected] @ planes[:, :3].T + planes[:, 3] <= 1e-6,
                                             axis=1)
    different = points[cad != approximate]
    distances = trimesh.proximity.closest_point(mesh, different)[1] if len(different) else np.array([])
    result = {
        'model': fid, 'convex_parts': len(doc['parts']), 'random_sample_count': 6000,
        'central_axis_points': 40, 'cad_axis_inside': cad[-40:].tolist(),
        'hulls_axis_inside': approximate[-40:].tolist(),
        'sample_false_positive_count': int(np.sum(approximate & ~cad)),
        'sample_false_negative_count': int(np.sum(cad & ~approximate)),
        'sample_disagreement_distance_to_CAD_mm_max': float(distances.max()) if len(distances) else 0.0,
        'interpretation': 'Finite random sample audit, not a global approximation bound or site validation.',
    }
    (output / (fid + '_decomposition_audit.json')).write_text(json.dumps(result, indent=2) + '\n')
    return result


def modify_spec(spec, scene, fid):
    """Attach separate convex parts to MuJoCo; never convexify the whole box."""
    import mujoco
    doc = json.loads(Path(scene['model_specific_box_collision']).read_text())
    if doc['model'] != fid:
        raise ValueError('Wrong box model bound to scene')
    # Hundreds of fine convex parts can report >10k near contacts.  The default
    # auto arena drops contacts at actual insertion poses; allocate explicitly.
    spec.memory = max(int(spec.memory), 64 * 1024 * 1024)
    T = np.asarray(scene['box_T_base_link'], float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    body = spec.worldbody.add_body(name='box_' + fid + '_cad', pos=T[:3, 3], quat=quat)
    for i, part in enumerate(doc.get('parts', [])):
        name = f'box_{fid}_cad_{i:03d}'
        spec.add_mesh(name=name, uservert=np.asarray(part['vertices']).flatten(),
                      userface=np.asarray(part['faces'], int).flatten())
        g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)
        g.contype, g.conaffinity = 2, 1 | 4
        # Match the simulator's 30 mm contact-reporting horizon; this is not
        # an inflated solid and the validator still compares signed distance.
        g.margin = 0.03
        g.rgba = [0.48, 0.51, 0.55, 1]
    for i, box in enumerate(doc.get('boxes', [])):
        g = body.add_geom(name=f'box_{fid}_scan_{i:04d}', type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=box['center'], size=box['half_size'])
        g.contype, g.conaffinity, g.margin = 2, 1 | 4, 0.03
        g.rgba = [0.65, 0.25, 0.3, 1] if box['unknown'] else [0.45, 0.50, 0.55, 1]
    for terminal in doc.get('measured_terminals', []):
        g = body.add_geom(name=terminal['name'], type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        g.fromto = terminal['a'] + terminal['b']
        g.size = [terminal['radius'], 0, 0]
        g.contype, g.conaffinity, g.margin = 2, 1 | 4, 0.03
        g.rgba = [0.78, 0.74, 0.55, 1]
    if 'tool_cad_collision' in scene:
        from build_tool_geometry import modify_spec as tool_modify_spec
        spec = tool_modify_spec(spec, scene)
    return spec


def build_model(scene, fid, original_build):
    """Use the original builder, adding CAD immediately before compilation.

    The cached simulator returns a compiled MjModel.  Clone its Python function
    and replace its single ``spec.compile()`` expression with this local hook;
    its module, on-disk source and global MuJoCo API remain untouched.  Refuse
    a changed builder structure instead of silently omitting box geometry.
    """
    tree = ast.parse(inspect.getsource(original_build))
    count = 0

    class AttachBeforeCompile(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal count
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'spec' and node.func.attr == 'compile'):
                count += 1
                return ast.copy_location(ast.Call(
                    func=ast.Name(id='_compile_box_spec', ctx=ast.Load()),
                    args=[ast.Name(id='spec', ctx=ast.Load())], keywords=[]), node)
            return node

    tree = AttachBeforeCompile().visit(tree)
    if count != 1:
        raise ValueError(f'Expected one spec.compile() in simulator builder, got {count}')
    ast.fix_missing_locations(tree)
    namespace = dict(original_build.__globals__)
    namespace['_compile_box_spec'] = lambda spec: modify_spec(spec, scene, fid).compile()
    exec(compile(tree, '<offline-box-geometry-builder>', 'exec'), namespace)
    return namespace[original_build.__name__](scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--models', nargs='+', choices=list(SOURCES), default=['JL100', 'JL132'],
                        help='JM160 must be requested explicitly because its CAD version is unresolved')
    parser.add_argument('--audit-only', action='store_true', help='Audit existing mesh outputs without rebuilding')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for fid in args.models:
        if not args.audit_only:
            build(fid, args.output)
        print(json.dumps(audit_decomposition(fid, args.output)), flush=True)


if __name__ == '__main__':
    main()
