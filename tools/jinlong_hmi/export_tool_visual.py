#!/usr/bin/env python3
"""Tessellate the verified tool STEP for display at the actual flange mounting.

Run in the OCP geometry environment. The resulting mesh is visual-only; the
validated plate-sector and boss collision geometry is not changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rz(degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = json.loads(a.result.read_text())
    if not result.get('full_common_home_cycles_verified'):
        raise ValueError('Expected a validated shared-station result')
    tool_paths = {path: value for path, value in result['geometry_sha256_at_build'].items()
                  if Path(path).name == 'tool_collision.json'}
    if len(tool_paths) != 1:
        raise ValueError('Expected one pinned tool-CAD collision source')
    metadata_path, expected = next(iter(tool_paths.items()))
    if sha(metadata_path) != expected:
        raise ValueError('Tool collision source changed')
    tool = json.loads(Path(metadata_path).read_text())
    source = Path(tool['source'])
    if sha(source) != tool['source_sha256']:
        raise ValueError('Tool STEP differs from the validated boss-CAD source')
    bundle_path = Path(result['run_arguments']['bundle']).resolve()
    if sha(bundle_path) != result['sources_sha256_at_load'][str(bundle_path)]:
        raise ValueError('Actual flange mounting source changed')
    bundle = json.loads(bundle_path.read_text())
    mount = np.asarray(bundle['site']['flange_T_tcp'], dtype=float)
    # make_tool_mesh.py: old flange = Rz(90) Rx(180) (STEP-[0,0,6mm]).
    # That old flange has TCP yaw -120. Thus TCP = Rz(-150) Rx(180),
    # exactly matching build_tool_geometry.modify_spec's independent CAD path.
    step_to_tcp = rz(-150) @ np.diag([1., -1., -1.])
    transform = mount.copy()
    transform[:3, :3] = mount[:3, :3] @ step_to_tcp
    transform[:3, 3] = mount[:3, 3] - transform[:3, :3] @ [0., 0., .006]
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12):
        raise ValueError('Tool visual transform must be rigid')
    sim = Path(result['run_arguments']['sim_root'])
    legacy_builder = sim/'sim/helios_insert/make_tool_mesh.py'
    boss_builder = Path(result['run_arguments']['tool_geometry_module']).resolve()
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    import trimesh
    reader = STEPControl_Reader()
    reader.ReadFile(str(source)); reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, .10, False, .15, True).Perform()
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    vertices, triangles = [], []
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri:
            offset = len(vertices)
            for i in range(1, tri.NbNodes()+1):
                v = tri.Node(i).Transformed(loc.Transformation())
                vertices.append([v.X(), v.Y(), v.Z()])
            for i in range(1, tri.NbTriangles()+1):
                x, y, z = tri.Triangle(i).Get()
                order = [x, z, y] if face.Orientation() == TopAbs_REVERSED else [x, y, z]
                triangles.append([offset+k-1 for k in order])
        faces.Next()
    mesh = trimesh.Trimesh(vertices, triangles, process=True)
    mesh.apply_scale(.001)
    mesh.apply_transform(transform)
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    output = a.output/'tool_actual_mount_visual.stl'
    mesh.export(output)
    manifest = {
        'path': str(output), 'sha256': sha(output), 'units': 'metres', 'frame': 'flange',
        'visual_only': True, 'triangles': len(mesh.faces), 'source': str(source),
        'source_sha256': sha(source), 'flange_T_tcp': mount.tolist(),
        'flange_T_source_m': transform.tolist(),
        'flange_T_source_matrix_sha256': hashlib.sha256(transform.astype('<f8').tobytes()).hexdigest(),
        'legacy_visual_and_hull_builder': str(legacy_builder), 'legacy_builder_sha256': sha(legacy_builder),
        'independent_boss_cad_builder': str(boss_builder), 'boss_builder_sha256': sha(boss_builder),
        'legacy_flange_T_tcp_yaw_deg': -120., 'legacy_source_clock_deg': 90.,
        'actual_source_clock_deg': float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))),
        'registration': 'Same STEP and source-to-TCP transform as validated boss CAD; legacy visual/hulls share one source transform.',
        'hidden_collision_display_geoms': [p['name'] for p in tool['parts']],
        'collision_geometry_changed': False,
        'exporter_sha256': sha(__file__),
    }
    (a.output/'tool_visual.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'path': str(output), 'triangles': len(mesh.faces),
                      'source_clock_deg': manifest['actual_source_clock_deg']}), flush=True)


if __name__ == '__main__':
    main()
