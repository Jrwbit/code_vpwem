#!/usr/bin/env python3
"""Inspect complete archived B3 motor STEP assemblies for a shared station.

Read-only CAD extraction. Station placement must be derived from the actual
mounting plane/holes and shaft direction, never from independent box heights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCES = {
    'JL100': Path('/home/wjr/code/r435/台州项目/建模/JL-100-B3.step'),
    'JL132': Path('/home/wjr/code/r435/台州项目/建模/JL-132M-B3.step'),
    'JM160': Path('/home/wjr/code/r435/台州项目/建模/JL-160M-B3.stp'),
}

# Zero-based TopExp solid indices, verified by analytic faces in the full STEP.
# Slots have two semicircle axes; their midpoint, not either semicircle, is the
# mounting-hole centre. The four slot centres define the common loading datum.
DATUMS = {
    'JL100': dict(feet=[1, 7], up_axis=1, slot_radius=6., shaft=12,
                  box_solids=list(range(14, 24)) + [59], accessories=[22, 23, 59],
                  drawing_length_mm=383., drawing_foot_pitch_mm=[140., 160.]),
    'JL132': dict(feet=[32, 37], up_axis=0, slot_radius=6., shaft=38,
                  box_solids=list(range(2, 12)) + [61], accessories=[5, 6, 61],
                  drawing_length_mm=513., drawing_foot_pitch_mm=[178., 216.]),
    'JM160': dict(feet=[43, 50], up_axis=1, slot_radius=7.5, shaft=38,
                  box_solids=[0, 2, 3, 4, 31, 32, 33, 34, 35, 36, 39, 40],
                  accessories=[35, 36, 39, 40],
                  drawing_length_mm=608., drawing_foot_pitch_mm=[210., 254.]),
}


def transformed_bounds(bounds_mm, transform_mm):
    """Conservative transformed AABB, including all eight source corners."""
    b = np.asarray(bounds_mm)
    points = np.array([[b[i, 0], b[j, 1], b[k, 2]]
                       for i in range(2) for j in range(2) for k in range(2)])
    points = points @ transform_mm[:3, :3].T + transform_mm[:3, 3]
    return np.array([points.min(0), points.max(0)])


def envelope(bounds_mm):
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(bounds_mm).mean(0) / 1000.
    return {'T': transform.tolist(), 'size': (np.diff(bounds_mm, axis=0)[0] / 1000.).tolist()}


def derive_station(features):
    """Derive placement exclusively from mounting feet, never from TCP/IK."""
    from build_measured_box_geometry import box_from_source

    result = {
        'schema_version': 1, 'units': 'metres',
        'intrinsic_geometry_confirmed': True,
        'confirmation_scope': 'Archived full B3 STEP datum extraction only; not physical-part metrology.',
        'station_frame': ('Origin is the four foot-slot centre rectangle centre projected onto the foot contact plane. '
                          '+Z points up, +X follows the output shaft away from the robot, +Y completes a right-handed frame. '
                          'A single fixed locating fixture must centre the three different foot patterns at this same datum.'),
        'station_T_boxes': {}, 'motor_envelopes': {}, 'motor_aux_envelopes': {}, 'model_datums': {},
        'limitations': [
            'No per-model station translation, rotation, riser or independently moved motor body is permitted.',
            'The common datum is a fixture design convention. The existing physical fixture has not been surveyed.',
            'Box frames reuse the independently extracted cover-hole-centre frames, not tool-derived terminal positions.',
            'JM160 archived box CAD version differs from the actual scan and manufacturer box drawing; '
            'its full-motor-to-box installation transform remains a CAD-based assumption when using the scanned cavity.',
            'B3 drawing overall lengths exceed the archived JL100/JL132 STEP by about 9/10 mm; '
            'the motor collision envelope includes this difference at the fan end, but the visual CAD mesh does not.',
            'Closed lids, lid seals and lid fasteners are removed for inspection; external cable fittings are separate auxiliary obstacles.',
        ],
    }
    for fid, data in features.items():
        cfg = DATUMS[fid]
        up_axis = cfg['up_axis']
        feet = [data['solids'][i] for i in cfg['feet']]
        floor = min(s['bounds_source_mm'][0][up_axis] for s in feet)
        contact_faces, axes = [], []
        for s in feet:
            for p in s['planes']:
                if (abs(p['normal_source'][up_axis]) > .99999 and
                        abs(p['plane_point_source_mm'][up_axis] - floor) < 1e-4):
                    contact_faces.append({'solid': s['index'], 'face': p['face_index'],
                                          'point_source_mm': p['plane_point_source_mm'], 'area_mm2': p['area_mm2']})
            for c in s['cylinders']:
                if (abs(c['axis_direction_source'][up_axis]) > .99999 and
                        abs(c['radius_mm'] - cfg['slot_radius']) < 1e-5 and
                        abs(c['bounds_source_mm'][0][up_axis] - floor) < 1e-4):
                    axes.append(c['axis_point_source_mm'])
        axes = np.unique(np.round(axes, 9), axis=0)
        if len(axes) != 8 or len(contact_faces) != 4:
            raise ValueError(f'{fid}: expected eight slot arc axes/four foot contact faces, got {len(axes)}/{len(contact_faces)}')
        origin = axes.mean(0)
        # Remove the harmless OCCT bounding tolerance from the actual plane.
        origin[up_axis] = np.mean([p['point_source_mm'][up_axis] for p in contact_faces])
        up = np.eye(3)[up_axis]
        longitudinal = np.array([0., 0., 1.])
        rotation = np.array([longitudinal, np.cross(up, longitudinal), up])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ origin
        source_T_box = np.linalg.inv(box_from_source(fid))
        station_T_box = transform @ source_T_box
        station_T_box[:3, 3] /= 1000.
        result['station_T_boxes'][fid] = station_T_box.tolist()
        # Each left/right foot has two slots; adjacent semicircle axes average
        # to their centre. Group by original longitudinal coordinate and side.
        holes = []
        transverse_axis = 0 if up_axis == 1 else 1
        for axial in sorted(np.unique(np.round(axes[:, 2], 6))):
            for side in [-1, 1]:
                group = axes[(np.abs(axes[:, 2] - axial) < 1e-5) &
                             (np.sign(axes[:, transverse_axis]) == side)]
                if len(group) != 2:
                    raise ValueError(f'{fid}: inconsistent mounting-slot pairing')
                centre = group.mean(0)
                centre[up_axis] = origin[up_axis]
                holes.append((rotation @ (centre - origin)) / 1000.)
        holes = np.asarray(holes)
        pitch = np.ptp(holes[:, :2], axis=0) * 1000.
        if not np.allclose(pitch, cfg['drawing_foot_pitch_mm'], atol=1e-5):
            raise ValueError(f'{fid}: foot pitch disagrees with B3 manufacturer drawing')
        shaft = data['solids'][cfg['shaft']]
        shaft_cylinders = [c for c in shaft['cylinders'] if abs(c['axis_direction_source'][2]) > .99999]
        cylinder = max(shaft_cylinders, key=lambda c: c['area_mm2'])
        shaft_point = np.array(cylinder['axis_point_source_mm'])
        shaft_point[2] = origin[2]
        shaft_station = rotation @ (shaft_point - origin) / 1000.
        body_ids = [s['index'] for s in data['solids'] if s['index'] not in cfg['box_solids']]
        body_bounds = np.array([transformed_bounds(data['solids'][i]['bounds_source_mm'], transform) for i in body_ids])
        bb = np.array([body_bounds[:, 0].min(0), body_bounds[:, 1].max(0)])
        cad_bb = bb.copy()
        # The shaft tip and shaft-to-front-foot distance agree with the PDFs.
        # Put any total-length discrepancy at the fan end conservatively.
        length_padding = max(0., cfg['drawing_length_mm'] - (bb[1, 0] - bb[0, 0]))
        bb[0, 0] -= length_padding
        bb[0, 2] = 0.  # analytic foot face, not the -1e-7 mm BRep tolerance
        result['motor_envelopes'][fid] = envelope(bb)
        result['motor_envelopes'][fid].update(cad_bounds_station_mm=cad_bb.tolist(),
                                            drawing_fan_end_extension_mm=length_padding,
                                            source_solid_indices=body_ids)
        result['motor_aux_envelopes'][fid] = [
            {'name': f'motor_fitting_{fid}_{i:03d}', 'source_solid_index': i,
             **envelope(transformed_bounds(data['solids'][i]['bounds_source_mm'], transform))}
            for i in cfg['accessories']]
        result['model_datums'][fid] = {
            'source': data['source'], 'source_sha256': data['source_sha256'],
            'source_origin_mm': origin.tolist(), 'station_T_source_mm': transform.tolist(),
            'foot_plane_source_mm': {'point': origin.tolist(), 'up_normal': up.tolist()},
            'foot_contact_faces': contact_faces, 'foot_datum_z_m': 0.,
            'foot_hole_centers_station_m': holes.tolist(), 'foot_pitch_longitudinal_transverse_mm': pitch.tolist(),
            'shaft_point_station_m': shaft_station.tolist(), 'shaft_direction_station': [1., 0., 0.],
            'shaft_reference_solid_index': cfg['shaft'], 'shaft_cylinder_face_index': cylinder['face_index'],
            'box_far_side_offset_m': station_T_box[0, 3],
            'box_origin_source_mm': source_T_box[:3, 3].tolist(),
            'removed_box_system_solids': cfg['box_solids'],
            'physical_motor_to_box_registration_confirmed': False,
        }
    return result


def export_visual(fid, result, output):
    """Export the actual horizontal motor, with separate external fittings.

    Meshes are visual-only. Using one concave mesh as MuJoCo collision geometry
    would silently convexify it; collision envelopes above are explicit.
    """
    import trimesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    reader.ReadFile(str(SOURCES[fid]))
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    indices = set(result['motor_envelopes'][fid]['source_solid_indices'] + DATUMS[fid]['accessories'])
    vertices, triangles, index = [], [], 0
    while explorer.More():
        solid = TopoDS.Solid_s(explorer.Current())
        if index in indices:
            BRepMesh_IncrementalMesh(solid, .35, False, .35, True).Perform()
            faces = TopExp_Explorer(solid, TopAbs_FACE)
            while faces.More():
                face = TopoDS.Face_s(faces.Current())
                loc = TopLoc_Location()
                tri = BRep_Tool.Triangulation_s(face, loc)
                if tri:
                    offset = len(vertices)
                    for i in range(1, tri.NbNodes() + 1):
                        p = tri.Node(i).Transformed(loc.Transformation())
                        vertices.append([p.X(), p.Y(), p.Z()])
                    for i in range(1, tri.NbTriangles() + 1):
                        a, b, c = tri.Triangle(i).Get()
                        order = [a, c, b] if face.Orientation() == TopAbs_REVERSED else [a, b, c]
                        triangles.append([offset + k - 1 for k in order])
                faces.Next()
        index += 1
        explorer.Next()
    mesh = trimesh.Trimesh(vertices, triangles, process=True)
    mesh.apply_transform(np.array(result['model_datums'][fid]['station_T_source_mm']))
    mesh.apply_scale(.001)
    path = output / (fid + '_motor_station_m.stl')
    mesh.export(path)
    return {'path': str(path.resolve()), 'units': 'metres', 'frame': 'station',
            'triangles': len(mesh.faces), 'visual_only': True,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def extract(fid, output):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    def vector(p):
        return [float(p.X()), float(p.Y()), float(p.Z())]

    def bounds(shape):
        b = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, b)
        data = b.Get()
        return [list(data[:3]), list(data[3:])]

    reader = STEPControl_Reader()
    reader.ReadFile(str(SOURCES[fid]))
    reader.TransferRoots()
    assembly = reader.OneShape()
    exp = TopExp_Explorer(assembly, TopAbs_SOLID)
    solids = []
    while exp.More():
        solid = TopoDS.Solid_s(exp.Current())
        volume = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, volume)
        item = {'index': len(solids), 'bounds_source_mm': bounds(solid),
                'volume_mm3': float(volume.Mass()), 'centroid_source_mm': vector(volume.CentreOfMass()),
                'cylinders': [], 'planes': []}
        faces = TopExp_Explorer(solid, TopAbs_FACE)
        index = 0
        while faces.More():
            face = TopoDS.Face_s(faces.Current())
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() in (GeomAbs_Cylinder, GeomAbs_Plane):
                prop = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, prop)
                record = {'face_index': index, 'area_mm2': float(prop.Mass()),
                          'centroid_source_mm': vector(prop.CentreOfMass()), 'bounds_source_mm': bounds(face)}
                if surface.GetType() == GeomAbs_Cylinder:
                    cylinder = surface.Cylinder()
                    record.update(radius_mm=float(cylinder.Radius()), axis_point_source_mm=vector(cylinder.Location()),
                                  axis_direction_source=vector(cylinder.Axis().Direction()))
                    item['cylinders'].append(record)
                else:
                    plane = surface.Plane()
                    record.update(normal_source=vector(plane.Axis().Direction()), plane_point_source_mm=vector(plane.Location()))
                    item['planes'].append(record)
            index += 1
            faces.Next()
        solids.append(item)
        exp.Next()
    result = {'model': fid, 'source': str(SOURCES[fid]),
              'source_sha256': hashlib.sha256(SOURCES[fid].read_bytes()).hexdigest(),
              'units': 'mm', 'assembly_bounds_source_mm': bounds(assembly), 'solids': solids}
    output.mkdir(parents=True, exist_ok=True)
    (output / (fid + '_assembly_features.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model': fid, 'solids': len(solids), 'bounds_mm': result['assembly_bounds_source_mm']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument('--reuse-features', action='store_true', help='Use previously extracted analytic feature JSONs')
    parser.add_argument('--visual-meshes', action='store_true', help='Export actual motor meshes without closed box lids')
    args = parser.parse_args()
    features = {}
    for fid in args.models:
        path = args.output / (fid + '_assembly_features.json')
        features[fid] = json.loads(path.read_text()) if args.reuse_features else extract(fid, args.output)
    result = derive_station(features)
    path = args.output / 'shared_station_geometry.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'station_geometry': str(path), 'station_T_boxes': result['station_T_boxes']}), flush=True)
    if args.visual_meshes:
        visuals = {fid: export_visual(fid, result, args.output) for fid in args.models}
        (args.output / 'motor_visuals.json').write_text(json.dumps(visuals, indent=2) + '\n')
        print(json.dumps({'motor_visuals': visuals}), flush=True)


if __name__ == '__main__':
    main()
