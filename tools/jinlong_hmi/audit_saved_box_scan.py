#!/usr/bin/env python3
"""Independent, manual-ROI audit of the archived 20261022_110619_cb9e44 scan.

No camera/network/robot calls. Coordinates below belong to this exact image.
The legacy result's fitted plane and inferred hole positions are not reused.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REFERENCE_TASK = '20261022_110619_cb9e44'
# Manually selected visible rim patches, native 3200x1944 image coordinates.
RIM = {
    'top': [[1188,136],[2125,178],[2125,208],[1188,169]],
    'left': [[973,391],[1008,391],[977,1250],[945,1250]],
    'right': [[2266,391],[2302,391],[2289,1250],[2250,1250]],
    'bottom': [[1125,1398],[2109,1423],[2109,1450],[1125,1431]],
}
GROUND = [[1848,1289],[1898,1289],[1898,1336],[1848,1336]]
# Straight visible edge segments; their intersections are virtual sharp corners.
# They omit chamfers, corner posts and interior obstructions.
EDGES = {
    'outer': [[[1183,114],[1494,133]], [[2331,350],[2325,706]],
              [[1197,1452],[1521,1457]], [[983,400],[947,1228]]],
    'inner': [[[1162,179],[2166,236]], [[2242,350],[2230,1237]],
              [[1141,1390],[1467,1394]], [[1046,400],[1014,1224]]],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.scan.name != REFERENCE_TASK:
        ap.error('Manual ROIs apply only to ' + REFERENCE_TASK)
    a.out.mkdir(parents=True, exist_ok=True)
    depth = np.load(a.scan/'03_depth_raw.npy')
    native_png = cv2.imread(str(a.scan/'03_depth_raw_u16.png'), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(depth, native_png):
        raise ValueError('NPY and PNG raw depths differ')
    raw = cv2.imread(str(a.scan/'01_rgb_original.png'))
    intr = json.loads((a.scan/'camera_intrinsics.json').read_text())
    if depth.shape != raw.shape[:2] or depth.shape != (intr['height'], intr['width']):
        raise ValueError('Image/depth/intrinsic dimensions differ')
    K = np.array([[intr['fx'],0,intr['cx']], [0,intr['fy'],intr['cy']], [0,0,1.]])
    keys = ['k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4']
    dist = np.array([intr['dist_coeffs'][k] for k in keys])
    scale = intr['depth_unit_raw'] / 1000

    def rays(uv):
        if len(uv) == 0:
            return np.empty((0,3))
        xy = cv2.undistortPoints(np.asarray(uv, float).reshape(-1,1,2), K, dist)[:,0]
        return np.column_stack([xy, np.ones(len(xy))])

    def sample(poly):
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 1)
        y,x = np.where(mask & (depth > 0))
        points = rays(np.column_stack([x,y])) * (depth[y,x] * scale)[:,None]
        return points, {'polygon_px': poly, 'roi_pixels': int(mask.sum()),
                        'valid_depth_pixels': len(points),
                        'valid_depth_fraction': float(len(points)/mask.sum())}

    patches, stats = [], {}
    for name, poly in RIM.items():
        xyz, stats[name] = sample(poly)
        patches.append(xyz)
    points = np.concatenate(patches)
    keep = np.ones(len(points), bool)
    for _ in range(5):
        center = points[keep].mean(axis=0)
        normal = np.linalg.svd(points[keep]-center, full_matrices=False)[2][-1]
        if normal[2] < 0:
            normal = -normal
        d = -float(normal @ center)
        residual = points @ normal + d
        keep = abs(residual) < .001
    rmse_mm = float(np.sqrt(np.mean(residual[keep]**2))*1000)
    ground, ground_stats = sample(GROUND)
    ground_offset = (ground @ normal + d)*1000
    ground_stats.update(depth_optical_mm_percentiles_5_50_95=np.percentile(ground[:,2]*1000,[5,50,95]).tolist(),
                        below_rim_plane_mm_percentiles_5_50_95=np.percentile(ground_offset,[5,50,95]).tolist())

    def measure(edges):
        lines = [np.cross([*p,1], [*q,1]) for p,q in edges]
        corners = []
        for i in range(4):
            h = np.cross(lines[i-1], lines[i]); corners.append((h[:2]/h[2]).tolist())
        r = rays(corners)
        xyz = r * (-d/(r@normal))[:,None]
        lengths = np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1)*1000
        return {'edge_segments_px': edges, 'virtual_corner_pixels': corners,
                'opposite_edge_lengths_mm': lengths.tolist(),
                'opposite_pair_mean_mm': [float(lengths[[0,2]].mean()), float(lengths[[1,3]].mean())]}

    geometry = {name: measure(edges) for name,edges in EDGES.items()}
    result = {'reference_task': REFERENCE_TASK,
              'input_sha256': {name: hashlib.sha256((a.scan/name).read_bytes()).hexdigest()
                              for name in ['03_depth_raw.npy','03_depth_raw_u16.png','01_rgb_original.png','camera_intrinsics.json','result.json','latest.jpg']},
              'raw_depth_npy_equals_png': True,
              'pairing_review': 'Raw RGB and own latest overlay show the same placement by visual review; NPY and PNG depth arrays agree. No hardware timestamp guarantee inferred.',
              'legacy_result_used_for_plane': False, 'rim_rois': stats,
              'independent_rim_plane': {'normal': normal.tolist(),'d_m': d,'rmse_mm':rmse_mm,
                                        'valid_points':len(points),'retained_points':int(keep.sum()),
                                        'retained_fraction':float(keep.mean()),'rejection_threshold_mm':1.0},
              'manual_visible_edge_estimates': geometry, 'ground_screw_cap_roi':ground_stats,
              'limitations': ['Manual visible straight edges are approximate; curved/chamfered corners and inner protrusions are excluded.',
                              'Plane residual is internal consistency, not absolute dimensional accuracy; no independent error bound or drawing-revision acceptance is established.',
                              'Rim plane is independently fitted; image edges use lens-undistorted ray intersections.',
                              'The left rim has missing depth; the other visible rim patches support the fit.',
                              'Ground ROI includes a screw cap and nearby pixels; quantiles are not a fitted screw datum.',
                              'This image has no installed inspection tool, so no tool-to-wall or pogo-pin clearance is measured.',
                              'No production geometry, thresholds or robot commands are changed.'],
              'executable':False}
    (a.out/'scan_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    for name,poly in RIM.items():
        q=np.vstack([poly,poly[0]]);ax.plot(q[:,0],q[:,1],color='cyan',lw=1)
        ax.text(*poly[0],name+' rim',color='cyan',fontsize=9)
    q=np.vstack([GROUND,GROUND[0]]);ax.plot(q[:,0],q[:,1],color='magenta',lw=2)
    ax.text(*GROUND[0],'ground ROI',color='magenta',fontsize=9)
    for name,doc in geometry.items():
        q=np.array(doc['virtual_corner_pixels']);q=np.vstack([q,q[0]])
        ax.plot(q[:,0],q[:,1],color='lime' if name=='inner' else 'yellow',lw=1.5,label=name+' straight-edge envelope')
    ax.set_xlim(850,2440);ax.set_ylim(1550,50);ax.legend(loc='upper right',fontsize=8)
    ax.set_title('Archived JM160: manual rim/ground ROIs and approximate visible edge envelopes\nIndependent rim fit; no inspection tool installed')
    fig.tight_layout();fig.savefig(a.out/'scan_audit.png',dpi=160);plt.close(fig)
    print(json.dumps({'rim':result['independent_rim_plane'],'edges':geometry,'ground':ground_stats},ensure_ascii=False))


if __name__ == '__main__':
    main()
