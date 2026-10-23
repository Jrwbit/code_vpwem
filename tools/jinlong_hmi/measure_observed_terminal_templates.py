#!/usr/bin/env python3
"""Independent six-stud templates from archived same-frame RGB-D observations.

Manual pixel centres refer only to the named raw images. The box frame is
constructed from visible cover-hole centres projected onto the independently
fitted rim. No fixture transform or robot solution enters the measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path('/home/wjr/code/r435/outputs/hik_web_pipeline')
DEFINITIONS = {
    'JL100': dict(task='20260730_205445_815e27',
                  holes=[[1721,541],[2212,526],[2208,1039],[1715,1030]],
                  studs=[[1915,667],[1914,788],[1911,903],[2033,667],[2030,788],[2028,903]],
                  ground=[1983,998], crop=[1620,2290,460,1110]),
    'JL132': dict(task='20260729_214004_7c4f40',
                  holes=[[1082,552],[1641,340],[1864,900],[1312,1128]],
                  studs=[[1344,623],[1395,761],[1450,902],[1483,567],[1538,703],[1594,847]],
                  ground=[1553,978], crop=[980,1970,250,1230]),
    'JM160': dict(task='20261022_110619_cb9e44',
                  holes=[[1056,176],[2263,245],[2241,1395],[1009,1375]],
                  studs=[[1300,794],[1485,979],[1660,1161],[1675,438],[1850,620],[2015,808]],
                  ground=[1873,1312], crop=[900,2390,80,1490]),
}
DISTORTION_KEYS = ('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4')


def measure(fid, output):
    definition = DEFINITIONS[fid]
    if fid == 'JM160':
        source = ROOT/'outputs/diagnostics/front-fixed-waist-20261023/scans'/definition['task']
        audit_path = source.parent/'cb9_audit/scan_audit.json'
        audit = json.loads(audit_path.read_text())
        plane = audit['independent_rim_plane']
        normal, offset = np.array(plane['normal']), plane['d_m']
        old_scan_path = ROOT/'outputs/diagnostics/full-solve-20261023/scan_contacts.json'
        old_scan = json.loads(old_scan_path.read_text())
    else:
        source = RAW_ROOT/definition['task']
        audit_path = ROOT/f'outputs/diagnostics/full-solve-20261023/jl_contacts/{fid}_contacts.json'
        audit = json.loads(audit_path.read_text())
        plane = audit['rim_plane']
        normal, offset = np.array(plane['normal']), plane['offset_m']
    rgb_path, depth_path = source/'01_rgb_original.png', source/'03_depth_raw_u16.png'
    rgb = cv2.imread(str(rgb_path)); depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    intr_path = source/'camera_intrinsics.json'
    intr = json.loads(intr_path.read_text())
    if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
        raise ValueError(f'{fid}: missing aligned raw RGB-D')
    if (source/'03_depth_raw.npy').exists() and not np.array_equal(depth,np.load(source/'03_depth_raw.npy')):
        raise ValueError('PNG and NPY depth disagree')
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    distortion = np.array([intr['dist_coeffs'][k] for k in DISTORTION_KEYS])

    def rays(uv):
        xy = cv2.undistortPoints(np.asarray(uv,float).reshape(-1,1,2),K,distortion)[:,0]
        return np.c_[xy,np.ones(len(xy))]

    def frame(holes):
        q = rays(holes); q *= (-offset/(q@normal))[:,None]
        x = (q[0]-q[1]+q[3]-q[2])/2
        x -= normal*(x@normal); x /= np.linalg.norm(x)
        T = np.eye(4); T[:3,:3] = np.c_[x,np.cross(normal,x),normal]; T[:3,3] = q.mean(0)
        return T,q

    camera_T_box, corners = frame(definition['holes'])
    inverse = np.linalg.inv(camera_T_box)
    features=[]
    for index, uv in enumerate(definition['studs']+[definition['ground']]):
        u,v=uv; radius=12 if fid=='JM160' else 8
        yy,xx=np.mgrid[v-radius:v+radius+1,u-radius:u+radius+1]
        mask=((xx-u)**2+(yy-v)**2 < radius**2)&(depth[yy,xx]>0)
        cloud=rays(np.c_[xx[mask],yy[mask]])*(depth[yy[mask],xx[mask]]*intr['depth_unit_raw']/1000)[:,None]
        local=(cloud-camera_T_box[:3,3])@camera_T_box[:3,:3]*1000
        xyz=np.median(local,axis=0)
        # Axis centre is the manually identified visible stud tip, not the
        # coordinate median of an irregular valid-depth patch.
        ray=rays([uv])[0]
        z=xyz[2]/1000
        camera=ray*((z-offset)/(ray@normal))
        xyz_axis=(inverse@np.r_[camera,1])[:3]*1000
        uv_perturb=np.asarray(uv)+np.array([[3,0],[-3,0],[0,3],[0,-3]])
        pr=rays(uv_perturb); pp=pr*((z-offset)/(pr@normal))[:,None]
        sensitivity=float(np.max(np.linalg.norm(pp-camera,axis=1))*1000)
        f={'id':(['U1','V1','W1','W2','U2','V2','ground'][index]),
           'legacy_id':f'stud_{index+1}' if index<6 else 'ground',
           'pixel_center':uv,'radius_px':radius,'xyz_box_mm':xyz_axis.tolist(),
           'xyz_camera_m':camera.tolist(),'median_patch_xyz_box_mm':xyz.tolist(),
           'depth_below_rim_mm_percentiles_5_50_95':np.percentile(local[:,2],[5,50,95]).tolist(),
           'valid_pixels':len(cloud),'pixel_center_plus_minus_3px_sensitivity_mm':sensitivity}
        if fid!='JM160':
            f['cap_surface_depth_mm_percentiles_5_50_95']=audit['features'][index]['outward_cap_cluster']['depth_mm_percentiles_5_50_95']
        if index==6:
            if fid=='JM160':
                f['cap_surface_depth_mm_percentiles_5_50_95']=audit['ground_screw_cap_roi']['below_rim_plane_mm_percentiles_5_50_95']
        features.append(f)
    # A bounded pixel-perturbation sensitivity exercise, not an error bar or
    # claim of metrology. Source RGB-D accuracy is not independently known.
    rng=np.random.default_rng(20261023)
    sensitivity=[]
    for _ in range(400):
        perturbed,_=frame(np.asarray(definition['holes'])+rng.uniform(-4,4,(4,2)))
        xyz=np.array([f['xyz_camera_m'] for f in features])
        local=(xyz-perturbed[:3,3])@perturbed[:3,:3]*1000
        baseline=np.array([f['xyz_box_mm'] for f in features])
        sensitivity.append(np.linalg.norm(local[:,:2]-baseline[:,:2],axis=1))
    row=np.array(features[2]['xyz_box_mm'])-np.array(features[0]['xyz_box_mm'])
    result={'model':fid,'status':'independent_manual_rgbd_template_offline','units':'metres',
            'source_scan':str(source),'source_task':definition['task'],'frame':
            'Origin is mean of four visible cover-hole centres projected onto independently fitted rim; '
            '+X points from image-right holes toward image-left holes, +Y toward side opposite the ground screw, +Z into cavity.',
            'camera_T_box':camera_T_box.tolist(),
            'hole_centres_pixel_visual_TL_TR_BR_BL':definition['holes'],
            'hole_centres_camera_m':corners.tolist(),
            'hole_centres_box_mm':((corners-camera_T_box[:3,3])@camera_T_box[:3,:3]*1000).tolist(),
            'cover_hole_edge_lengths_mm':(np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)*1000).tolist(),
            'rim_plane':plane,'features':features,
            'first_row_direction_deg_in_box_xy':float(np.degrees(np.arctan2(row[1],row[0]))),
            'hole_center_plus_minus_4px_max_sampled_xy_sensitivity_mm':np.max(sensitivity,axis=0).tolist(),
            'motor_axis_or_station_registration_confirmed':False,
            'tool_geometry_used_for_measurement':False,
            'limitations':['Manual visible feature centres; pixel perturbations are sensitivity tests, not confidence intervals.',
                          'Radius/shaft length/socket bore are not measured by this point template.',
                          'The frame does not establish motor-foot/shaft-to-box registration.',
                          'Historical raw image belongs to the named capture only; no present camera calibration or live robot data are used.',
                          'Depth medians at ground centre may include screw recess; cap surface depths are separately supplied.'],
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),rgb_path,depth_path,intr_path,audit_path]}}
    if fid=='JM160':
        result['observed_T_previous_scan_box']= (inverse@np.asarray(old_scan['camera_T_box'])).tolist()
        result['previous_scan_contacts_source']=str(old_scan_path)
        result['source_sha256'][str(old_scan_path)]=hashlib.sha256(old_scan_path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/(fid+'_observed_template.json')).write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    axes[0].imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
    hp=np.asarray(definition['holes']);axes[0].plot(*np.vstack([hp,hp[0]]).T,'c.-')
    for i,p in enumerate(hp): axes[0].annotate('H'+str(i+1),p,color='cyan')
    for f in features:
        axes[0].plot(*f['pixel_center'],'yo',ms=4)
        axes[0].annotate(f['id'],f['pixel_center'],xytext=(8,-8),textcoords='offset points',color='yellow')
        axes[1].plot(*f['xyz_box_mm'][:2],'o',label=f['id'])
        axes[1].annotate(f['id'],f['xyz_box_mm'][:2],xytext=(4,4),textcoords='offset points')
    x0,x1,y0,y1=definition['crop'];axes[0].set(xlim=(x0,x1),ylim=(y1,y0),title=f'{fid} {definition["task"]}: same raw RGB-D')
    bh=np.asarray(result['hole_centres_box_mm']);axes[1].plot(*np.vstack([bh,bh[0]])[:,:2].T,'c.-')
    axes[1].set(aspect='equal',xlabel='Observed box +X (mm)',ylabel='Observed box +Y (mm)',title='Independent cover-hole frame; six actual studs + ground')
    axes[1].grid(True);fig.tight_layout();fig.savefig(output/(fid+'_observed_template.png'),dpi=140);plt.close(fig)
    print(json.dumps({'model':fid,'file':str(output/(fid+'_observed_template.json')),
                      'row_direction_deg':result['first_row_direction_deg_in_box_xy'],
                      'features_mm':{f['id']:f['xyz_box_mm'] for f in features}}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for fid in DEFINITIONS: measure(fid,args.output)


if __name__=='__main__': main()
