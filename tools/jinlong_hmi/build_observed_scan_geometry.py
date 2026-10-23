#!/usr/bin/env python3
"""Conservative JL100/JL132 cavity geometry in independently observed frames.

Every missing depth column stays occupied from the rim. Observed point centres
and nearest cap heights come from one RGB-D capture; cylinder radius and hidden
solid extent remain explicit model assumptions. No tool geometry is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENVELOPES_MM={'JL100':([103.5,103.5],46.),'JL132':([112.,112.],52.)}
KEYS=('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4')


def build(template_path,output,pitch=2.):
    template=json.loads(template_path.read_text());fid=template['model']
    if fid not in ENVELOPES_MM or template['tool_geometry_used_for_measurement']:
        raise ValueError('Only independent JL100/JL132 observations are supported')
    for path,expected in template['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Observation source changed: {path}')
    source=Path(template['source_scan']);intr_path=source/'camera_intrinsics.json'
    intr=json.loads(intr_path.read_text());depth_path=source/'03_depth_raw_u16.png'
    depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    if depth is None: raise ValueError('Missing source depth')
    T=np.array(template['camera_T_box']);widths,bottom=ENVELOPES_MM[fid]
    half=np.ceil(np.array(widths)/(2*pitch))*pitch
    shape=np.rint(2*half/pitch).astype(int)
    K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    distortion=np.array([intr['dist_coeffs'][key] for key in KEYS])
    yy,xx=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2];xx=xx.ravel();yy=yy.ravel()
    valid=depth[yy,xx]>0;xx,yy=xx[valid],yy[valid]
    rays=cv2.undistortPoints(np.c_[xx,yy].astype(float).reshape(-1,1,2),K,distortion)[:,0]
    camera=np.c_[rays,np.ones(len(rays))]*(depth[yy,xx]*intr['depth_unit_raw']/1000)[:,None]
    points=(camera-T[:3,3])@T[:3,:3]*1000
    points=points[(np.all(abs(points[:,:2])<half,axis=1))&(points[:,2]>=-5)&(points[:,2]<bottom)]
    exclusion=np.zeros(len(points),bool);terminals=[]
    # Collision numbering follows actual tool hole order, not alphabetical
    # winding labels. This mapping changes IDs only, never measured positions.
    order=[2,1,0,5,4,3,6] if fid=='JL132' else list(range(7))
    for index,feature_index in enumerate(order):
        f=template['features'][feature_index];ground=f['id']=='ground';radius=4. if ground else 3.
        xyz=np.array(f['xyz_box_mm'],float)
        cap=f.get('cap_surface_depth_mm_percentiles_5_50_95',f['depth_below_rim_mm_percentiles_5_50_95'])
        xyz[2]=float(cap[0])
        if not 0<=xyz[2]<bottom: raise ValueError(f'Invalid measured cap depth: {fid}/{f["id"]}')
        exclusion|=np.linalg.norm(points[:,:2]-xyz[:2],axis=1)<radius
        suffix=f'stud_{index+1}' if index<3 else (f'other_{index-2}' if index<6 else 'gnd')
        end=xyz.copy();end[2]=bottom
        terminals.append({'name':f'term_{fid}_{suffix}','a':(xyz/1000).tolist(),'b':(end/1000).tolist(),
                          'radius':radius/1000,'observed_electrical_label':f['id'],
                          'center_source':f['legacy_id'],
                          'center_measurement_method':'same-frame manual pixel ray at measured depth',
                          'radius_source':'Inherited M6/D8 collision assumption; not independently measured',
                          'length_source':'Conservatively fill hidden material from observed cap to manufacturer axial-depth envelope',
                          'depth_source':'Outward cap cluster P5 from same raw depth',
                          'observed_template_feature':f})
    surface=points[~exclusion];index=np.floor((surface[:,:2]+half)/pitch).astype(int)
    heights=np.full(shape,np.inf);counts=np.zeros(shape,int)
    np.minimum.at(heights,(index[:,0],index[:,1]),surface[:,2]);np.add.at(counts,(index[:,0],index[:,1]),1)
    centres=np.stack(np.meshgrid((np.arange(shape[0])+.5)*pitch-half[0],(np.arange(shape[1])+.5)*pitch-half[1],indexing='ij'),axis=-1)
    terminal_interior=np.zeros(shape,bool)
    for t in terminals:
        distance=np.linalg.norm(centres-np.asarray(t['a'][:2])*1000,axis=-1)
        terminal_interior|=distance+pitch/np.sqrt(2)<=t['radius']*1000
    unknown=~np.isfinite(heights)&~terminal_interior
    heights[unknown]=0.;heights[terminal_interior]=bottom
    # All seven replacement cylinders reach bottom; no column is left empty
    # beneath a short stud model when the point-cloud pixels are split out.
    levels=np.floor(heights).astype(int);used=np.zeros(shape,bool);boxes=[]
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            if used[ix,iy] or levels[ix,iy]>=bottom:continue
            level,missing=levels[ix,iy],unknown[ix,iy];end_y=iy+1
            while end_y<shape[1] and not used[ix,end_y] and levels[ix,end_y]==level and unknown[ix,end_y]==missing:end_y+=1
            end_x=ix+1
            while (end_x<shape[0] and not np.any(used[end_x,iy:end_y]) and
                   np.all(levels[end_x,iy:end_y]==level) and np.all(unknown[end_x,iy:end_y]==missing)):end_x+=1
            used[ix:end_x,iy:end_y]=True
            lo=np.array([ix*pitch-half[0],iy*pitch-half[1],level]);hi=np.array([end_x*pitch-half[0],end_y*pitch-half[1],bottom])
            boxes.append({'center':((lo+hi)/2000).tolist(),'half_size':((hi-lo)/2000).tolist(),'unknown':bool(missing)})
    report_path=ROOT/'docs/jinlong_terminal_box_geometry_20261023.md'
    result={'model':fid,'units':'metres','status':'offline_conservative_same_frame_rgbd_height_field',
            'source_scan':str(source),'camera_T_box':T.tolist(),'frame':template['frame'],
            'frame_id':f'{fid}/{template["source_task"]}/cover-hole-observed-box',
            'observation_template':str(template_path.resolve()),'grid_pitch_mm':pitch,'z_quantization_mm':1.,
            'envelope_dimensions_mm':{'manufacturer_outer_xy':widths,'axial_total_depth':bottom,
                                      'rounded_grid_outer_xy':(2*half).tolist(),
                                      'source':str(report_path),'meaning':'Conservative outer envelope, never interpreted as clear inner opening'},
            'grid_shape':shape.tolist(),'retained_depth_points':len(points),'total_columns':int(unknown.size),
            'unknown_columns':int(unknown.sum()),'terminal_only_columns':int(terminal_interior.sum()),
            'boxes':boxes,'parts':[],'measured_terminals':terminals,
            'hidden_terminal_columns_filled_to_bottom':True,'physical_station_registration_verified':False,
            'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path(__file__),template_path,intr_path,depth_path,report_path]},
            'limitations':['Nearest observed surface fills all hidden space below; this overestimates some concavities.',
                          'Missing-depth columns start at rim and may conservatively block feasible insertion.',
                          'Outer dimensions are manufacturer values centered on observed cover-hole frame; exact as-built exterior asymmetry is not measured.',
                          'Stud/pin diameters and hidden solid extents remain modelling assumptions.',
                          'No CAD ground-hole/cable-side registration is assumed.',
                          'Motor feet, shaft, table and camera are not registered by this cavity model.',
                          'This historical capture does not establish current electrical wiring state or physical insertion success.']}
    output.mkdir(parents=True,exist_ok=True)
    path=output/(fid+'_collision.json');path.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/(fid+'_height_field.npz'),heights=heights,counts=counts,unknown=unknown,terminal_interior=terminal_interior,half_extent=half)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(8,7));shown=heights.copy();shown[unknown|terminal_interior]=np.nan
    extent=[-half[0],half[0],-half[1],half[1]]
    im=ax.imshow(shown.T,origin='lower',extent=extent,cmap='viridis_r',vmin=0,vmax=bottom)
    ax.imshow(np.ma.masked_where(~unknown.T,unknown.T),origin='lower',extent=extent,cmap=ListedColormap(['#db5e70']),alpha=.85)
    for t in terminals:
        xy=np.array(t['a'][:2])*1000;ax.plot(*xy,'wo',mec='black');ax.annotate(t['observed_electrical_label'],xy,xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='Observed box X (mm)',ylabel='Observed box Y (mm)',title=f'{fid}: same-frame conservative cavity\nPink = missing depth retained as occupied')
    fig.colorbar(im,ax=ax,label='Observed nearest surface below rim (mm)');fig.tight_layout();fig.savefig(output/(fid+'_height_field.png'),dpi=140);plt.close(fig)
    print(json.dumps({'model':fid,'path':str(path),'boxes':len(boxes),'unknown_columns':int(unknown.sum()),'total_columns':int(unknown.size)}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pitch-mm',type=float,default=2.)
    args=parser.parse_args()
    if not 0<args.pitch_mm<=2: parser.error('Grid pitch must be in (0,2] mm')
    build(args.template,args.output,args.pitch_mm)


if __name__=='__main__':main()
