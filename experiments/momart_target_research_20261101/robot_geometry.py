"""Geometry from robot-owned measurements and observed depth, never object GT poses."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


class RobotGeometry:
    def __init__(self,urdf,scale=1.):
        import pybullet as p
        self.p=p;self.client=p.connect(p.DIRECT)
        self.robot=p.loadURDF(str(urdf),useFixedBase=True,globalScaling=scale,
            flags=p.URDF_IGNORE_COLLISION_SHAPES|p.URDF_IGNORE_VISUAL_SHAPES,physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.robot,[0,0,0],[0,0,0,1],physicsClientId=self.client)
        info={i:p.getJointInfo(self.robot,i,physicsClientId=self.client) for i in range(p.getNumJoints(self.robot,physicsClientId=self.client))}
        self.movable=[i for i,value in info.items() if value[2]!=p.JOINT_FIXED]
        self.joints={value[1].decode():i for i,value in info.items()}
        self.links={value[12].decode():i for i,value in info.items()}
        assert p.getNumBodies(physicsClientId=self.client)==1

    def infer_robot_layout(self,states,proprio):
        """Offline schema validation; matches head/EEF measurements, not scene objects."""
        p=self.p;head=[self.movable.index(self.joints[name]) for name in ('head_pan_joint','head_tilt_joint')]
        candidates=[]
        for offset in range(states.shape[1]-13-len(self.movable)+1):
            error=float(np.sqrt(np.mean((states[:,[offset+13+j for j in head]]-proprio[:,:2])**2)))
            if error>.02:continue
            errors=[]
            for step in np.linspace(0,len(states)-1,min(16,len(states)),dtype=int):
                for joint,value in zip(self.movable,states[step,offset+13:offset+13+len(self.movable)]):
                    p.resetJointState(self.robot,joint,float(value),physicsClientId=self.client)
                ee=p.getLinkState(self.robot,self.links['gripper_link'],computeForwardKinematics=True,physicsClientId=self.client)[4]
                errors.append(float(np.linalg.norm(np.asarray(ee)-proprio[step,3:6])))
            candidates.append(dict(offset=offset,head_rmse=error,eef_median=float(np.median(errors)),eef_p95=float(np.quantile(errors,.95))))
        accepted=[row for row in candidates if row['eef_median']<.03 and row['eef_p95']<.10]
        if len(accepted)!=1:raise ValueError('Ambiguous/invalid robot state schema: '+str(candidates))
        return accepted[0]

    def project(self,proprio,torso,fov):
        p=self.p
        for name,value in [('head_pan_joint',proprio[0]),('head_tilt_joint',proprio[1]),('torso_lift_joint',torso)]:
            p.resetJointState(self.robot,self.joints[name],float(value),physicsClientId=self.client)
        pose=p.getLinkState(self.robot,self.links['eyes'],computeForwardKinematics=True,physicsClientId=self.client)
        rotation=np.asarray(p.getMatrixFromQuaternion(pose[5])).reshape(3,3)
        ee=(np.asarray(proprio[3:6])-pose[4])@rotation
        return ee,project_uv(ee,fov)

    def close(self):self.p.disconnect(physicsClientId=self.client)


def project_uv(point,fov):
    point=np.asarray(point);depth=point[0]
    if depth<=.01:return np.zeros(2,np.float32),False
    tangent=np.tan(np.deg2rad(fov)/2)
    return np.array([.5-point[1]/(2*depth*tangent*4/3),.5-point[2]/(2*depth*tangent)],np.float32),True


def box_geometry(box,depth,ee,fov,depth_high):
    box=np.asarray(box);image=np.asarray(depth)
    if image.ndim==3 and image.shape[-1]==1:image=image[...,0]
    if image.ndim!=2:raise ValueError('Expected HxW or HxWx1 depth')
    height,width=image.shape
    uv,forward=project_uv(ee,fov);center=(box[:2]+box[2:])/2
    gap=np.maximum(0,np.maximum(box[:2]-uv,uv-box[2:]));inside=float(forward and np.all(uv>=box[:2]) and np.all(uv<=box[2:]))
    x1,y1,x2,y2=(box*[width,height,width,height]).astype(int)
    x1=max(0,min(width-1,x1));y1=max(0,min(height-1,y1));x2=max(x1+1,min(width,x2));y2=max(y1+1,min(height,y2))
    cropped=image[y1:y2,x1:x2];yy,xx=np.mgrid[y1:y2,x1:x2];valid=np.isfinite(cropped)&(cropped>0)
    tail=np.r_[center-uv,gap,inside,valid.mean()]
    if not valid.any():return np.r_[np.zeros(10),tail].astype(np.float32)
    d=cropped[valid]*depth_high;u=(xx[valid]+.5)/width;v=(yy[valid]+.5)/height;t=np.tan(np.deg2rad(fov)/2)
    points=np.stack([d,-(u-.5)*2*d*t*4/3,-(v-.5)*2*d*t],-1);delta=points-np.asarray(ee)
    distance=np.linalg.norm(delta,axis=-1);nearest=int(distance.argmin())
    return np.r_[distance[nearest],np.quantile(distance,[.1,.5]),np.mean(distance<.05),delta[nearest],np.median(delta,axis=0),tail].astype(np.float32)


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.corpus/'manifest.json').read_text());meta=json.loads(a.features.with_suffix('.json').read_text())
    arrays=np.load(a.features,allow_pickle=False);assert meta['ids']==[row['id'] for row in manifest['records']]
    assert meta['manifest_sha256']==hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest()
    files={};geometry=None;layouts={};candidate=[];contexts=[]
    try:
        for i,row in enumerate(manifest['records']):
            if row['source'] not in files:files[row['source']]=h5py.File(row['source'],'r')
            f=files[row['source']];cfg=json.loads(f['data'].attrs['env_args'])['env_kwargs']['ig_config'];ep=f['data'][row['demo']]
            if geometry is None:geometry=RobotGeometry(a.urdf,float(cfg.get('robot_scale',1.)))
            if row['source'] not in layouts:
                layouts[row['source']]=geometry.infer_robot_layout(np.asarray(ep['states']),np.asarray(ep['obs/proprio']))
            layout=layouts[row['source']];t=row['step'];past=max(0,t-8)
            qcol=layout['offset']+13+geometry.movable.index(geometry.joints['torso_lift_joint'])
            fov=float(cfg.get('vertical_fov',90.));depth_high=float(cfg.get('depth_high',5.))
            measured=np.asarray(ep['obs/proprio'][t]);previous=np.asarray(ep['obs/proprio'][past])
            torso=float(ep['states'][t,qcol]);old_torso=float(ep['states'][past,qcol])
            ee,(uv,forward)=geometry.project(measured,torso,fov)
            old_ee,(old_uv,_)=geometry.project(previous,old_torso,fov)
            contexts.append(np.r_[ee,uv,torso,float(forward),fov/90,depth_high,ee-old_ee,uv-old_uv])
            depth=np.asarray(ep['obs/depth'][t]);extra=np.zeros((8,16),np.float32)
            for slot in np.flatnonzero(arrays['valid'][i]):extra[slot]=box_geometry(arrays['candidates'][i,slot,128:132],depth,ee,fov,depth_high)
            candidate.append(extra)
            if (i+1)%64==0:print(json.dumps(dict(processed=i+1,total=len(manifest['records']),device='CPU PyBullet DIRECT, one robot only')),flush=True)
    finally:
        for f in files.values():f.close()
        if geometry is not None:geometry.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,candidates=np.asarray(candidate,np.float32),context=np.asarray(contexts,np.float32))
    a.output.with_suffix('.json').write_text(json.dumps(dict(ids=meta['ids'],manifest_sha256=meta['manifest_sha256'],layouts=layouts,
        inputs='head joints and relative EEF pose from proprio; robot-owned torso joint measurement; observed normalized depth; calibrated URDF/FOV',
        required_online_addition='Expose measured trunk_joint_pos already available from robot proprio API; do not load full simulator state online',
        no_object_gt_poses=True,no_future_inputs=True,urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
        camera_axes='forward +x, screen right -y, screen up +z; depth normalized by configured depth_high'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['corpus','features','urdf','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
