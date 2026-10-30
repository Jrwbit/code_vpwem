"""Annotation-only visible mesh labels, with RGB alignment and review sheets.

Curated assets: dishwasher 12349_handle and drawer cabinet 49140. CAD/state
are privileged label sources only. No detector or online policy imports this.
Each drawer has one continuous pull with two support ends. CAD face selection
must be visually reviewed before the labels are used as a pilot reference.
"""
import argparse
import copy
import json
from pathlib import Path
import types

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw


def face_render(self, shadow_pass=0):
    saved = {k: self.renderer.faces[k] for k in self.annotation_faces}
    try:
        for k, value in self.annotation_faces.items():
            self.renderer.faces[k] = value
        type(self).render(self, shadow_pass=shadow_pass)
    finally:
        for k, value in saved.items():
            self.renderer.faces[k] = value


def clone_group(inst, selected, label, faces=None):
    clone = copy.copy(inst)
    clone.objects = [inst.objects[i] for i in selected]
    for name in ['poses_trans','poses_rot','last_trans','last_rot','link_ids']:
        setattr(clone, name, [getattr(inst,name)[i] for i in selected])
    clone.class_id = label
    if faces is not None:
        clone.annotation_faces = faces
        clone.render = types.MethodType(face_render, clone)
    return clone


def drawer_faces(renderer, visual):
    from gibson2.render.mesh_renderer import tinyobjloader
    reader=tinyobjloader.ObjReader(); assert reader.ParseFromFile(visual.filename)
    vertices=np.asarray(reader.GetAttrib().vertices).reshape(-1,3)
    partitions=[{}, {}, {}]
    for vao, shape in zip(visual.VAO_ids, reader.GetShapes()):
        indices=shape.mesh.numpy_indices().reshape(-1,3)[:,0]
        xyz=vertices[indices].reshape(-1,3,3)
        # Curated 49140 CAD: a continuous pull with ends at x≈±.44,
        # near the top of each drawer. A y-only split captures the panel;
        # require all triangle vertices inside the actual grip region.
        zmax=vertices[:,2].max()
        handle=((xyz[:,:,1].max(1)<=-.56124) &
                (np.abs(xyz[:,:,0]).min(1)>.38) & (np.abs(xyz[:,:,0]).max(1)<.50) &
                (xyz[:,:,2].min(1)>zmax-.097) & (xyz[:,:,2].max(1)<zmax-.071))
        left=handle & (xyz[:,:,0].mean(1)<0)
        right=handle & ~left
        for result, mask in zip(partitions,[~handle,left,right]):
            result[vao]=np.ascontiguousarray(renderer.faces[vao][mask])
        assert left.any() and right.any() and (~handle).any()
    return partitions


def sync_all(renderer):
    """Restore renderer transforms even for sleeping/static recorded bodies."""
    import pybullet as p
    from gibson2.render.mesh_renderer.instances import Instance, InstanceGroup
    from gibson2.utils.mesh_util import xyz2mat,quat2rotmat,xyzw2wxyz
    def base_pose(body):
        info=p.getDynamicsInfo(body,-1)
        inv=p.invertTransform(info[3],info[4])
        pos,orn=p.getBasePositionAndOrientation(body)
        return p.multiplyTransforms(pos,orn,*inv)
    for inst in renderer.instances:
        if isinstance(inst,Instance):
            pos,orn=base_pose(inst.pybullet_uuid)
            inst.set_position(pos);inst.set_rotation(quat2rotmat(xyzw2wxyz(orn)))
        elif isinstance(inst,InstanceGroup):
            for index,link in enumerate(inst.link_ids):
                if link==-1:pos,orn=base_pose(inst.pybullet_uuid)
                else:
                    state=p.getLinkState(inst.pybullet_uuid,link,computeForwardKinematics=True)
                    pos,orn=state[4:6]
                inst.set_position_for_part(xyz2mat(pos),index)
                inst.set_rotation_for_part(quat2rotmat(xyzw2wxyz(orn)),index)


def descriptors(renderer):
    parts=[]
    for inst in renderer.instances:
        if not hasattr(inst,'objects'):
            if '024_bowl' in str(getattr(getattr(inst,'object',None),'filename','')):
                parts.append(dict(inst=inst, indices=None, cls=0, identity=f'{inst.id}:bowl',
                                  parents=[], faces=None))
            continue
        files=[str(o.filename) for o in inst.objects]
        if any('/dishwasher/12349_handle/' in f for f in files):
            assert len(files)==6 and len(set(inst.link_ids))==4
            # base, door mesh, door primitive handle, rack1 mesh/handle, rack2.
            for indices, cls, key, parents in [([1],1,'door',[]),([2],2,'door_handle',['door']),
                                               ([3,4],3,'rack1',[]),([5],3,'rack2',[])]:
                parts.append(dict(inst=inst,indices=indices,cls=cls,identity=f'{inst.id}:{key}',
                                  parents=[f'{inst.id}:{p}' for p in parents],faces=None))
        elif any('/bottom_cabinet/49140/' in f for f in files):
            for idx,f in enumerate(files):
                if '/link_0_' in f:continue
                assert '/link_' in f
                masks=drawer_faces(renderer,inst.objects[idx])
                drawer=f'{inst.id}:drawer:{idx}'
                handle={k:np.concatenate([masks[1][k],masks[2][k]]) for k in masks[1]}
                for suffix,cls,faces in zip(['body','handle'],[4,5],[masks[0],handle]):
                    identity=drawer if cls==4 else drawer+':'+suffix
                    parts.append(dict(inst=inst,indices=[idx],cls=cls,identity=identity,
                                      parents=[drawer] if cls==5 else [],faces=faces))
    assert {p['cls'] for p in parts} == set(range(6)), 'Curated assets not present'
    return parts


def masks_at_state(renderer, parts):
    original=renderer.instances
    classes=[i.class_id for i in original]
    try:
        for inst in original:inst.class_id=0
        removed={id(p['inst']) for p in parts if p['indices'] is not None}
        new=[i for i in original if id(i) not in removed]
        # Keep non-target CAD surfaces as depth occluders with background ID.
        for inst in original:
            if id(inst) in removed:
                covered=set(i for p in parts if p['inst'] is inst for i in p['indices'])
                rest=[i for i in range(len(inst.objects)) if i not in covered]
                if rest:new.append(clone_group(inst,rest,0))
        for label,p in enumerate(parts,1):
            if p['indices'] is None:p['inst'].class_id=label
            else:new.append(clone_group(p['inst'],p['indices'],label,p['faces']))
        renderer.instances=new
        rendered=renderer.render_robot_cameras(modes=['seg'])[0]
        ids=np.rint(np.asarray(rendered)[...,0]*255).astype(np.uint8)
        ids=cv2.resize(ids,(120,90),interpolation=cv2.INTER_NEAREST)
    finally:
        renderer.instances=original
        for inst,cls in zip(original,classes):inst.class_id=cls
    masks={p['identity']:ids==label for label,p in enumerate(parts,1)}
    # Parent labels include the same visible pixels as their own handles.
    for p in parts:
        for parent in p['parents']:masks[parent] |= masks[p['identity']]
    return masks


def annotate(args):
    import egl_probe
    egl_probe.get_available_devices=lambda:[args.gpu]
    from robomimic.envs.env_ig_momart import EnvGibsonMOMART
    from gibson2.external.pybullet_tools.utils import WorldSaver
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    rows=[r for r in manifest['records'] if args.task is None or r['task']==args.task]
    annotations=[];rejected=[];env=None;scene_signature=None
    for task in sorted(set(r['task'] for r in rows)):
        selected=[r for r in rows if r['task']==task]
        with h5py.File(selected[0]['source'],'r') as f:
            meta=json.loads(f['data'].attrs['env_args'])
            config=meta['env_kwargs']['ig_config']
            signature={k:config[k] for k in ['scene_id','scene_instance','robot','load_room_types']}
            signature['objects']=[(o['name'],o['filename'],o['class_id']) for o in config['objects']]
            if env is None:
                env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=config,
                    postprocess_visual_obs=False,render=False,render_offscreen=True,use_image_obs=True,
                    image_height=480,image_width=640)
                scene_signature=signature
            else:
                assert signature==scene_signature, 'Only identical scene assets can share a replay renderer'
            renderer=env.env.simulator.renderer
            assert not renderer.optimized
            parts=descriptors(renderer)
            (args.corpus/'masks').mkdir(exist_ok=True)
            for r in selected:
                # Native reset_to advances physics once, which can move an arm
                # under stale motor targets. Annotation must render the stored
                # state itself, without taking a control/physics step.
                state=WorldSaver(exclude_body_ids=[])
                state.deserialize(np.asarray(f['data'][r['demo']]['states'][r['step']]))
                state.restore()
                # Static scene bodies can also have randomized recorded poses.
                # Simulator.sync skips them; force every mesh to the saved pose.
                sync_all(renderer)
                raw=np.asarray(renderer.render_robot_cameras(modes=['rgb'])[0])[...,:3]
                rgb=cv2.resize(raw*255,(120,90),interpolation=cv2.INTER_CUBIC)
                actual=np.asarray(Image.open(args.corpus/r['file']))
                mae=float(np.abs(rgb-actual).mean())
                if mae>12:
                    Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(args.corpus/'alignment_rendered.png')
                    Image.fromarray(actual).save(args.corpus/'alignment_original.png')
                    if not args.reject_unaligned:
                        raise RuntimeError(f'RGB alignment failed {r["id"]}: {mae}')
                    rejected.append(dict(id=r['id'],reason='RGB_alignment_MAE_gt_12',mae=mae))
                    print(json.dumps(dict(rejected=r['id'],mae=mae)),flush=True)
                    continue
                masks=masks_at_state(renderer,parts)
                objects=[]; saved=[]
                for p in parts:
                    mask=masks[p['identity']];y,x=np.where(mask)
                    if not len(x):continue
                    objects.append(dict(class_id=p['cls'],instance_id=p['identity'],
                                        box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
                                        pixels=int(len(x)),tiny=len(x)<4))
                    saved.append(mask)
                np.savez_compressed(args.corpus/'masks'/(r['id']+'.npz'),
                                    masks=np.stack(saved) if saved else np.empty((0,90,120),bool))
                annotations.append(dict(id=r['id'],rgb_alignment_mae=mae,objects=objects,
                                        source='curated_CAD_visible_mesh',review='pending'))
                print(json.dumps(dict(id=r['id'],objects=len(objects),mae=round(mae,2))),flush=True)
    if hasattr(env.env.simulator,'disconnect'):env.env.simulator.disconnect()
    del env,renderer
    path=args.corpus/('annotations_'+(args.task or 'all')+'.json')
    path.write_text(json.dumps(dict(standard='visible_pixels_xyxy_exclusive_v1',
        label_source='privileged_CAD_and_simulator_states_offline_only',
        reference='assistant-curated CAD; visual review pending; not independent human GT',
        annotations=annotations,rejected=rejected),indent=2))
    if args.reject_unaligned and args.task is None:
        assert not (args.corpus/'candidate_manifest.json').exists(), 'Do not replace a filtered corpus in place'
        (args.corpus/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2))
        accepted={a['id'] for a in annotations}
        manifest['records']=[r for r in manifest['records'] if r['id'] in accepted]
        manifest['alignment_filter']=dict(max_rgb_mae=12,candidates=len(rows),rejected=rejected)
        (args.corpus/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--task');p.add_argument('--gpu',type=int,default=5)
    p.add_argument('--reject-unaligned',action='store_true');annotate(p.parse_args())
