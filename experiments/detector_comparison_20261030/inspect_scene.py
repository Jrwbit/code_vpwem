"""Inspect annotation-only renderer geometry; never used by online perception."""
import json
from pathlib import Path
import h5py
import egl_probe
# EnvGibson otherwise chooses the first EGL GPU, independently of CUDA visibility.
egl_probe.get_available_devices = lambda: [5]
from robomimic.envs.env_ig_momart import EnvGibsonMOMART
with h5py.File('/home/jrw/vpwem_data/momart/unload_dishwasher_to_dresser_expert.hdf5','r') as f:
    meta=json.loads(f['data'].attrs['env_args'])
    env=EnvGibsonMOMART(env_name=meta['env_name'],ig_config=meta['env_kwargs']['ig_config'],postprocess_visual_obs=False,render=False,render_offscreen=True,use_image_obs=True,image_height=480,image_width=640)
    renderer=env.env.simulator.renderer
    rows=[]
    for inst in renderer.instances:
        objects=getattr(inst,'objects',[getattr(inst,'object',None)])
        files=[getattr(o,'filename',None) for o in objects]
        if any(any(s in str(name).lower() for s in ['dishwasher','drawer','dresser','bowl']) for name in files):
            rows.append(dict(id=inst.id,body=inst.pybullet_uuid,class_id=inst.class_id,links=getattr(inst,'link_ids',None),files=files))
    print('ANNOTATION_GEOMETRY='+json.dumps(rows),flush=True)
    Path('/home/jrw/vpwem_outputs/momart_detector_comparison_20261030/geometry.json').write_text(json.dumps(rows,indent=2))
