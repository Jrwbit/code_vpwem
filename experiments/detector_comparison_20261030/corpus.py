"""Export RGB-only pilot data with episode-disjoint detector splits.

Simulation metadata is used only by a separate annotation command. This
manifest and PNGs identify exact source observations without rewriting data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

TASKS = ['table_cleanup_to_dishwasher_expert', 'table_setup_from_dishwasher_expert',
         'table_setup_from_dresser_expert', 'unload_dishwasher_to_dresser_suboptimal']
CLASSES = ['red_bowl', 'dishwasher_door', 'dishwasher_door_handle',
           'dishwasher_rack', 'drawer', 'drawer_handle']


def export(source, output, seed=20261031):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records, episodes = [], []
    for task_index, task in enumerate(TASKS):
        path = source / (task + '.hdf5')
        with h5py.File(path, 'r') as f:
            decode = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
            train, valid = decode(f['mask/train'][:]), decode(f['mask/valid'][:])
            assert not set(train) & set(valid)
            selected = rng.choice(sorted(train), 4, replace=False).tolist()
            splits = {'train': selected[:3], 'dev': selected[3:],
                      'test': rng.choice(sorted(valid), 1, replace=False).tolist()}
            for split, names in splits.items():
                for demo in names:
                    ep = f['data'][demo]
                    n = len(ep['actions'])
                    anchors = np.linspace(0, n-1, 8, dtype=int).tolist()
                    # Eight consecutive physical frames for a causal tracker probe.
                    start = int(.55 * max(n-8, 0))
                    clip = list(range(start, min(n, start+8))) if split != 'train' else []
                    episode_id = task + '/' + demo
                    episodes.append(dict(source=str(path), task=task, demo=demo, split=split,
                                         episode_id=episode_id, length=n, anchors=anchors, clip=clip))
                    for step in sorted(set(anchors + clip)):
                        identity = f'{task_index}_{demo}_{step}'
                        im = np.asarray(ep['obs/rgb'][step])
                        assert im.shape == (90,120,3) and im.dtype == np.uint8
                        filename = 'images/' + identity + '.png'
                        (output/'images').mkdir(exist_ok=True)
                        Image.fromarray(im).save(output / filename)
                        records.append(dict(id=identity, episode_id=episode_id, source=str(path),
                                            task=task, demo=demo, step=step, split=split,
                                            file=filename, anchor=step in anchors,
                                            clip=step in clip, sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    manifest = dict(seed=seed, image_hw=[90,120], classes=CLASSES, episodes=episodes,
                    records=records, protocol='episode_disjoint_rgb_detector_pilot_v1')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for split in ['train','dev','test']:
        rows = [r for r in records if r['split']==split and r['anchor']]
        for page in range((len(rows)+15)//16):
            subset=rows[page*16:(page+1)*16]
            sheet=Image.new('RGB',(1200,4*250),'white'); draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(subset):
                im=Image.open(output/r['file']).resize((288,216),Image.Resampling.NEAREST)
                x=(i%4)*300;y=(i//4)*250;sheet.paste(im,(x,y));draw.text((x,y+216),r['id'],fill='black')
            sheet.save(output/f'{split}_page{page}.png')
    print(json.dumps(dict(frames=len(records), episodes=len(episodes),
                          splits={s:sum(r['split']==s for r in records) for s in splits})),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    export(args.source,args.output)
