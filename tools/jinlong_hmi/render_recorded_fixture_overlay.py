#!/usr/bin/env python3
"""Overlay real recorded fixture axes and independent stud pixels on source RGB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/diagnostics/actual-terminal-alignment-20261023'
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_box_points(points_mm, camera_T_box, intrinsic):
    T = np.asarray(camera_T_box)
    points_camera = np.asarray(points_mm) / 1000 @ T[:3, :3].T + T[:3, 3]
    K = np.asarray(intrinsic['rgb_intrinsic']).reshape(3, 3)
    d = intrinsic['dist_coeffs']
    distortion = np.asarray([d[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3')])
    pixels, _ = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3), K, distortion)
    return pixels.reshape(-1, 2)


def main(args):
    audit = json.loads(args.audit.read_text())
    fig = plt.figure(figsize=(17, 16), layout='constrained')
    grid = fig.add_gridspec(4, 2, width_ratios=[1.6, 1.], height_ratios=[1, 1, 1, .40])
    receipt = {'audit': str(args.audit), 'audit_sha256': sha(args.audit),
               'renderer_sha256': sha(__file__), 'controller_writes': False, 'panels': []}
    rows = []
    for index, fid in enumerate(['JL100', 'JL132', 'JM160']):
        if fid == 'JL132':
            result = audit['ROKAE132_same_task_direction_check']
            vision = json.loads(Path(result['result_source']).read_text())
            image_path = Path(result['result_source']).parent / 'latest.jpg'
            pixels = np.asarray(result['observed_tip_pixels'])
            check = result['sensitivity'][1]
            camera_T_box = vision['camera_T_box']
            intrinsic = vision['live']['intrinsics']
            provenance = '珞石实际到位法兰 + 现场 −90° 安装'
            task = result['task_id']
            crop = [1300, 230, 1770, 850]
        else:
            result = audit['cases'][fid]
            evidence = json.loads(Path(result['evidence_path']).read_text())
            image_path = Path(result['evidence_path']).parent / 'actual_box_annotated.jpg'
            pixels = np.asarray(result['observed_stud_pixels'])
            check = result['sensitivity'][1]
            camera_T_box = evidence['camera_T_box']
            intrinsic = json.loads((image_path.parent / 'camera_intrinsics.json').read_text())
            provenance = 'UR 历史实际到位 TCP'
            task = result['task_id']
            crop = ([1100, 350, 1450, 820] if fid == 'JL100' else [1410, 280, 2240, 1110])
        predicted = project_box_points(check['socket_axes_at_plane_mm'], camera_T_box, intrinsic)
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        matching = check['matching']
        for col in range(2):
            ax = fig.add_subplot(grid[index, col])
            ax.imshow(rgb)
            ax.scatter(*pixels.T, s=95 if col else 70, facecolors='none',
                       edgecolors='#00d8ff', linewidths=1.7, label='独立手点：6根真实接线柱')
            ax.scatter(*predicted.T, s=100 if col else 70, c='#ff7a18',
                       marker='x', linewidths=1.7, label='STEP套筒轴 + 实际机器人反馈')
            for number, xy in enumerate(pixels):
                ax.annotate(str(number), xy, xytext=(8, -4), textcoords='offset points',
                            color='#00e5ff', fontsize=10, weight='bold',
                            bbox={'facecolor': '#111111', 'alpha': .65, 'edgecolor': 'none', 'pad': .6})
            if col:
                ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
                ax.set_title('同张图近景：圆圈为实物，橙叉为套筒轴投影', fontproperties=FONT, fontsize=11)
            else:
                ax.add_patch(Rectangle(crop[:2], crop[2]-crop[0], crop[3]-crop[1],
                                       fill=False, edgecolor='#ffc94f', linewidth=1.2))
                ax.set_title(f'{fid}  |  {provenance}\n任务 {task}；原图完整保留电机、轴端及电缆口参照',
                             fontproperties=FONT, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        rows.append([fid, '实反馈→U1/V1/W1列', f"{matching['rms_mm']:.2f}",
                     f"{matching['max_mm']:.2f}", f"{matching['row_angle_difference_deg']:.2f}°"])
        receipt['panels'].append({'fixture': fid, 'source_image': str(image_path),
                                 'image_sha256': sha(image_path), 'task': task,
                                 'provenance': provenance, 'observed_pixels': pixels.tolist(),
                                 'projected_socket_pixels': predicted.tolist(), 'matching': matching})
    table_ax = fig.add_subplot(grid[3, :]); table_ax.axis('off')
    table = table_ax.table(cellText=rows,
                          colLabels=['型号', '匹配到的真实端子', '横向 RMS（mm）', '最大残差（mm）', '排向差'],
                          cellLoc='center', loc='center', colWidths=[.12, .28, .19, .19, .16])
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.get_text().set_fontproperties(FONT)
        if row == 0:
            cell.set_facecolor('#e4edf5')
    table_ax.set_title('蓝圈：独立手点6柱；橙叉：STEP套筒轴按真实记录回投；0、1、2为未短接列\n'
                       '数值在“柱顶距盒框平面8 mm”的假定平面计算；手点约3 px不确定度，不能当成插接精度验收。',
                       fontproperties=FONT, fontsize=10, pad=5)
    fig.suptitle('真实历史路径与实际接线柱方向对照\n'
                 '本图不是新工位仿真终点：只检查盒内配合方向，不证明盒体与电机底脚／轴的拼接方向已校准。',
                 fontproperties=FONT, fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)
    receipt['image_sha256'] = sha(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=BASE / 'archive/mating-audit/recorded_fixture_mating.json')
    parser.add_argument('--output', type=Path, default=BASE / 'archive/mating-audit/真实路径与端子方向对照.png')
    main(parser.parse_args())
