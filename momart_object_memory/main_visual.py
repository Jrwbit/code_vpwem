"""Archived red-bowl main-image path, independent of object-memory perception."""
from .perception import file_sha256


def main_visual_contract(config):
    spec = config.main_visual
    if spec.version != 'archived_red_bowl_v1':
        raise ValueError('Unsupported main visual path')
    if file_sha256(spec.precision_bundle) != spec.precision_bundle_sha256:
        raise ValueError('Main visual precision bundle checksum differs')
    return dict(version=spec.version, precision_bundle_sha256=spec.precision_bundle_sha256,
                offline='unchanged_BoxedNoHistoryDataset', online='PrecisionSequenceDetector',
                color=[255, 0, 0], width=1, object_boxes_paint_main_rgb=False)


def build_main_boxer(config, device, num_envs, bundle_override=None):
    from momart_bbox_memory.precision_detector import PrecisionSequenceDetector
    from momart_boxed_no_history.painting import OnlineBoxedRGB
    main_visual_contract(config)
    bundle = bundle_override or config.main_visual.precision_bundle
    if file_sha256(bundle) != config.main_visual.precision_bundle_sha256:
        raise ValueError('Online main visual bundle differs from archived baseline')
    detector = PrecisionSequenceDetector(bundle, device=device, num_streams=num_envs)
    return OnlineBoxedRGB(detector, batch_size=num_envs * int(config.obs_steps))
