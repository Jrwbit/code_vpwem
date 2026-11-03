"""Bind fixed-clock public sensor semantics before prospective confirmation."""
KEYS=['protocol','grid_stride','context_dim','tasks','feature_checkpoint_sha256','urdf_sha256','tf32']


def public_history_contract(metadata):
    value={k:metadata[k] for k in KEYS}
    if value['protocol']!='fixed_clock_main_context_v1' or value['grid_stride']!=8 or value['context_dim']!=294 or value['tf32'] is not False:
        raise ValueError('Unsupported public history contract')
    return value


def validate_history_contract(metadata,frozen,manifest_sha256):
    if public_history_contract(metadata)!=frozen:raise ValueError('Public history sensor/encoder contract differs from frozen selection')
    if metadata.get('corpus_manifest_sha256')!=manifest_sha256:raise ValueError('Public history belongs to a different confirmation manifest')
