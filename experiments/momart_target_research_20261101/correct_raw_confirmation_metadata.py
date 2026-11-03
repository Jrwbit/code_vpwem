"""Archive and remove inherited processing claims from an unprocessed raw reserve.

Membership, sensor/image hashes, and exclusions are preserved. This migration
never renders, labels, selects, scores or resamples a reserved episode.
"""
import argparse
import hashlib
import json
from pathlib import Path


STALE=['alignment_filter','annotation_version','source_manifest_sha256','selection_restriction']


def cohort_sha(doc):
    keys=['episodes','records','excluded_episodes','image_hw','classes','id_prefix','source_pool','episodes_per_task','exclusions_sha256']
    return hashlib.sha256(json.dumps({k:doc[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(a):
    path=a.corpus/'manifest.json';backup=a.corpus/'manifest.before_metadata_correction.json';audit=a.corpus/'metadata_correction.json'
    if backup.exists() or audit.exists():raise FileExistsError('Metadata correction already archived; inspect without restarting')
    if any((a.corpus/name).exists() for name in ['annotations_all.json','candidate_manifest.json']):raise ValueError('Only an unprocessed raw reserve can be corrected')
    raw=path.read_bytes();before=hashlib.sha256(raw).hexdigest()
    if before!=a.expected_sha256:raise ValueError('Unexpected raw reserve manifest')
    old=json.loads(raw);new=dict(old);removed={k:new.pop(k) for k in STALE if k in new}
    assert cohort_sha(old)==cohort_sha(new)
    backup.write_bytes(raw);path.write_text(json.dumps(new,indent=2));after=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(before_manifest_sha256=before,after_manifest_sha256=after,cohort_sha256=cohort_sha(new),removed_fields=list(removed),
        episodes=len(new['episodes']),frames=len(new['records']),membership_and_image_hashes_unchanged=True,
        note='Metadata-only correction; archived original manifest retained. No reserve annotations, predictions, new images or resampling.')
    audit.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--expected-sha256',required=True);main(p.parse_args())
