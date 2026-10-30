"""Fail if manifests differ outside mode/output; compare common initial weights."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    manifests = [json.loads((p/'object_protocol.json').read_text()) for p in args.runs]
    initializations = [json.loads((p/'initialization.json').read_text()) for p in args.runs]
    if any(v != initializations[0] for v in initializations):
        raise ValueError('Arm initial weights or parameter counts differ')
    first = manifests[0]
    for p, m in zip(args.runs, manifests):
        if (m['comparison_contract'] != first['comparison_contract']
                or m['perception'] != first['perception']
                or m['source_sha256'] != first['source_sha256']
                or m['comparison_sha256'] != first['comparison_sha256']):
            raise ValueError('Uncontrolled difference in '+str(p))
    print(json.dumps(dict(verified=True,modes=[m['mode'] for m in manifests],
                          comparison_sha256=first['comparison_sha256'],
                          initialization=initializations[0]),indent=2))


if __name__ == '__main__':
    main()
