#!/usr/bin/env python3
"""Export Proteus for X2, plus a simulation YAML that uses measured world root position."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.export_proteus_onnx import export_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proteus-root', type=Path, default=Path('../Proteus-wamseq'))
    p.add_argument('--release', default='proteus_wamseq_62')
    args = p.parse_args()
    folder = export_policy('lingxi_x2', proteus_root=args.proteus_root, robot='x2', release=args.release)
    path = folder / 'policy-sim.yaml'
    print(folder / 'policy.yaml')
    print(path)


if __name__ == '__main__':
    main()
