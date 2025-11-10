#!/usr/bin/env python3
import argparse
from pathlib import Path

# Skeleton exporter: just creates an empty file to indicate export point.
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_bytes(b'')
print(f"[export_onnx] Wrote placeholder ONNX to {args.out}")
