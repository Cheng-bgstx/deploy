#!/usr/bin/env python3
"""Render X2 using the shared integrated sim2sim recorder (actual physics beside reference)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.oli.render_sim2sim import main

if __name__ == '__main__':
    main(default_robot='x2')
