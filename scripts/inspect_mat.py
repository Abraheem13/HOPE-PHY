#!/usr/bin/env python
"""Print keys/shapes/dtypes of a .mat file to pin data_key in configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hope_phy.data.llm4cp_dataset import _load_mat

d = _load_mat(Path(sys.argv[1]))
for k, v in d.items():
    print(f"{k:30s} shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
