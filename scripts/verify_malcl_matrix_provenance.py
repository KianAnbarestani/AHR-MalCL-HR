#!/usr/bin/env python3
"""Verify Path-D MalCL preserved-output-to-normalized-matrix chains."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from v15_5_checks import check_malcl_provenance
if __name__=="__main__":
    check_malcl_provenance()
    print("verified 20 preserved MalCL CSV-to-matrix chains; runner/postprocessor remains not recovered")
