# V15.5 Standard-A runner provenance

The source handoff is included at runners/v15_5_standard_a. The documented source-hash procedure hashes each sorted Python file in v15_5_runner, followed by launch_v15_5_standard_a.py; each file contributes its basename bytes followed by its raw bytes.

The resulting combined SHA-256 is:

    e239660453035ba93abe2a1de2517044fd492569f7da7c5714b06d900269f470

This is the runner identity recorded for the newly executed Standard-A ablation rows. The included validation report records 40/40 rows validated. The handoff does not cover Full-Hybrid historical reuse rows or the K-sensitivity training runs. The repository includes analysis outputs for both studies, with their boundaries described in RELEASE_METADATA_V15_5.md.
