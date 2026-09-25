# K-sensitivity hardware provenance

All 30 new rows were executed on Tesla T4 with Python 3.12.13 and PyTorch 2.11.0+cu128. Historical MAIN and LOW rows do not serialize a complete per-run GPU/environment record; their provenance is therefore reported as historical/not serialized. K level is partially confounded with provenance (all new levels are T4, while historical portions are older evidence), but no rows were discarded. New-vs-historical boundaries and fidelity classifications are explicit in the canonical index.
