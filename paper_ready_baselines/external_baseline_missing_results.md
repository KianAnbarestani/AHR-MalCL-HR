# External Baseline Missing Results

Input root: `/home/kian/projects/shz_uni/paper-works/AHR-MalCL/result_external_baselines`.
External seed-level rows found: 30.
Parse errors: 0.

## Expected Comparisons

- ER EMBER K=100: present (5 seed rows).
- DERPP EMBER K=100: present (5 seed rows).
- Joint EMBER K=offline: present (5 seed rows).
- ER AZ-Class K=200: present (5 seed rows).
- DERPP AZ-Class K=200: present (5 seed rows).
- Joint AZ-Class K=offline: present (5 seed rows).

## Notes

- Paired tests require matching seeds and metric availability for AHR-MalCL-HR and the external baseline.
- Joint is compared by dataset and seed only; ER and DERPP are compared by dataset, K, and seed.
- Keep external outputs under `result_external_baselines/`; do not place them under trusted `result/`.
