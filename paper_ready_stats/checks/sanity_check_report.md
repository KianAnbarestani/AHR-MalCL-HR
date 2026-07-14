# Sanity Check Report

- Run rows checked: 110
- Aggregate rows checked: 37
- Critical failures: 0
- Warnings: 18
- Expected exclusions: 1
- Legacy/config mismatch rows: 10
- Final main HR table is safe to use: YES

## Critical Failures

- None.

## Warnings

- inconsistent_config_report: az_class ablation classifier_real_only K=100 seed= Multiple config hashes in group: ['67bc0d92e8af9ceb', '809603f015c13616', 'b592b71d97840a6a', 'c1b1f3cd155478a4', 'e36ff0eb89867a56']
- inconsistent_config_report: az_class ablation hybrid_diversity_buffer K=100 seed= Multiple config hashes in group: ['67d44792e8e55fd9', 'a297cf4e848cc134', 'a2cc58548e215cd3', 'cc5f4b33d7d48e43', 'dcb823370af3de5c']
- inconsistent_config_report: az_class ablation hybrid_random_buffer K=100 seed= Multiple config hashes in group: ['617976a3a581c3aa', '87e684aa3261bd8a', 'cd874afbae7f6aae', 'e9de735d182ae120', 'eed388777e45a557']
- inconsistent_config_report: az_class ablation malcl_like K=100 seed= Multiple config hashes in group: ['10ef5173dc2e69a5', '1ac5f6afebce5d50', '421021a4155c2e8f', '837f55db4411f490', 'aa4e7816df4cb85a']
- inconsistent_config_report: az_class ablation plus_kd K=100 seed= Multiple config hashes in group: ['09e23f00641f0639', '4277eb0fbce0c439', '43fe61fafb6d5c00', '8d7ba547b69f0032', '9773d02f8039a3b6']
- inconsistent_config_report: az_class ablation real_buffer_only K=100 seed= Multiple config hashes in group: ['26885a3195467dc1', '72b22a69e9f6bed0', '7be651e41ec314d2', '9e20f54c60cc032a', 'efc714e9659b9191']
- inconsistent_config_report: az_class ablation wgan_projection_generated_only K=100 seed= Multiple config hashes in group: ['03d90b1155fe7740', '5acfeab50caf4cea', '991e1fcabaa489fc', 'a841383e22e69b3c', 'c08a2253be174989']
- inconsistent_config_report: az_class final_best_performance hybrid_random_buffer K=200 seed= Multiple config hashes in group: ['49b4669f01655cf6', '57184a79aac643f7', '80bedf77e9011419', '8695a18d8f5c3e78', '9ce8d1a5fef303f8']
- inconsistent_config_report: az_class final_memory_efficient hybrid_random_buffer K=100 seed= Multiple config hashes in group: ['0939296008275640', '0ceab2d4a153e96a', 'afab2921eb68f91e', 'ec1d834d4c0bfa09', 'f5c7ac61b8b0c518']
- inconsistent_config_report: ember ablation classifier_real_only K=25 seed= Multiple config hashes in group: ['01c8789e575a46a0', '2cabf0134aaa335d', '2d6b3df5639afb72', '3d3a9b228af577d6', 'ee634eb8e34bbf00']
- inconsistent_config_report: ember ablation hybrid_diversity_buffer K=25 seed= Multiple config hashes in group: ['2658528c6bfae862', '3fec623a815747fe', '708fafef67ca2502', 'acd5e88452f2b130', 'c9170155c8af0343']
- inconsistent_config_report: ember ablation hybrid_random_buffer K=25 seed= Multiple config hashes in group: ['646324bfa5dd3f2d', '750b2fc74d166cbe', '9f79673513beaa92', 'a04630b82264b268', 'e072794a348b3c7a']
- inconsistent_config_report: ember ablation malcl_like K=25 seed= Multiple config hashes in group: ['348ba559b975a018', '8172ecb7a2b94c2b', '832c965c1c924640', '83e86a7e4b0c12a3', 'cf76f95e361b7c26']
- inconsistent_config_report: ember ablation plus_kd K=25 seed= Multiple config hashes in group: ['10cc13baeaac1370', '5b42a0366f7ac0dc', '678db8987811cfaa', '871b240429da9141', 'c1f5cfe247fc5a64']
- inconsistent_config_report: ember ablation real_buffer_only K=25 seed= Multiple config hashes in group: ['4d8288ad2e207d1b', 'ae43d25910cc130c', 'c8ef43d6389b9cf9', 'e569ee0afd6242ed', 'f9042b865a81473f']
- inconsistent_config_report: ember ablation wgan_projection_generated_only K=25 seed= Multiple config hashes in group: ['0a746ec907cdb67d', '4766f0a7095e2603', '7d16e8afbde6d5cc', 'b895a2b890cc8ec8', 'ed30958765a2400f']
- inconsistent_config_report: ember final_best_performance hybrid_random_buffer K=100 seed= Multiple config hashes in group: ['79af7404124fe4f9', 'a05cb5c5f44662c2', 'bfee1d37291f4800', 'c4a7e0db8f5f7261', 'f8d951075b702435']
- inconsistent_config_report: ember final_memory_efficient hybrid_random_buffer K=25 seed= Multiple config hashes in group: ['0b6649414269c805', 'b90ba2295604d901', 'c351e99718a6b8a3', 'c99be68d4932cb01', 'cf0fea57de6cb0e4']

## Expected Exclusions

- stray incomplete EMBER critic2 memory-efficient seed file; not part of declared final result set; source=result/final-run/EMBER/memory-efficient setting/seed=42/full-result.json

## Legacy / Config Mismatch Rows

- az_class final_memory_efficient hybrid_random_buffer K=100 seed=42: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=43: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=44: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=45: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- az_class final_memory_efficient hybrid_random_buffer K=100 seed=46: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- ember final_memory_efficient hybrid_random_buffer K=25 seed=42: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- ember final_memory_efficient hybrid_random_buffer K=25 seed=43: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- ember final_memory_efficient hybrid_random_buffer K=25 seed=44: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- ember final_memory_efficient hybrid_random_buffer K=25 seed=45: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method
- ember final_memory_efficient hybrid_random_buffer K=25 seed=46: memory-efficient file uses full-complexity flags; not the final simplified AHR-MalCL-HR method

## Final Decision

- Main final HR results are safe to use.

## Notes

- Missing metrics are recorded in `raw/missing_metrics_report.csv` and summarized in the CSV report.
- Pairing compatibility for ablation tests is written to `checks/pairing_matrix.csv`.
