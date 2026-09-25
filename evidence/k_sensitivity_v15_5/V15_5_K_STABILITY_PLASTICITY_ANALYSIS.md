# K stability–plasticity analysis

Metrics are recomputed under the locked native V15.5 implementation; values below are LOW → MAIN → HIGH means.

EMBER: Final TM 73.1737% → 78.7527% → 80.5537%; forgetting 19.1319% → 9.2235% → 6.0702%; acquisition 90.5664% → 87.0872% → 85.8073%; final-old 71.4059% → 77.9204% → 79.8642%; BWT -19.1319% → -9.1680% → -5.7789%.

AZ-Class: Final TM 70.0742% → 73.5104% → 76.4354%; forgetting 14.7021% → 9.1591% → 6.1872%; acquisition 83.4347% → 81.5570% → 81.0509%; final-old 68.8665% → 72.8666% → 76.0426%; BWT -14.6965% → -8.8513% → -5.0771%.

Increasing K changes both anchor capacity and native classifier exposure. The trend CSV reports seed-level monotonicity and slopes. The correct interpretation is native-protocol K sensitivity, not fixed-compute memory causality; forgetting is read jointly with acquisition, final-old accuracy, BWT, and final TM.
