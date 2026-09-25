# Retention decomposition

Conventional forgetting is never interpreted alone. Lower forgetting can be caused by genuinely stronger final retention, a lower acquisition peak, or both. The paired comparisons below jointly report acquisition, final old-task accuracy, BWT, and forgetting; values are benefit-aligned pp for Full Hybrid minus the named comparator.

| Dataset | Comparator | Final TM | Forgetting | BWT | Acquisition | Final old-task accuracy | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| EMBER | Current-only | +69.59 | +87.02 | +87.09 | −9.59 | +77.65 | Very large final-retention gain despite lower acquisition. |
| EMBER | Anchor-only | +0.39 | −1.94 | −2.60 | +2.75 | +0.29 | Full trades a small acquisition gain for worse forgetting/BWT; final accuracy is tied. |
| EMBER | Generated-only | +37.85 | +50.17 | +50.24 | −7.82 | +42.50 | Anchors in Full produce much stronger final retention, despite lower acquisition. |
| AZ-Class | Current-only | +64.53 | +87.41 | +87.80 | −15.28 | +72.84 | Very large final-retention gain despite lower acquisition. |
| AZ-Class | Anchor-only | +0.17 | −1.63 | −2.48 | +2.43 | −0.02 | Same stability–plasticity tradeoff; no final-accuracy advantage for Full. |
| AZ-Class | Generated-only | +44.29 | +62.72 | +63.11 | −13.09 | +50.29 | Anchors in Full produce much stronger final retention, despite lower acquisition. |

The decisive comparison is therefore not “which method has the lowest forgetting?” alone. Anchor-only has the most favorable forgetting/BWT relative to Full Hybrid, whereas Full Hybrid versus Current-only or Generated-only yields much higher final-old and final task-macro accuracy. This supports anchors as the primary retention mechanism and characterizes generation as an acquisition-oriented addition with a retention cost when added to anchors.
