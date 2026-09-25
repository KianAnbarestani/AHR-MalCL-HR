# Scientific interpretation of the final 80-row Standard-A ablation

## Scope and inference boundary

This is `PROTOCOL_MATCHED_STANDARD_A`: the original V15.5 native protocol runs three class-balanced epochs over component-dependent classifier pools. Therefore every Standard-A contrast combines the replay/content choice with the native update-budget change caused by pool size. These are algorithm-level effects under the V15.5 protocol, not fixed-compute causal component effects. The V16 evidence remains separate and is not pooled here.

## What the data support

All quantities below are paired means in percentage points (pp), calculated from the common harmonized task-matrix implementation. The primary final-TM Wilcoxon values are Holm-adjusted within each dataset × metric family.

- **Full Hybrid versus Current-only — SUPPORTED.** Final task-macro accuracy is higher by +69.59 pp on EMBER and +64.53 pp on AZ-Class (both Holm-adjusted exact Wilcoxon p=0.005859). Final old-task accuracy, BWT, and benefit-aligned forgetting all improve in every seed on both datasets. The cost is lower acquisition: −9.59 pp (EMBER) and −15.28 pp (AZ-Class). This is a strong stability–plasticity tradeoff that decisively favors replay for final retention.
- **Anchor-only versus Current-only — SUPPORTED.** Final TM improves by +69.20 pp and +64.36 pp, respectively (both exact p=0.001953), while retention improves strongly and acquisition is lower. Anchors are the dominant retention component in this native protocol.
- **Generated-only versus Current-only — SUPPORTED.** Final TM improves by +31.74 pp on EMBER and +20.25 pp on AZ-Class (both exact p=0.001953), with better final-old accuracy and lower forgetting/BWT, but with lower acquisition. Generation is beneficial compared with no replay, though substantially weaker than anchors.
- **Full Hybrid versus Generated-only — SUPPORTED.** Adding anchors to generated replay increases final TM by +37.85 pp (EMBER) and +44.29 pp (AZ-Class), with large, consistent retention gains (all final-TM Holm-adjusted p=0.005859). Thus anchors materially improve generated replay.
- **Full Hybrid versus Anchor-only / generation added to Anchor-only — MIXED.** Final TM differences are only +0.39 pp and +0.17 pp and are non-significant (Holm p=0.492188 and 0.556641). Full Hybrid has significantly higher acquisition (+2.75 and +2.43 pp), but Anchor-only has significantly lower forgetting (Full-minus-Anchor benefit difference −1.94 and −1.63 pp) and better BWT (−2.60 and −2.48 pp). Final-old accuracy is effectively tied. Generation therefore does not establish an overall final-retention advantage beyond anchors; it changes the stability–plasticity balance.

## Factorial decomposition

The file `V15_5_STANDARD_A_NATIVE_PROTOCOL_FACTORIAL_EFFECTS.csv` is explicitly a `STANDARD_A_NATIVE_PROTOCOL_FACTORIAL_DECOMPOSITION`, not a pure causal decomposition. Anchor main effects on final TM are +53.52 pp (EMBER) and +54.32 pp (AZ-Class), compared with generation main effects of +16.06 and +10.21 pp. The benefit-aligned anchor×generation interaction is negative for final TM (−31.35 and −20.08 pp), forgetting (−38.80 and −26.32 pp), BWT (−39.45 and −27.17 pp), and final-old accuracy (−34.86 and −22.56 pp), while positive for acquisition (+4.52 and +4.62 pp). Its appropriate label is **ANTAGONISTIC** at the algorithm level, not “synergy.”

## Bottom line

The completed Standard-A ablation supports the Anchored Hybrid Replay design over Current-only and Generated-only, and robustly identifies anchors as the stronger retention component. It does **not** show that Full Hybrid is universally superior to Anchor-only: Anchor-only is at least as good for final accuracy and is better on conventional forgetting/BWT, while Full Hybrid provides modestly greater acquisition. This is a scientifically useful, non-favorable qualification that must remain visible in any later manuscript discussion.

## Recommendation

`PROCEED_WITH_30_RUN_K_SENSITIVITY` after independent review. No K-sensitivity runs were launched and no manuscript text, tables, figures, or experimental outputs were edited.
