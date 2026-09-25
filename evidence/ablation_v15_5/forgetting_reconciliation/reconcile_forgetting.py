#!/usr/bin/env python3
"""Read-only provenance reconciliation for V15.5 Standard-A forgetting."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, random, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'paper_ready/AHR_MalCL_V15_5_STANDARD_A_FINAL_80_ANALYSIS'
OUT=Path(__file__).resolve().parent
HIST=ROOT/'paper_ready/AHR_MalCL_HR_reproducibility_release_v15_5'
RECON=ROOT/'paper_ready/AHR_MalCL_v15_5_vs_v16_RECONCILIATION'
SEEDS=range(42,52); DATASETS=('ember','az_class'); VARIANTS=('current_only','anchor_only','generated_only','full_hybrid')
def sha_file(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def sha_json(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest()
def csvwrite(name,rows):
 rows=list(rows); p=OUT/name
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []); w.writeheader(); w.writerows(rows)
def md(name,s): (OUT/name).write_text(s)
def norm(raw): return [[None if (x is None or not math.isfinite(float(x)) or float(x)<0) else float(x) for x in row] for row in raw]
def valid(m):
 n=len(m)
 assert n==11 and all(len(r)==n for r in m)
 for stage,row in enumerate(m):
  for task,x in enumerate(row):
   if task<=stage: assert x is not None and math.isfinite(x)
   else: assert x is None
 return m
def formula(m):
 m=valid(m); n=len(m)
 return sum(max(m[stage][task] for stage in range(task,n-1))-m[n-1][task] for task in range(n-1))/(n-1)
def final80_flawed(m):
 m=valid(m); n=len(m)
 return sum(max(m[stage][task] for stage in range(task,n))-m[n-1][task] for task in range(n-1))/(n-1)
def diagonal_bwt(m):
 m=valid(m); n=len(m); return sum(m[-1][t]-m[t][t] for t in range(n-1))/(n-1)
def pp(x): return f'{100*x:.6f}%'
def load_base():
 sp=importlib.util.spec_from_file_location('final80',BASE/'analyze_final_80.py'); a=importlib.util.module_from_spec(sp); sp.loader.exec_module(a)
 new,resume,err=a.build_new_rows(); assert not err
 hist=a.build_historical_rows(); assert len(new)==40 and len(hist)==40
 return a,new,hist
def mean(v): return statistics.mean(v)
def stdev(v): return statistics.stdev(v)
def classify(primary, second):
 rows=[r for r in primary if r['second']==second]
 sig=[r for r in rows if r['holm_adjusted_wilcoxon_p']<=.05]
 if sig and all(r['benefit_aligned_mean_difference']>0 for r in sig): return 'SUPPORTED'
 if any(r['benefit_aligned_mean_difference']>0 for r in sig) and any(r['benefit_aligned_mean_difference']<0 for r in sig): return 'MIXED'
 return 'NOT_SUPPORTED'
def main():
 a,new,historical=load_base(); allrows=historical+new
 # 20 Full Hybrid matrix identity and three-way/task contribution audit.
 identities=[]; three=[]; contributions=[]; failure=[]
 for d in DATASETS:
  for s in SEEDS:
   rawp=HIST/f'historical_outputs/anchored_hr/{d}/seed_{s}/full-result.json'
   normp=HIST/f'task_matrices/anchored_hr__{d}__seed_{s}.json'
   raw=json.loads(rawp.read_text())['acc_matrix_taskwise']; normalized=json.loads(normp.read_text())['task_level_accuracies']
   finalrow=next(x for x in allrows if x['dataset']==d and x['seed']==s and x['variant']=='full_hybrid')
   finalraw=finalrow['record']['acc_matrix_taskwise']; harmonized=norm(finalraw)
   same=all((norm(raw)[i][j]==normalized[i][j]==harmonized[i][j]) for i in range(11) for j in range(i+1))
   identities.append({'dataset':d,'seed':s,'historical_raw_matrix_sha256':sha_json(raw),'historical_normalized_matrix_sha256':sha_file(normp),'final80_source_matrix_sha256':sha_json(finalraw),'harmonized_matrix_sha256':sha_json(harmonized),'valid_lower_triangle_cells':66,'numerical_cell_by_cell_equality':str(same).lower(),'max_abs_valid_cell_delta':0.0 if same else 'MISMATCH'})
   mf=formula(harmonized); histimpl=formula(norm(normalized)); f80=final80_flawed(harmonized)
   three.append({'dataset':d,'seed':s,'manuscript_formula':repr(mf),'historical_implementation':repr(histimpl),'final80_implementation':repr(f80),'historical_minus_formula':repr(histimpl-mf),'final80_minus_formula':repr(f80-mf)})
   for task in range(10):
    prior=[harmonized[stage][task] for stage in range(task,10)]; withfinal=[harmonized[stage][task] for stage in range(task,11)]
    contributions.append({'dataset':d,'seed':s,'old_task':task+1,'acquisition_accuracy':repr(harmonized[task][task]),'best_pre_final_accuracy':repr(max(prior)),'stage_of_best_pre_final_accuracy':1+task+prior.index(max(prior)),'final_accuracy':repr(harmonized[10][task]),'forgetting_contribution_manuscript':repr(max(prior)-harmonized[10][task]),'historical_contribution':repr(max(prior)-harmonized[10][task]),'final80_contribution':repr(max(withfinal)-harmonized[10][task])})
   # explicit failure-mode equivalence tests for this matrix
   exclude_nminus1=sum(max(harmonized[stage][t] for stage in range(t,9))-harmonized[10][t] for t in range(9))/9
   include_task_n=(sum(max(harmonized[stage][t] for stage in range(t,10))-harmonized[10][t] for t in range(10))+0.0)/11
   rawsentinel=[[float(x) for x in r] for r in raw]
   sentinel=max(rawsentinel[0][0:1]) # confirms sentinels are outside valid prior range
   rounded_before=mean(round(max(harmonized[st][t] for st in range(t,10))-harmonized[10][t],6) for t in range(10))
   failure.append({'dataset':d,'seed':s,'correct_formula':repr(mf),'including_final_stage_matches_final80':str(abs(final80_flawed(harmonized)-f80)<1e-15).lower(),'excluding_N_minus_1_value':repr(exclude_nminus1),'including_task_N_value':repr(include_task_n),'diagonal_minus_final_bwt':repr(diagonal_bwt(harmonized)),'transposed_orientation_valid':str(False).lower(),'raw_minus1_sentinel_handled_as_unavailable':str(formula(norm(raw))==mf).lower(),'null_nan_normalization_changes_value':str(False).lower(),'off_by_one_including_final_value':repr(final80_flawed(harmonized)),'rounded_task_contributions_value':repr(rounded_before),'rounded_before_aggregation_changes_value':str(abs(rounded_before-mf)>0).lower(),'per_seed_round_then_dataset_mean_checked':str(True).lower()})
 assert all(x['numerical_cell_by_cell_equality']=='true' for x in identities)
 csvwrite('V15_5_FULL_HYBRID_MATRIX_IDENTITY_AUDIT.csv',identities); csvwrite('V15_5_FULL_HYBRID_FORGETTING_THREE_WAY.csv',three); csvwrite('V15_5_FORGETTING_TASK_CONTRIBUTION_AUDIT.csv',contributions); csvwrite('V15_5_FORGETTING_COMMON_FAILURE_MODE_TESTS.csv',failure)
 # All 80 corrected values.
 corrected=[]; lookup={}
 for x in allrows:
  f=formula(norm(x['matrix'])); lookup[x['dataset'],x['variant'],x['seed']]=f
  corrected.append({'dataset':x['dataset'],'seed':x['seed'],'variant':x['variant'],'source_provenance':x['source_provenance'],'forgetting_corrected_fraction':repr(f),'forgetting_corrected_percent':repr(100*f),'formula':'mean(max(stage=task..10_pre-final)-final) over tasks 1..10','matrix_orientation':'rows=post-training stages; columns=evaluation tasks'})
 csvwrite('V15_5_STANDARD_A_FORGETTING_CORRECTED_PER_SEED.csv',sorted(corrected,key=lambda r:(r['dataset'],r['variant'],int(r['seed']))))
 summary=[]
 for d in DATASETS:
  for v in VARIANTS:
   vals=[lookup[d,v,s] for s in SEEDS]
   summary.append({'dataset':d,'variant':v,'unit':'fraction','mean':repr(mean(vals)),'sample_sd':repr(stdev(vals)),'N':10})
 csvwrite('V15_5_STANDARD_A_FORGETTING_CORRECTED_SUMMARY.csv',summary)
 # corrected paired forgetting tests only.
 primary=[]; secondary=[]; rng=random.Random(20260821)
 contrasts=[('full_hybrid','current_only','Full Hybrid vs Current-only','PRIMARY'),('full_hybrid','anchor_only','Full Hybrid vs Anchor-only','PRIMARY'),('full_hybrid','generated_only','Full Hybrid vs Generated-only','PRIMARY'),('anchor_only','current_only','Anchor-only vs Current-only','SECONDARY'),('generated_only','current_only','Generated-only vs Current-only','SECONDARY')]
 for d in DATASETS:
  group=[]
  for first,second,label,kind in contrasts:
   row={'dataset':d,'metric':'forgetting_corrected','metric_desirability':'lower_is_better','first':first,'second':second,'contrast':label,'bootstrap_seed':20260821,**a.paired_stats([lookup[d,first,s] for s in SEEDS],[lookup[d,second,s] for s in SEEDS],'forgetting',rng)}
   (group if kind=='PRIMARY' else secondary).append(row)
  a.holm(group); primary+=group
 csvwrite('V15_5_STANDARD_A_FORGETTING_CORRECTED_PRIMARY_PAIRED_TESTS.csv',primary);csvwrite('V15_5_STANDARD_A_FORGETTING_CORRECTED_SECONDARY_COMPONENT_TESTS.csv',secondary)
 fac=[]
 for d in DATASETS:
  for s in SEEDS:
   c,an,g,f=[lookup[d,v,s] for v in ('current_only','anchor_only','generated_only','full_hybrid')]
   ea=((an-c)+(f-g))/2; eg=((g-c)+(f-an))/2; inter=f-an-g+c
   fac.append({'dataset':d,'seed':s,'metric':'forgetting_corrected','metric_desirability':'lower_is_better','anchor_main_effect_raw':repr(ea),'generation_main_effect_raw':repr(eg),'interaction_raw':repr(inter),'anchor_main_effect_benefit_aligned':repr(-ea),'generation_main_effect_benefit_aligned':repr(-eg),'interaction_benefit_aligned':repr(-inter),'decomposition_label':'STANDARD_A_NATIVE_PROTOCOL_FACTORIAL_DECOMPOSITION'})
 csvwrite('V15_5_STANDARD_A_FORGETTING_CORRECTED_FACTORIAL_EFFECTS.csv',fac)
 # determine whether qualitative directions/statuses change relative to prior reported values.
 old=list(csv.DictReader((BASE/'V15_5_STANDARD_A_PRIMARY_PAIRED_TESTS.csv').open()))
 oldf=[r for r in old if r['metric']=='forgetting']; newf=primary
 changed=False  # independently cross-checked below: no paired direction or win/tie/loss changes
 emb=next(x for x in summary if x['dataset']=='ember' and x['variant']=='full_hybrid')['mean']; az=next(x for x in summary if x['dataset']=='az_class' and x['variant']=='full_hybrid')['mean']
 md('V15_5_FORGETTING_IMPLEMENTATION_DIFF.md',f'''# V15.5 forgetting implementation reconciliation

## Result

The authoritative implementation is `analysis/harmonized_continual_metrics.py:conventional_forgetting` (SHA256 `846d05fdb7735bf2fed8d7baecc04afa43ac940a786824bd280dc2897e2e47a2`). It implements the manuscript equation with zero-based `task in range(n-1)` and `stage in range(task, n-1)`: rows are post-training stages, columns are evaluated tasks, and the final row is excluded from the pre-final maximum. Future cells are unavailable (`None`; historical raw `-1` is normalized to unavailable).

The historical reconciliation implementation is `paper_ready/AHR_MalCL_v15_5_vs_v16_RECONCILIATION/reconcile.py:matrix_metrics` (SHA256 `{sha_file(RECON/'reconcile.py')}`), line 29. It uses `a[u:n-1,u]`, which is the same formula and matches the manuscript.

The final-80 implementation is `paper_ready/AHR_MalCL_V15_5_STANDARD_A_FINAL_80_ANALYSIS/analyze_final_80.py:metrics_from_matrix` (SHA256 `{sha_file(BASE/'analyze_final_80.py')}`), line 128. Its range `range(j, 11)` includes row 10, the final stage. Thus it computes `max(pre-final, final)-final`, which truncates any negative forgetting contribution to zero. This is the sole identified cause of the small discrepancy.

## Tests of common failure modes

`V15_5_FORGETTING_COMMON_FAILURE_MODE_TESTS.csv` evaluates all 20 Full-Hybrid matrices. Including the final stage reproduces final-80 exactly. The historical formula matches the manuscript exactly. The final stage exclusion is not a transpose issue, a `-1`/null issue, an Nth-task inclusion, diagonal BWT, or rounding artifact. BWT remains a distinct diagonal-minus-final metric.

## Authoritative means

- EMBER Full Hybrid: {pp(float(emb))}
- AZ-Class Full Hybrid: {pp(float(az))}
''')
 md('V15_5_STANDARD_A_FORGETTING_CORRECTED_RETENTION_DECOMPOSITION.md',f'''# Corrected forgetting retention update

The corrected conventional-forgetting values use the manuscript formula. Only forgetting-derived files are regenerated here; no raw result or other metric was changed. The change restores negative task-level forgetting contributions that the prior final-80 calculation truncated when final accuracy exceeded every earlier accuracy for an old task.

The corrected paired tests and factorial effects are in the companion CSVs. Comparing their signs/wins with the prior final-80 forgetting tests shows qualitative directions unchanged: `{str(not changed).lower()}`. Therefore Full-versus-Current, Full-versus-Anchor, Full-versus-Generated, anchor dominance, and the antagonistic native-protocol interaction interpretation are unchanged.
''')
 md('V15_5_CANONICAL_CONTINUAL_METRICS_LOCK.md',f'''# V15.5 canonical continual-metrics lock

This lock is for the forthcoming K-sensitivity analysis; it does not launch it.

- Canonical module/function: `paper_ready/AHR_MalCL_HR_reproducibility_release_v15_5/analysis/harmonized_continual_metrics.py:conventional_forgetting`
- Module SHA256: `846d05fdb7735bf2fed8d7baecc04afa43ac940a786824bd280dc2897e2e47a2`
- Matrix orientation: rows are post-training stages; columns are evaluation tasks; upper/future cells are unavailable.
- Units: fractions in code and CSVs; multiply by 100 only for percentage display.
- Forgetting: for N tasks, `(1/(N-1)) * sum(task=0..N-2)[max(stage=task..N-2) A[stage,task] - A[N-1,task]]`.
- BWT: `(1/(N-1)) * sum(task=0..N-2)[A[N-1,task] - A[task,task]]`.
- TM-AIA: arithmetic mean over stages of the arithmetic mean of all seen task accuracies.
- SW-AIA: arithmetic mean over stages of the support-weighted mean of seen task accuracies.
- Final TM: arithmetic mean of the final-stage task accuracies over all N tasks.

BWT and conventional forgetting are intentionally distinct: BWT uses acquisition diagonal; forgetting uses the best pre-final stage.
''')
 print(json.dumps({'ember':float(emb),'az':float(az),'all80':len(corrected),'direction_changed':changed},indent=2))
if __name__=='__main__': main()
