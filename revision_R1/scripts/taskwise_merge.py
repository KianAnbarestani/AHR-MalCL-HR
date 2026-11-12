"""Offline verification/index of worker snapshots; never alters source artifacts."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from revision_r1.compatibility import CHECKPOINT_IDENTITY_SHA256
from revision_r1.validation import REQUIRED_ARTIFACTS, file_sha256, validate_run_artifacts
from scripts.production_workflow import _run_id_for, run_output_root, validate_frozen_plan
from scripts.taskwise_matrix import TASKWISE_NAMESPACE, _record_directory, validate_matrix_progress, validate_worker_assignments


def verify_result(directory: Path, plan: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Use the production artifact validator plus exact frozen scientific identities."""
    if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob('*')):
        raise ValueError('symlinked result artifacts are not accepted')
    manifest = json.loads((directory / 'artifact_manifest.json').read_text(encoding='utf-8'))
    files = manifest.get('files', {})
    if not set(REQUIRED_ARTIFACTS).difference({'artifact_manifest.json'}).issubset(files):
        raise ValueError('manifest does not bind every required result artifact')
    if any(Path(name).name != name or name in {'.', '..'} for name in files):
        raise ValueError('artifact manifest contains an unsafe relative path')
    validate_run_artifacts(directory, require_production=True, verify_hashes=True)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'full-result.json').read_text(encoding='utf-8'))
    order = json.loads((directory / 'class_order.json').read_text(encoding='utf-8'))
    checks = {
        'config_sha256': config.get('config_sha256') == record['config_sha256'],
        'training_source': config.get('code_sha256') == plan['training_source_sha256'],
        'config_payload': all(config.get(key) == value for key, value in record['config'].items()),
        'namespace': config.get('namespace') == 'PRODUCTION' and summary.get('namespace') == 'PRODUCTION',
        'dataset': summary.get('dataset') == record['dataset'],
        'seed': summary.get('seed') == record['seed'],
        'arm': summary.get('arm') == record['arm'],
        'schedule': summary.get('schedule_sha256') == plan['schedule_sha256'],
        'class_order': order.get('class_order') == record['class_order']
            and order.get('sha256') == record['class_order_sha256']
            and summary.get('class_order_sha256') == record['class_order_sha256'],
        'status': json.loads((directory/'status.json').read_text()).get('status') == 'COMPLETED_VALIDATED',
    }
    budgets = record.get('update_budgets_by_task')
    if budgets is not None:
        if not isinstance(budgets, list) or len(budgets) != 11 or any(type(n) is not int or n < 1 for n in budgets):
            raise ValueError('frozen plan update budgets are malformed')
        checks['classifier_updates'] = summary.get('classifier_updates') == sum(budgets)
        checks['sampled_examples'] = summary.get('sampled_examples') == 256 * sum(budgets)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError('result differs from frozen plan: ' + ', '.join(failed))
    return {'run_index': record['run_index'], 'run_key': record['run_key'],
            'run_id': _run_id_for(record, synthetic=False),
            'config_sha256': record['config_sha256'], 'output_path': str(directory),
            'classifier_update_budget_verified': budgets is not None,
            'artifact_manifest_sha256': file_sha256(directory/'artifact_manifest.json'),
            'summary': summary}


def merge_worker_results(plan: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    """Workers specify logical assignments and read-only local snapshot locations.

    Each entry contains assignment (the same dictionary validated by Cell F),
    local_progress_path, and local_output_root. Only assigned rows contribute.
    The report indexes existing results; it neither copies nor rewrites them.
    """
    validate_frozen_plan(plan, expected_source_sha256=CHECKPOINT_IDENTITY_SHA256)
    validate_worker_assignments([worker['assignment'] for worker in workers])
    records = {row['run_key']: row for row in plan['runs']}
    identities = {(row['dataset'], row['seed'], row['arm']): row['run_key'] for row in plan['runs']}
    expected_owner = {}
    for worker in workers:
        a = worker['assignment']
        for index in range(a['start_run'], a['end_run'] + 1):
            expected_owner[plan['runs'][index-1]['run_key']] = a['worker_id']
    candidates = {key: [] for key in records}
    mismatched = []
    source_paths = []
    for worker in workers:
        a = worker['assignment']
        worker_id = a['worker_id']
        root = Path(worker['local_output_root']).absolute()
        local_progress = Path(worker['local_progress_path']).absolute()
        source_paths.extend([root, local_progress])
        try:
            if root.is_symlink() or local_progress.is_symlink():
                raise ValueError('worker snapshot must not be a symlink')
            progress = json.loads(local_progress.read_text(encoding='utf-8'))
            validate_matrix_progress(progress, plan)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            mismatched.append({'worker_id': worker_id, 'path': str(local_progress), 'error': str(exc)})
            continue
        runs_root = root / TASKWISE_NAMESPACE / plan['plan_sha256'] / 'runs'
        directories = sorted({path.parent for path in runs_root.rglob('full-result.json')}) if runs_root.exists() else []
        if a['start_run'] == 1:
            legacy = Path(worker.get('local_run1_output') or run_output_root(root, plan['runs'][0]))
            source_paths.append(legacy.absolute())
            if legacy.is_dir():
                directories.append(legacy)
        for directory in directories:
            key = None
            try:
                summary = json.loads((directory/'full-result.json').read_text(encoding='utf-8'))
                key = identities.get((summary.get('dataset'), summary.get('seed'), summary.get('arm')))
                if key is None:
                    raise ValueError('unrecognized result identity')
                record = records[key]
                # Register every claim before validation so two corrupt/valid copies
                # can never be silently reduced to a single apparently valid run.
                claim = {'worker_id': worker_id, 'path': str(directory), 'valid': False}
                candidates[key].append(claim)
                if expected_owner.get(key) != worker_id:
                    raise ValueError('result belongs to another worker assignment')
                row = progress['runs'][record['run_index']-1]
                if not row['run_complete']:
                    raise ValueError('artifacts exist but progress does not record completed publication')
                imported = record['run_index'] == 1 and row.get('completion_provenance', {}).get('kind') == 'existing_registry_entry'
                expected_directory = root / TASKWISE_NAMESPACE / plan['plan_sha256'] / 'runs' / _record_directory(record)
                if not imported and directory != expected_directory:
                    raise ValueError('result directory differs from the deterministic frozen-plan route')
                if not imported:
                    checkpoint_dir = Path(a['checkpoint_root']) / TASKWISE_NAMESPACE / plan['plan_sha256'] / _record_directory(record)
                    for task in row['tasks']:
                        if task['checkpoint_path'] != str(checkpoint_dir/f"task-{task['task_id']:02d}.pt"):
                            raise ValueError('recorded task checkpoint route differs from this worker assignment')
                result = verify_result(directory, plan, record)
                if imported and (
                        row['completion_provenance']['artifact_manifest_sha256'] != result['artifact_manifest_sha256']
                        or row['completion_provenance'].get('run_id') != result['run_id']):
                    raise ValueError('imported Run 1 result differs from its existing registry evidence')
                claim.update(valid=True, result=result)
            except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                mismatched.append({'worker_id': worker_id, 'run_key': key, 'path': str(directory), 'error': str(exc)})
    duplicate = [{'run_key': key, 'claims': [{'worker_id': c['worker_id'], 'path': c['path']} for c in rows]}
                 for key, rows in candidates.items() if len(rows) > 1]
    verified = [rows[0]['result'] for rows in candidates.values() if len(rows) == 1 and rows[0]['valid']]
    verified.sort(key=lambda row: row['run_index'])
    missing = [{'run_index': row['run_index'], 'run_key': row['run_key']}
               for row in plan['runs'] if not any(c['valid'] for c in candidates[row['run_key']])]
    return {'schema_version': 1, 'status': 'PASS' if not (missing or duplicate or mismatched) else 'INCOMPLETE_OR_INVALID',
            'plan_sha256': plan['plan_sha256'], 'schedule_sha256': plan['schedule_sha256'],
            'training_source_sha256': plan['training_source_sha256'], 'verified_runs': verified,
            'missing_results': missing, 'duplicate_results': duplicate, 'mismatched_results': mismatched,
            'checkpoint_payloads_loaded': False, 'source_files_modified': False,
            'source_paths': sorted(set(str(p) for p in source_paths))}


def write_merge_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path).absolute()
    resolved = target.resolve()
    for source in report['source_paths']:
        source_path = Path(source).resolve()
        if resolved == source_path or source_path in resolved.parents:
            raise ValueError('merge report must be outside every source snapshot')
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'
    # Exclusive creation also protects an existing report. No source writes.
    with target.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='exact runtime frozen plan JSON from your existing run')
    parser.add_argument('--workers', type=Path, required=True, help='list of assignments and local snapshot roots')
    parser.add_argument('--report', type=Path, required=True, help='new report file outside all source snapshots')
    args = parser.parse_args()
    report = merge_worker_results(json.loads(args.plan.read_text()), json.loads(args.workers.read_text()))
    write_merge_report(args.report, report)
    print(json.dumps({key: (len(report[key]) if isinstance(report[key], list) else report[key])
                      for key in ('status', 'verified_runs', 'missing_results', 'duplicate_results', 'mismatched_results')}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
