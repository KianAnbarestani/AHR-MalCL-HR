"""Small schema fixtures exercise the real offline production validators."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from revision_r1.validation import REQUIRED_ARTIFACTS, build_artifact_manifest, file_sha256
from scripts import taskwise_matrix as matrix
from scripts.taskwise_merge import merge_worker_results, write_merge_report


ROOT = Path(__file__).resolve().parents[1]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _plan():
    return json.loads((ROOT/'configs/production_run_plan_v1.json').read_text())


def _artifacts(directory, plan, record):
    directory.mkdir(parents=True)
    for name in REQUIRED_ARTIFACTS:
        (directory/name).write_text('{}')
    _json(directory/'config.json', {**record['config'], 'config_sha256': record['config_sha256'],
                                   'code_sha256': plan['training_source_sha256'], 'namespace': 'PRODUCTION'})
    _json(directory/'full-result.json', {
        'dataset': record['dataset'], 'seed': record['seed'], 'arm': record['arm'],
        'schedule_sha256': plan['schedule_sha256'], 'class_order_sha256': record['class_order_sha256'],
        'namespace': 'PRODUCTION', 'training_reproduction_claimed': False,
        'final_metrics': {'accuracy': 0.5},
    })
    _json(directory/'status.json', {'status': 'COMPLETED_VALIDATED'})
    _json(directory/'class_order.json', {'class_order': record['class_order'], 'sha256': record['class_order_sha256']})
    _json(directory/'task_matrix.json', {'task_matrix': [[0.5 if col <= row else None for col in range(11)] for row in range(11)]})
    _json(directory/'source_proportions.json', [{'sampled_examples': 0, 'realized_draws': {}} for _ in range(11)])
    _json(directory/'classifier_mode_trace.json', [
        {'phase': phase, 'task_id': i, 'classifier_training': active, 'dropout_training': [active]}
        for i in range(1, 12) for phase, active in [('classifier_training', True), ('evaluation', False)]
    ])
    _json(directory/'task_boundary_prediction_manifest.json', {'stages': [{'stage': i, 'test_example_count': 1} for i in range(1, 12)]})
    np.savez(directory/'predictions.npz', y_true_global=np.array([0]), y_pred_global=np.array([0]))
    np.savez(directory/'task_boundary_predictions.npz', **{
        f'stage_{i:02d}_{field}': np.array([0]) for i in range(1, 12)
        for field in ['test_indices', 'y_true_global', 'y_pred_global', 'head_class_ids']
    })
    (directory/'per_task_metrics.csv').write_text('task_id\n' + '\n'.join(str(i) for i in range(1, 12)))
    (directory/'selection_records.jsonl').write_bytes(b'')
    _json(directory/'artifact_manifest.json', build_artifact_manifest(directory))


def _worker_fixture(tmp_path, *, number=61):
    plan = _plan()
    assignment = matrix.build_worker_assignments('/drive/production/results', '/drive/production/checkpoints')[1]
    root = tmp_path/'source-results'
    progress = matrix.create_matrix_progress(plan, run1_checkpoint_paths=[f'/old/{i}.pt' for i in range(11)],
        run1_registry_evidence={'run_id': 'old-run1', 'artifact_manifest_sha256': 'e'*64})
    record = plan['runs'][number-1]
    row = progress['runs'][number-1]
    row['run_complete'] = True
    for task in row['tasks']:
        task.update(done=True, checkpoint_path=str(Path(assignment['checkpoint_root'])/matrix.TASKWISE_NAMESPACE/plan['plan_sha256']/matrix._record_directory(record)/f"task-{task['task_id']:02d}.pt"))
    directory = root/matrix.TASKWISE_NAMESPACE/plan['plan_sha256']/'runs'/matrix._record_directory(record)
    _artifacts(directory, plan, record)
    path = root/'matrix_progress.json'
    _json(path, progress)
    return plan, [{'assignment': assignment, 'local_progress_path': str(path), 'local_output_root': str(root)}], directory


def test_merge_verifies_real_artifacts_missing_results_and_preserves_all_sources(tmp_path):
    plan, workers, directory = _worker_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report = merge_worker_results(plan, workers)
    assert [r['run_index'] for r in report['verified_runs']] == [61]
    assert len(report['missing_results']) == 99
    assert not report['mismatched_results'] and not report['duplicate_results']
    assert report['verified_runs'][0]['run_id'].startswith(plan['runs'][60]['run_key'])
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report_path = tmp_path/'merge-report.json'
    write_merge_report(report_path, report)
    with pytest.raises(FileExistsError):
        write_merge_report(report_path, report)
    with pytest.raises(ValueError, match='outside'):
        write_merge_report(directory/'do-not-create.json', report)
    assert not (directory/'do-not-create.json').exists()


def test_merge_reports_duplicate_and_rejects_changed_seed_and_corrupt_file(tmp_path):
    plan, workers, directory = _worker_fixture(tmp_path)
    shutil.copytree(directory, directory.parent/'duplicate-copy')
    report = merge_worker_results(plan, workers)
    assert report['duplicate_results'][0]['run_key'] == plan['runs'][60]['run_key']
    assert not report['verified_runs']
    config = json.loads((directory/'config.json').read_text())
    config['seed'] = 999
    _json(directory/'config.json', config)
    _json(directory/'artifact_manifest.json', build_artifact_manifest(directory))
    report = merge_worker_results(plan, workers)
    assert any('config_payload' in item['error'] for item in report['mismatched_results'])
    (directory/'trace.jsonl').write_text('corrupt')
    report = merge_worker_results(plan, workers)
    assert any('hash/size mismatch' in item['error'] for item in report['mismatched_results'])


def test_merge_rejects_wrong_plan_progress_and_overlap(tmp_path):
    plan, workers, _ = _worker_fixture(tmp_path)
    with pytest.raises(ValueError, match='overlap|unique'):
        merge_worker_results(plan, workers + copy.deepcopy(workers))
    path = Path(workers[0]['local_progress_path'])
    progress = json.loads(path.read_text())
    progress['schedule_sha256'] = '0'*64
    _json(path, progress)
    report = merge_worker_results(plan, workers)
    assert not report['verified_runs']
    assert 'frozen identities' in report['mismatched_results'][0]['error']


def test_cell_f_default_is_offline_and_selects_each_worker_without_rebinding_base(tmp_path):
    from scripts.build_github_handoff import notebook_document
    cell = ''.join(notebook_document('1'*40)['cells'][6]['source'])
    for start, end in [(1, 60), (61, 80), (81, 100)]:
        namespace = {'LOGICAL_OUTPUT_ROOT': Path('/drive/production/results'),
                     'LOGICAL_CHECKPOINT_ROOT': Path('/drive/production/checkpoints'),
                     'OUTPUT_ROOT': tmp_path/'results', 'CHECKPOINT_ROOT': tmp_path/'checkpoints'}
        modified = cell.replace('START_RUN = 1 ', f'START_RUN = {start} ').replace('END_RUN = 60 ', f'END_RUN = {end} ')
        exec(compile(modified, 'Cell-F', 'exec'), namespace)
        assert namespace['_worker']['start_run'] == start
        assert namespace['_worker']['end_run'] == end
        assert namespace['OUTPUT_ROOT'] == tmp_path/'results'
        assert namespace['LAUNCH_MATRIX'] is False
        assert not tmp_path.joinpath('results').exists()


def test_complete_three_worker_merge_is_ordered_and_imports_run1_from_existing_evidence(tmp_path):
    from scripts.production_workflow import _run_id_for
    plan = _plan()
    assignments = matrix.build_worker_assignments('/drive/production/results', '/drive/production/checkpoints')
    workers = []
    for assignment in assignments:
        root = tmp_path/assignment['worker_id']/'results'
        progress = matrix.create_matrix_progress(plan, run1_checkpoint_paths=[f'/old/{i}.pt' for i in range(11)],
            run1_registry_evidence={'run_id': 'old-run1', 'artifact_manifest_sha256': 'e'*64})
        worker = {'assignment': assignment, 'local_progress_path': str(root/'matrix_progress.json'), 'local_output_root': str(root)}
        for index in range(assignment['start_run'], assignment['end_run']+1):
            record = plan['runs'][index-1]
            directory = root/matrix.TASKWISE_NAMESPACE/plan['plan_sha256']/'runs'/matrix._record_directory(record)
            if index == 1:
                directory = root/'existing-run1'
                worker['local_run1_output'] = str(directory)
            _artifacts(directory, plan, record)
            row = progress['runs'][index-1]
            row['run_complete'] = True
            if index == 1:
                row['completion_provenance']['artifact_manifest_sha256'] = file_sha256(directory/'artifact_manifest.json')
                row['completion_provenance']['run_id'] = _run_id_for(record, synthetic=False)
            else:
                for task in row['tasks']:
                    task.update(done=True, checkpoint_path=str(Path(assignment['checkpoint_root'])/matrix.TASKWISE_NAMESPACE/plan['plan_sha256']/matrix._record_directory(record)/f"task-{task['task_id']:02d}.pt"))
        _json(root/'matrix_progress.json', progress)
        workers.append(worker)
    report = merge_worker_results(plan, workers)
    assert report['status'] == 'PASS', report['mismatched_results']
    assert [row['run_index'] for row in report['verified_runs']] == list(range(1, 101))
    assert not report['missing_results'] and not report['duplicate_results']


def test_bound_plan_update_counts_and_final_predictions_corruption_are_rejected(tmp_path):
    from scripts.taskwise_merge import verify_result
    plan, workers, directory = _worker_fixture(tmp_path)
    record = plan['runs'][60]
    record['update_budgets_by_task'] = [1]*11
    summary = json.loads((directory/'full-result.json').read_text())
    summary.update(classifier_updates=11, sampled_examples=11*256)
    _json(directory/'full-result.json', summary)
    _json(directory/'artifact_manifest.json', build_artifact_manifest(directory))
    assert verify_result(directory, plan, record)['classifier_update_budget_verified'] is True
    summary['classifier_updates'] = 12
    _json(directory/'full-result.json', summary)
    _json(directory/'artifact_manifest.json', build_artifact_manifest(directory))
    with pytest.raises(ValueError, match='classifier_updates'):
        verify_result(directory, plan, record)
    plan2, workers2, directory2 = _worker_fixture(tmp_path/'corrupt-archive')
    (directory2/'predictions.npz').write_bytes(b'PK\x03\x04truncated')
    _json(directory2/'artifact_manifest.json', build_artifact_manifest(directory2))
    report = merge_worker_results(plan2, workers2)
    assert report['status'] == 'INCOMPLETE_OR_INVALID'
    assert len(report['mismatched_results']) == 1
    assert not report['verified_runs']


@pytest.mark.parametrize('start,end', [(1, 60), (61, 80), (81, 100)])
def test_cell_f_enabled_setup_uses_assigned_storage_and_safe_rerun_without_a_mount(tmp_path, monkeypatch, start, end):
    from scripts import production_workflow
    from scripts.build_github_handoff import notebook_document
    from test_taskwise_matrix import _MemoryStorage
    plan = _plan()
    progress = matrix.create_matrix_progress(plan, run1_checkpoint_paths=[f'/old/{i}.pt' for i in range(11)],
        run1_registry_evidence={'run_id': 'old-run1', 'artifact_manifest_sha256': 'e'*64})
    storage = _MemoryStorage()
    logical = '/drive/production/results/matrix_progress.json'
    original_bytes = json.dumps(progress).encode()
    storage.files[logical] = original_bytes
    original_runner = matrix.run_taskwise_matrix
    called = []
    def check_setup(plan, **kwargs):
        called.append(kwargs.copy())
        # Execute the actual range/path/progress checks; never enter training.
        kwargs['launch_enabled'] = False
        return original_runner(plan, **kwargs)
    monkeypatch.setattr(matrix, 'run_taskwise_matrix', check_setup)
    monkeypatch.setattr(production_workflow, 'validate_runtime_bound_plan', lambda *_: None)
    cell = ''.join(notebook_document('1'*40)['cells'][6]['source'])
    cell = cell.replace('START_RUN = 1 ', f'START_RUN = {start} ').replace('END_RUN = 60 ', f'END_RUN = {end} ')
    cell = cell.replace('LAUNCH_MATRIX = False', 'LAUNCH_MATRIX = True').replace('ACKNOWLEDGE_SINGLE_WORKER = False', 'ACKNOWLEDGE_SINGLE_WORKER = True')
    namespace = dict(RUNTIME_PLAN=plan, STORAGE_ADAPTER=storage,
        LOGICAL_OUTPUT_ROOT=Path('/drive/production/results'), LOGICAL_CHECKPOINT_ROOT=Path('/drive/production/checkpoints'),
        OUTPUT_ROOT=tmp_path/'results', CHECKPOINT_ROOT=tmp_path/'checkpoints',
        MATRIX_PROGRESS_PATH=tmp_path/'results/matrix_progress.json', LOGICAL_MATRIX_PROGRESS_PATH=Path(logical),
        DATA_ROOTS={}, CACHE_ROOT=tmp_path/'cache', PREFLIGHT={}, EFFECTIVE_SCHEDULE_PATH=tmp_path/'schedule',
        EFFECTIVE_SCHEDULE_MANIFEST_PATH=tmp_path/'schedule.json')
    for _ in range(2):
        exec(compile(cell, 'Cell-F-enabled-with-disabled-runner', 'exec'), namespace)
    assert len(called) == 2
    assert called[0]['start_run'] == start and called[0]['end_run'] == end
    assert called[0]['output_root'] == called[1]['output_root']
    assert called[0]['logical_progress_path'] == called[1]['logical_progress_path']
    assert storage.files[logical] == original_bytes
    assert namespace['OUTPUT_ROOT'] == tmp_path/'results'
