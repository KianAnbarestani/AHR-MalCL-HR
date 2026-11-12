from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import taskwise_matrix as matrix
from test_taskwise_matrix import _MemoryStorage, _plan, _state, _tiny_bundle


def _progress(plan):
    return matrix.create_matrix_progress(
        plan, run1_checkpoint_paths=[f'/drive/run1/task-{i:02d}.pt' for i in range(1, 12)],
        run1_registry_evidence={'run_id': 'verified-run1', 'artifact_manifest_sha256': 'e' * 64},
    )


def _complete(row):
    row['run_complete'] = True
    for task in row['tasks']:
        task.update(done=True, checkpoint_path=f"/existing/{row['run_index']}/task-{task['task_id']:02d}.pt")


def test_assignments_are_inclusive_disjoint_and_preserve_original_paths():
    assignments = matrix.build_worker_assignments('/drive/production/results', '/drive/production/checkpoints')
    matrix.validate_worker_assignments(assignments)
    assert [(a['start_run'], a['end_run']) for a in assignments] == [(1, 60), (61, 80), (81, 100)]
    assert assignments[0]['progress_path'] == '/drive/production/results/matrix_progress.json'
    assert assignments[0]['checkpoint_root'] == '/drive/production/checkpoints'
    assert assignments[1]['output_root'] == '/drive/production/workers/runs_061_080/results'
    assert assignments[2]['checkpoint_root'] == '/drive/production/workers/runs_081_100/checkpoints'
    for bad_range in [(0, 60), (61, 101), (80, 61), (True, 60), (1.0, 60)]:
        bad = copy.deepcopy(assignments)
        bad[0]['start_run'], bad[0]['end_run'] = bad_range
        with pytest.raises(ValueError):
            matrix.validate_worker_assignments(bad)
    bad = copy.deepcopy(assignments)
    bad[1]['start_run'] = 60
    with pytest.raises(ValueError, match='overlap'):
        matrix.validate_worker_assignments(bad)
    for field in ('progress_path', 'output_root', 'checkpoint_root'):
        bad = copy.deepcopy(assignments)
        bad[1][field] = bad[0][field]
        with pytest.raises(ValueError):
            matrix.validate_worker_assignments(bad)
    bad = copy.deepcopy(assignments)
    bad[1]['checkpoint_root'] = bad[0]['output_root'] + '/nested'
    with pytest.raises(ValueError, match='overlap'):
        matrix.validate_worker_assignments(bad)


def test_original_run44_resumes_existing_state_stops_at60_and_safe_rerun(tmp_path, monkeypatch):
    from scripts import production_workflow

    plan = _plan()
    progress = _progress(plan)
    for row in progress['runs'][1:43]:
        _complete(row)
    # Other already-complete rows are skipped, but no task outside 1..60 is touched.
    for row in progress['runs'][44:59]:
        _complete(row)
    storage = _MemoryStorage()
    path = tmp_path / 'matrix_progress.json'
    logical = '/drive/production/results/matrix_progress.json'
    matrix.atomic_write_json(path, progress)
    storage.files[logical] = path.read_bytes()
    record44 = plan['runs'][43]
    cp = '/drive/production/checkpoints/taskwise_v1/' + plan['plan_sha256'] + '/' + matrix._record_directory(record44) + '/task-01.pt'
    state = _state(1)
    state.identity = {'config_sha256': record44['config_sha256'], 'schedule_sha256': plan['schedule_sha256'], 'code_sha256': plan['training_source_sha256']}
    state.classifier_optimizer_state = {'step': torch.tensor(17), 'momentum': torch.tensor([2.5])}
    matrix.persist_task_completion(
        progress, run_index=44, task_id=1, state=state, checkpoint_path=cp,
        local_checkpoint_path=tmp_path/'existing-task-01.pt', progress_path=path,
        logical_progress_path=logical, storage_adapter=storage,
    )
    monkeypatch.setattr(production_workflow, 'validate_runtime_bound_plan', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    calls = []
    fail = {'once': True}

    def runner(_config, _bundle, output, **kwargs):
        index = int(output.name[:3])
        restored = kwargs['taskwise_resume_state']
        calls.append((index, None if restored is None else restored.cursor.task_index))
        start = 1 if restored is None else restored.cursor.task_index + 1
        if index == 44 and start == 2:
            assert torch.equal(restored.classifier_optimizer_state['momentum'], torch.tensor([2.5]))
            assert torch.equal(restored.rng_state['torch_cpu'], state.rng_state['torch_cpu'])
        for task_id in range(start, 12):
            if index == 44 and task_id == 3 and fail['once']:
                fail['once'] = False
                return {'status': 'INTERRUPTED_TASK', 'cursor': {'task_index': 2}}
            saved = _state(task_id)
            saved.identity = {'config_sha256': plan['runs'][index-1]['config_sha256'], 'schedule_sha256': plan['schedule_sha256'], 'code_sha256': plan['training_source_sha256']}
            kwargs['taskwise_checkpoint_callback'](saved, task_id)
        output.mkdir(parents=True, exist_ok=True)
        (output/'full-result.json').write_text('{}')
        (output/'artifact_manifest.json').write_text('{}')
        return {'status': 'COMPLETED_VALIDATED', 'validated': True}

    common = dict(
        plan=plan, progress_path=path, logical_progress_path=logical,
        data_roots={}, cache_root=tmp_path/'cache', preflight={}, schedule_path=path,
        schedule_manifest_path=path, output_root=tmp_path/'results', logical_output_root='/drive/production/results',
        checkpoint_root=tmp_path/'checkpoints', logical_checkpoint_root='/drive/production/checkpoints',
        storage_adapter=storage, launch_enabled=True, acknowledge_single_worker=True,
        bundle_loader=lambda *args, **kwargs: _tiny_bundle(), runner_fn=runner,
        start_run=1, end_run=60,
        worker_assignments=matrix.build_worker_assignments('/drive/production/results', '/drive/production/checkpoints'),
    )
    assert matrix.run_taskwise_matrix(**common)['status'] == 'INTERRUPTED_TASK'
    assert matrix.run_taskwise_matrix(**common)['status'] == 'RANGE_COMPLETE'
    assert calls == [(44, 1), (44, 2), (60, None)]
    assert matrix.run_taskwise_matrix(**common)['status'] == 'RANGE_COMPLETE'
    assert calls == [(44, 1), (44, 2), (60, None)]
    persisted = json.loads(storage.files[logical])
    assert all(r['run_complete'] for r in persisted['runs'][:60])
    assert persisted['runs'][60:] == progress['runs'][60:]
    assert not any('/061_' in p for p in storage.files)


def test_selection_respects_range_even_when_earlier_run_is_pending():
    progress = _progress(_plan())
    assert matrix.first_unfinished_run(progress, start_run=61, end_run=80)['run_index'] == 61
    for row in progress['runs'][60:80]:
        _complete(row)
    assert matrix.first_unfinished_run(progress, start_run=61, end_run=80) is None


def test_worker_progress_initialization_never_changes_original(tmp_path):
    plan = _plan()
    original = _progress(plan)
    for row in original['runs'][1:43]:
        _complete(row)
    storage = _MemoryStorage()
    storage.files['/drive/production/results/matrix_progress.json'] = json.dumps(original).encode()
    before = dict(storage.files)
    secondary = matrix.prepare_matrix_progress(
        plan, local_progress_path=tmp_path/'secondary.json',
        logical_progress_path='/drive/production/workers/runs_061_080/results/matrix_progress.json',
        storage_adapter=storage,
        run1_checkpoint_paths=[t['checkpoint_path'] for t in original['runs'][0]['tasks']],
        run1_registry_evidence=original['runs'][0]['completion_provenance'],
    )
    assert secondary['runs'][60]['tasks'][0]['done'] is False
    assert storage.files['/drive/production/results/matrix_progress.json'] == before['/drive/production/results/matrix_progress.json']


@pytest.mark.parametrize('start,end', [(61, 80), (81, 100)])
def test_secondary_workers_start_at_assigned_first_run_skip_done_and_stop_at_end(tmp_path, monkeypatch, start, end):
    from scripts import production_workflow
    plan = _plan()
    progress = _progress(plan)
    for row in progress['runs'][start:end-1]:
        _complete(row)
    assignments = matrix.build_worker_assignments('/drive/production/results', '/drive/production/checkpoints')
    assignment = matrix.select_worker_assignment(assignments, start, end)
    storage = _MemoryStorage()
    path = tmp_path/'matrix_progress.json'
    matrix.atomic_write_json(path, progress)
    storage.files[assignment['progress_path']] = path.read_bytes()
    monkeypatch.setattr(production_workflow, 'validate_runtime_bound_plan', lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    calls = []
    def runner(_config, _bundle, output, **kwargs):
        index = int(output.name[:3])
        calls.append(index)
        assert kwargs['taskwise_resume_state'] is None
        for task_id in range(1, 12):
            saved = _state(task_id)
            saved.identity = {'config_sha256': plan['runs'][index-1]['config_sha256'], 'schedule_sha256': plan['schedule_sha256'], 'code_sha256': plan['training_source_sha256']}
            kwargs['taskwise_checkpoint_callback'](saved, task_id)
        output.mkdir(parents=True)
        (output/'full-result.json').write_text('{}')
        (output/'artifact_manifest.json').write_text('{}')
        return {'status': 'COMPLETED_VALIDATED', 'validated': True}
    common = dict(plan=plan, progress_path=path, logical_progress_path=assignment['progress_path'],
        output_root=tmp_path/'results', logical_output_root=assignment['output_root'],
        checkpoint_root=tmp_path/'checkpoints', logical_checkpoint_root=assignment['checkpoint_root'],
        data_roots={}, cache_root=tmp_path/'cache', preflight={}, schedule_path=path, schedule_manifest_path=path,
        storage_adapter=storage, launch_enabled=True, acknowledge_single_worker=True,
        start_run=start, end_run=end, worker_assignments=assignments,
        bundle_loader=lambda *args, **kwargs: _tiny_bundle(), runner_fn=runner)
    result = matrix.run_taskwise_matrix(**common)
    assert result['status'] == 'RANGE_COMPLETE'
    assert result['completed_runs'] == end-start+1
    assert calls == [start, end]
    assert matrix.run_taskwise_matrix(**common)['status'] == 'RANGE_COMPLETE'
    assert calls == [start, end]
    original = dict(storage.files)
    common['logical_progress_path'] = '/drive/production/results/matrix_progress.json'
    with pytest.raises(ValueError, match='differs from the declared assignment'):
        matrix.run_taskwise_matrix(**common)
    assert storage.files == original
