from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "revision_r1"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _call_line(function: ast.FunctionDef, method: str) -> int:
    return min(
        node.lineno for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == method)
            or (isinstance(node.func, ast.Name) and node.func.id == method)
        )
    )


def test_classifier_phase_explicitly_enters_train_mode_before_forward():
    tree = ast.parse((PACKAGE / "runner.py").read_text(encoding="utf-8"))
    run_one = _function(tree, "run_one")
    classifier_phase = next(
        node for node in ast.walk(run_one)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "phase_before"
        and any(isinstance(value, ast.Constant) and value.value == "classifier" for value in node.test.comparators)
    )
    train_lines = [
        node.lineno for node in ast.walk(classifier_phase)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "classifier"
        and node.func.attr == "train"
    ]
    forward_lines = [
        node.lineno for node in ast.walk(classifier_phase)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "classifier"
    ]
    assert train_lines and forward_lines and min(train_lines) < min(forward_lines)


def test_resume_rng_restore_follows_head_and_model_and_optimizer_state_loading():
    tree = ast.parse((PACKAGE / "runner.py").read_text(encoding="utf-8"))
    run_one = _function(tree, "run_one")
    restore_line = _call_line(run_one, "_restore_rng")
    required_loads = ("expand", "load_state_dict")
    reconstruction_lines = [
        node.lineno for node in ast.walk(run_one)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in required_loads
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {
            "classifier", "generator", "critic", "classifier_optimizer",
            "generator_optimizer", "critic_optimizer",
        }
        and node.lineno < restore_line
    ]
    names_loaded_before_restore = {
        node.func.value.id for node in ast.walk(run_one)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in required_loads
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {
            "classifier", "generator", "critic", "classifier_optimizer",
            "generator_optimizer", "critic_optimizer",
        }
        and node.lineno < restore_line
    }
    assert reconstruction_lines
    assert {"classifier", "generator", "critic"} <= names_loaded_before_restore
    assert {"classifier_optimizer", "generator_optimizer", "critic_optimizer"} <= names_loaded_before_restore


def test_classifier_timing_encloses_forward_loss_backward_and_optimizer_step():
    tree = ast.parse((PACKAGE / "runner.py").read_text(encoding="utf-8"))
    run_one = _function(tree, "run_one")
    classifier_measure = next(
        node for node in ast.walk(run_one)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "measure"
            and len(item.context_expr.args) == 1
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "classifier"
            for item in node.items
        )
    )
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(classifier_measure)
        if isinstance(node, ast.Call)
    }
    assert "classifier" in calls
    assert "derpp_loss" in calls
    assert "step" in calls


def test_resource_measurement_uses_synchronized_boundaries_and_checkpoint_component():
    tree = ast.parse((PACKAGE / "resources.py").read_text(encoding="utf-8"))
    tracker = _function(tree, "measure")
    start_timer = _function(tree, "start_timer")
    finish_timer = _function(tree, "finish_timer")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "start_timer" for node in ast.walk(tracker))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "finish_timer" for node in ast.walk(tracker))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_synchronize_cuda" for node in ast.walk(start_timer))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_synchronize_cuda" for node in ast.walk(finish_timer))
    source = (PACKAGE / "resources.py").read_text(encoding="utf-8")
    assert '"checkpoint"' in source


def test_corrected_validation_build_id_is_domain_separated():
    source = (PACKAGE / "validation_workflow.py").read_text(encoding="utf-8")
    assert 'VALIDATION_BUILD_ID = "revision_R1_colab_validation_v6_2_final_gan_recovery_qa"' in source
    assert "VALIDATION_BUILD_ID" in source[source.index("def _workflow_id"):source.index("def _validation_root")]
