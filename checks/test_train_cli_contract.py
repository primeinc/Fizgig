"""Prevent upstream CLI keyword arguments from drifting out of train_family().

The real CLI main() call is checked against the real function AST, not a copied
whitelist. A negative control deletes one parameter from the parsed signature.
This catches training-start failures before any model or GPU is needed.
"""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src/fizgig/families/train.py"


def _contract(source):
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    trainer = functions["train_family"]
    main = functions["main"]
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "train_family"]
    assert len(calls) == 1, "Expected one direct main() -> train_family() call"
    call = calls[0]
    assert all(kw.arg for kw in call.keywords), "Dynamic **kwargs must be verified separately"
    params = {a.arg for a in trainer.args.posonlyargs + trainer.args.args + trainer.args.kwonlyargs}
    kwargs = {kw.arg for kw in call.keywords}
    return kwargs - params, kwargs, params


def test_main_cli_arguments_are_accepted_by_real_train_family_signature():
    missing, passed, accepted = _contract(SOURCE.read_text(encoding="utf-8"))
    assert not missing, f"main() passes keyword arguments rejected by train_family(): {sorted(missing)}"
    for required in ("training_maps", "training_maps_images", "recaption_replace_rules"):
        assert required in passed
        assert required in accepted


def test_negative_control_exposes_removed_upstream_training_maps_parameter():
    source = SOURCE.read_text(encoding="utf-8")
    original = "training_maps=False, training_maps_images=3,"
    assert source.count(original) == 1
    broken = source.replace(original, "training_maps_images=3,", 1)
    missing, _, _ = _contract(broken)
    assert "training_maps" in missing
