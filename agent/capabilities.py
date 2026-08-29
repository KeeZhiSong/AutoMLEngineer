"""
Capability map: what the editable code ALREADY supports, derived by static analysis.

WHY THIS EXISTS. In run 12 the planner scored `List-Length Re-batching` at
isolation 2, cheapness 2 -- invasive and expensive -- and ranked it LAST of four.
It is in fact a config-level change: `eval_sized_groups()`, `user_batches(...,
group_size=)`, `segment_softmax()` and the `batch_mode` switch all already exist
in train.py. The model was estimating implementation cost from generic intuition
because it had never read the codebase, and its prior was exactly backwards --
"re-batching" sounds like surgery, "loss weighting" sounds surgical.

This reports FACTS ONLY: which functions exist, which config keys are read, and
which config values gate which code paths. It never recommends an intervention
and never contains a tuned value. `group_size` appears as a parameter name; the
number 5 does not appear anywhere, because that is the answer.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger("amra.capabilities")

EDITABLE = ("features.py", "model.py", "train.py")
ROOT = Path(__file__).resolve().parent.parent / "solution"


def _config_keys(tree: ast.AST) -> list[tuple[str, str]]:
    """Every cfg.get("key", default) the module reads."""
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            default = ""
            if len(n.args) > 1:
                try:
                    default = ast.unparse(n.args[1])
                except Exception:                          # noqa: BLE001
                    default = "?"
            out.append((n.args[0].value, default))
    seen, uniq = set(), []
    for k, d in out:
        if k not in seen:
            seen.add(k)
            uniq.append((k, d))
    return uniq


def _functions(tree: ast.AST) -> list[str]:
    """Top-level function signatures, so the planner sees what it can call."""
    out = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            args = [a.arg for a in n.args.args]
            out.append(f"{n.name}({', '.join(args)})")
    return out


def _gated_keys(tree: ast.AST) -> list[str]:
    """Config keys that are only READ inside a branch another config value gates.

    MEASURED, run 15: the planner produced `group_size` in one plan and
    `batch_mode="user"` in another, and neither did anything -- `group_size` is
    read only inside the `batch_mode == "user"` branch, so setting it alone is
    inert. The map listed both keys and never said one gates the other, so the
    agent had both facts and no link between them.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        gate = None
        if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                and node.test.comparators
                and isinstance(node.test.comparators[0], ast.Constant)):
            gate = (node.test.left.id, node.test.comparators[0].value)
        if gate is None:
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "get" and inner.args
                    and isinstance(inner.args[0], ast.Constant)):
                out.append(f"`{inner.args[0].value}` is read ONLY when "
                           f"`{gate[0]} == \"{gate[1]}\"` -- setting it alone "
                           f"does nothing")
    return sorted(set(out))


def _gated_paths(src: str, tree: ast.AST) -> list[str]:
    """Config values that SELECT a code path -- the cheap levers.

    A comparison of a config-derived name against a string constant is a switch:
    the branch already exists and setting the value turns it on. This is the
    class of change the planner most badly misprices, because no new code is
    needed at all.
    """
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name):
            for c in n.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.append(f"`{n.left.id} == \"{c.value}\"` selects a branch "
                               f"that already exists")
    return sorted(set(out))


def _objective_facts(tree: ast.AST) -> list[str]:
    """Which objective is in force, and whether it is a GROUPED one.

    MEASURED, run 15 follow-up: the planner assembled `batch_mode="user"` plus
    `group_size` correctly and never touched the objective. Grouped batching
    under a pointwise loss is a measured NO-OP, so that patch would satisfy its
    contract, score baseline, and be recorded as evidence AGAINST the correct
    hypothesis. The map has to say that grouping only bites on a loss that
    consumes the grouping.

    The test is structural, not semantic: a loss taking `user_id` sees which
    rows share a list; one taking only (model, X, y) cannot.
    """
    out, losses = [], {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            args = [a.arg for a in n.args.args]
            if {"model", "X", "y"} <= set(args):
                losses[n.name] = "user_id" in args
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "fit":
            for arg, default in zip(n.args.args[-len(n.args.defaults):] if n.args.defaults else [],
                                    n.args.defaults):
                if arg.arg == "loss_fn" and isinstance(default, ast.Name):
                    cur = default.id
                    grouped = losses.get(cur)
                    out.append(f"the objective currently in force is `{cur}`")
                    if grouped is False:
                        out.append(
                            f"`{cur}` does NOT take `user_id`, so it cannot see "
                            f"which rows share a list -- it is NOT a grouped "
                            f"objective. Changing how rows are grouped has NO "
                            f"effect on a loss that does not consume the grouping.")
                    elif grouped:
                        out.append(f"`{cur}` takes `user_id`, so it is a grouped "
                                   f"objective and grouping affects it")
    if losses:
        out.append("loss functions defined here: " + ", ".join(
            f"{k} ({'grouped' if v else 'not grouped'})" for k, v in losses.items()))
    return out


def capability_map(root: Path | str = ROOT) -> str:
    """Human-readable map of what the editable modules already do."""
    root = Path(root)
    blocks = []
    for name in EDITABLE:
        path = root / name
        if not path.exists():
            continue
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            blocks.append(f"{name}\n  (does not parse)")
            continue

        fns = _functions(tree)
        keys = _config_keys(tree)
        gates = _gated_paths(src, tree)
        gated_keys = _gated_keys(tree)
        objective = _objective_facts(tree)

        lines = [name, "-" * len(name)]
        if fns:
            lines.append("  functions that already exist:")
            lines += [f"    {f}" for f in fns]
        if keys:
            lines.append("  config keys this module reads (name: default):")
            lines += [f"    {k}: {d}" for k, d in keys]
        if gates:
            lines.append("  config-selected code paths (no new code needed):")
            lines += [f"    {g}" for g in gates]
        if gated_keys:
            lines.append("  KEYS THAT DEPEND ON ANOTHER KEY (set both or neither):")
            lines += [f"    {g}" for g in gated_keys]
        if objective:
            lines.append("  CURRENT OBJECTIVE:")
            lines += [f"    {o}" for o in objective]
        blocks.append("\n".join(lines))

    return ("WHAT THE CODEBASE ALREADY SUPPORTS (static analysis, facts only --\n"
            "this does not suggest an intervention):\n\n" + "\n\n".join(blocks))


def legal_config_keys(root: Path | str = ROOT) -> dict[str, str]:
    """{key: unparsed default} for every config key the editable modules read.

    Used to validate a plan's activation config. Run 15 emitted
    `{"loss_function": "focal_loss"}` -- not a key anything reads -- and
    `{"group_size": "config_value"}`, a placeholder string where an int belongs,
    which cost one cycle 16 minutes before it failed.
    """
    out: dict[str, str] = {}
    for name in EDITABLE:
        path = Path(root) / name
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for k, d in _config_keys(tree):
            out.setdefault(k, d)
    return out


def validate_config(cfg: dict, root: Path | str = ROOT) -> tuple[dict, list[str]]:
    """Drop keys nothing reads and values whose type cannot work. Returns (kept, dropped)."""
    legal = legal_config_keys(root)
    kept, dropped = {}, []
    for k, v in (cfg or {}).items():
        if k not in legal:
            dropped.append(f"{k} (no module reads it)")
            continue
        default = legal[k]
        numeric_default = default not in ("", "None") and not default.startswith(("'", '"'))
        if numeric_default and isinstance(v, str):
            dropped.append(f"{k}={v!r} (expected a number, got a string)")
            continue
        if k == "group_size" and not isinstance(v, int):
            dropped.append(f"{k}={v!r} (must be an int)")
            continue
        kept[k] = v
    return kept, dropped
