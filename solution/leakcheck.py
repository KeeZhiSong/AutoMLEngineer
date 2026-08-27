"""
leakcheck.py -- FROZEN. Static leakage detection for agent-generated code.

WHY THIS EXISTS. A previous run scored 0.6449 against a 0.6016 baseline and the
loop KEPT it. The winning "feature" was:

    history = group['is_click'].rolling(window=5, min_periods=1).sum()

`is_click` and `long_view` are outcomes of the SAME impression -- you cannot know
whether a user clicked before deciding how to rank the video -- and the window
included the current row, so each row's feature contained its own click. The
earlier guard missed it because it compared FIELD NAMES against the feedback list
and the derived field was called `user_history_interest`.

The lesson: check what the code READS, not what it CALLS things.

Rules by module, and by function within it:
  features.py fit()     -- MAY read train labels. fit() only ever receives the
                           training split, so aggregating the label into a
                           FeatureState is target encoding, not leakage.
  features.py elsewhere -- may not. transform() receives valid/test, so reading
                           labels there scores a split with its own answers.
  model.py              -- may not. build() receives a FeatureState, never data;
                           anything label-derived must be computed in fit() and
                           carried across in the state.
  train.py              -- may reference them. Auxiliary multi-task objectives
                           are a legitimate, organiser-suggested direction, and
                           train.py cannot inject them into X (features.py owns
                           that), so the risk does not arise.
"""

from __future__ import annotations

import ast

from .dataset import FEEDBACK_COLUMNS, LABEL

FORBIDDEN = frozenset(FEEDBACK_COLUMNS) | {LABEL}
INPUT_MODULES = ("features.py", "model.py")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are docstrings, so prose may discuss the rule."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _fit_scoped(tree: ast.AST) -> set[int]:
    """ids of nodes lexically inside fit(), where label access is LEGITIMATE.

    fit() receives ONLY the training split, by construction -- runner.py calls
    `features.fit(ds.train)` and nothing else. Aggregating the label over train
    into a FeatureState is ordinary target encoding (a smoothed per-video
    long_view rate, say), not leakage. transform() is the dangerous one: it
    receives valid and test, so reading labels there scores a split using its
    own answers.

    This distinction was added after the guard blocked a legitimate
    popularity-prior idea -- a guard that is too strict has a real cost, it just
    shows up as forgone ideas instead of as a bad number.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == "fit":
            for child in ast.walk(node):
                out.add(id(child))
    return out


def find_leaks(source: str, module: str) -> list[str]:
    """Return human-readable leak findings. Empty list means clean."""
    if module not in INPUT_MODULES:
        return []

    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    # The fit() exemption applies ONLY to features.py. model.py's build()
    # receives a FeatureState and never touches data, so a function named fit()
    # there has no legitimate reason to read a label -- and must not inherit the
    # exemption just by sharing a name.
    if module == "features.py":
        skip = skip | _fit_scoped(tree)
    found: list[str] = []

    for node in ast.walk(tree):
        # log['is_click'] / df.get('is_click') / feedback('is_click')
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in FORBIDDEN and id(node) not in skip:
                found.append(
                    f"line {node.lineno}: reads {node.value!r}, a feedback/label "
                    f"column -- it is an outcome of the same impression being "
                    f"ranked and cannot be a model input"
                )
        # log.is_click
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN \
                and id(node) not in skip:
            found.append(
                f"line {node.lineno}: attribute access .{node.attr} on a "
                f"feedback/label column"
            )
        # split.feedback(...) is the auxiliary-target API; not valid in inputs
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "feedback" and id(node) not in skip:
            found.append(
                f"line {node.lineno}: calls .feedback() -- that API supplies "
                f"auxiliary TARGETS for train.py, never model inputs"
            )
    return found


def assert_clean(source: str, module: str) -> None:
    """Raise with an actionable message if the code leaks."""
    leaks = find_leaks(source, module)
    if leaks:
        raise ValueError(
            f"Label leakage in {module}:\n" + "\n".join(f"  - {l}" for l in leaks)
            + "\n\nFeedback columns (is_click, is_like, play_time_ms, ...) and the "
              "label (long_view) are outcomes of the impression being ranked. "
              "Using one as an input inflates validation and is not a real model. "
              "For user history, derive it from PRIOR impressions only, shifted so "
              "the current row is excluded, and fitted on train alone."
        )
