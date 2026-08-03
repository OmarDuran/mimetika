r"""Mark everything under ``tests/benchmarks`` as ``benchmark``, and deselect it.

These are not unit tests of the source.  They are published-problem reproductions --
Terzaghi consolidation, the Novikov fault-reactivation cases -- that solve real
systems on real meshes and take minutes, not milliseconds.  They answer "does the
library still reproduce the literature", which is a different question from "is this
operator correct", and it should not be asked on every edit.

Marking is done here rather than by decorating each module so that a new benchmark
file is covered the moment it is added, with nothing to remember.

    pytest                      # source tests only -- benchmarks deselected
    pytest -m benchmark         # only the benchmarks
    pytest -m ""                # everything, overriding the default filter

The deselection lives in ``addopts`` in ``pyproject.toml``.  Note that deselected is
not the same as passing: ``pytest -m benchmark`` is what tells you whether the
published results still reproduce, and it is worth running before a release or after
touching an operator, a mesh generator or a solver.
"""

import pathlib

import pytest

HERE = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(items):
    """Mark only the items collected from this directory.

    The hook is invoked for a subdirectory conftest but is handed the *whole*
    session's item list, not just this directory's.  Marking unconditionally would
    tag every test in the project as a benchmark and deselect the entire suite --
    which reads as "3707 deselected, 0 passed" and looks like a passing run.
    """
    for item in items:
        if HERE in pathlib.Path(str(item.path)).parents:
            item.add_marker(pytest.mark.benchmark)
