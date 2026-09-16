"""Make ``src/``, ``python/`` and the repository root importable without an
editable install.

The root goes on the path too so ``tests/benchmarks`` can import the ``benchmarks``
packages, which are runnable studies and live outside ``src/``.

And make THIS tree the one that is tested. An editable install of another
checkout outranks every insert below, because the finder it leaves on
``sys.meta_path`` is consulted before ``sys.path`` at all; ``python/tests/
_shadowing.py`` says what that costs and undoes it.
"""

import sys
from pathlib import Path

root = Path(__file__).parent
# Highest priority LAST: each insert goes in front of the one before it, so the
# root ends up ahead of ``python`` and ``tests`` keeps meaning ``<root>/tests``
# rather than ``<root>/python/tests``.
for path in (root / "python", root / "src", root):
    sys.path.insert(0, str(path))

# ``python`` is on the path from the loop above, so the helper the examples and
# the mpi script also use resolves from this tree
from _shadowing import prefer_this_tree  # noqa: E402

_unshadowed = prefer_this_tree(root)


def pytest_report_header():
    """Say so in the header when a redirect was undone.

    Silence is what made this worth fixing: the suite passed against the wrong
    tree and said nothing either way.
    """
    if not _unshadowed:
        return None
    return (
        "unshadowed an editable install of another checkout: "
        + ", ".join(_unshadowed)
    )
