"""Make the tree this example lives in the one it imports.

IMPORT BEFORE mimetika_cxx. An editable install of another checkout leaves a
finder on ``sys.meta_path`` that resolves ``mimetika_cxx`` to the tree it was
installed from, and a meta-path finder outranks ``sys.path`` entirely -- so an
example run from a git worktree silently exercises the other checkout, and a
route added here reads as missing. ``python/_shadowing.py`` says what that costs.

What is imported is then this tree's own CMake build: `cmake --build build` refreshes it.
"""

import sys
from pathlib import Path

# <root>/python: where mimetika_cxx and _shadowing both live in THIS tree
_python = Path(__file__).resolve().parents[1]
if str(_python) not in sys.path:
    sys.path.insert(0, str(_python))

from _shadowing import prefer_this_tree  # noqa: E402

undone = prefer_this_tree(_python.parent)
