"""Make ``src/`` and the repository root importable without an editable install.

The root goes on the path too so ``tests/benchmarks`` can import the ``benchmarks``
packages, which are runnable studies and live outside ``src/``.
"""

import sys
from pathlib import Path

root = Path(__file__).parent
for path in (root / "src", root):
    sys.path.insert(0, str(path))
