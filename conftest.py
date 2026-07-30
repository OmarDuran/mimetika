"""Make ``src/`` and the repository root importable without an editable install.

The root goes on the path too so ``tests/benchmarks`` can import the
``benchmarks.contact_mechanics`` package, which is a runnable study rather than
part of the library and so deliberately lives outside ``src/``.
"""

import sys
from pathlib import Path

root = Path(__file__).parent
for path in (root / "src", root):
    sys.path.insert(0, str(path))
