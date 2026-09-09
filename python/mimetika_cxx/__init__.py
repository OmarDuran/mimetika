"""Python interface to the mimetika C++ application.

The compiled stack -- graphos topology, exokal operators, mimetika models --
reached through pybind11. It is not the pure-Python implementation in
``src/mimetika``; both are importable at once, so a disagreement between them
is a failing test rather than an import-order accident.

Surface: a mesh, a stress or flux realization, a model assembled on it,
boundary conditions imposed as forms, a solve (direct, Krylov under the Riesz
map, or hybridized), the DOF addresses to read the answer back with, and the
handoff that carries an assembled system to ``mimetika_hypre``.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__ as _core_doc  # noqa: F401
