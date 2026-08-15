"""Python interface to the mimetika C++ application.

This is the compiled stack -- graphos topology, exokal operators, mimetika
models -- reached through pybind11. It is *not* the pure-Python implementation
in ``../src/mimetika``; both are importable at once, on purpose, so that a
disagreement between them shows up as a failing test instead of as whichever
one happened to be on the path.

The surface is the one the fixed-dimensional model tests need: a mesh, a stress
or flux realization, a model assembled on it, boundary conditions imposed as
forms, a direct solve, and the DOF addresses to read the answer back with.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__ as _core_doc  # noqa: F401
