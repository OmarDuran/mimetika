"""mimetika through hypre's ADS, called directly.

A SEPARATE module from ``mimetika_cxx`` on purpose. That one links PETSc, and
PETSc links its own libHYPRE; two copies of hypre in one process export the
same names and a call reaches whichever the loader saw first. This module links
its own hypre with those symbols hidden, so importing both is safe -- but the
two do not share types, so a mesh built in one cannot be passed to the other.

What it exists for is the part of hypre PETSc does not forward: the strength
thresholds of the auxiliary hierarchies, ``amg_theta`` and ``ams_theta``, which
PCHYPRE registers as options and never queries.
"""

from ._hypre import *  # noqa: F401,F403
from ._hypre import __doc__ as _doc  # noqa: F401
