"""mimetika through hypre's ADS, called directly rather than through PCHYPRE.

A separate module from ``mimetika_cxx``: that one links PETSc, PETSc links its
own libHYPRE, and two copies of hypre in one process export the same names, so
a call resolves by load order. This module links its own hypre with those
symbols unexported, so importing both is safe -- but the two do not share
types, so a mesh built in one cannot be passed to the other.

It reaches what PCHYPRE does not forward: ``amg_theta`` and ``ams_theta``, the
strength thresholds of the auxiliary hierarchies inside ADS, which PCHYPRE
registers as options and never queries, and the MGR reduction. ``solve_system``
takes a system assembled by ``mimetika_cxx.ads_handoff``.
"""

from ._hypre import *  # noqa: F401,F403
from ._hypre import __doc__ as _doc  # noqa: F401
