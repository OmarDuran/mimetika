"""Which stage an example is in, and how long it took.

The stages that dominate -- reading, assembling, factorizing -- produce no
output of their own, and a silent process is indistinguishable from a hung one.
Every stage names itself before it starts and reports its cost when it ends.
"""

import contextlib
import time


@contextlib.contextmanager
def stage(what):
    print(f"  {what} ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException:
        print(" FAILED", flush=True)
        raise
    print(f" {time.perf_counter() - t0:.2f} s", flush=True)
