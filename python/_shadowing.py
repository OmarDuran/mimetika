"""Make the tree a test run lives in outrank an editable install of another one.

``pip install -e python/`` leaves a finder on ``sys.meta_path`` that resolves
``mimetika_cxx`` to the checkout it was installed from. A meta-path finder is
consulted BEFORE ``sys.path`` at all, so neither the ``PYTHONPATH`` the ctest
targets set (python/CMakeLists.txt) nor a conftest's front-inserts decide which
extension module is imported: a suite run from a git worktree silently exercises
the OTHER checkout and reports the result as its own. Nothing fails; the suite
simply describes code that is not the code under test.

The redirect does not even land on that checkout's build. Its
``known_source_files`` sends the package to the installed-from source tree while
its ``known_wheel_files`` sends the extension to the copy under site-packages,
so ``_core`` is whatever ``pip install -e`` last put there -- older than both
trees as soon as either is rebuilt.

Dropping the finder is the whole of it. What the install's ``.pth`` adds to
``sys.path`` is appended during site initialization and therefore sits behind
the conftest's inserts already; only the meta-path redirect jumps the queue.

This fires in EVERY checkout, the installed-from one included: the extension is
redirected to site-packages whichever tree the source redirect names. What a run
imports is then the tree's own CMake build, which is where python/CMakeLists.txt
writes it and what `cmake --build` refreshes.
"""

from __future__ import annotations

import os
import sys


def prefer_this_tree(root) -> list[str]:
    """Drop editable-install redirects resolving outside ``root``.

    Returns the module names that were being redirected, so a caller can report
    what it undid; a second call returns nothing. Call before the first import
    of the redirected package.
    """
    here = os.path.realpath(str(root))

    def outside(path: str) -> bool:
        real = os.path.realpath(path)
        # A WORKTREE CAN LIE UNDER THE CHECKOUT IT WAS MADE FROM --
        # .claude/worktrees does exactly that -- so this asks whether a path
        # falls outside THIS tree, never whether it falls under the other one.
        # The containment runs the wrong way for that second question.
        return real != here and not real.startswith(here + os.sep)

    redirected: dict[str, str] = {}
    kept = []
    for finder in sys.meta_path:
        # BOTH maps, not just the source one. The source map sends the package
        # to the checkout it was installed from; the wheel map sends the
        # EXTENSION to a copy under site-packages, relative to the finder's own
        # directory -- outside every source tree, the installed-from one
        # included. Reading only the first kept the finder in that checkout and
        # loaded whatever `pip install -e` last built, however many times
        # `cmake --build` had run since.
        sources = getattr(finder, "known_source_files", None)
        wheel = getattr(finder, "known_wheel_files", None)
        base = getattr(finder, "dir", None)
        targets = list(sources.values()) if isinstance(sources, dict) else []
        if isinstance(wheel, dict) and base:
            targets += [os.path.join(base, rel) for rel in wheel.values()]
        if targets and any(outside(t) for t in targets):
            redirected.update(sources if isinstance(sources, dict) else {})
            redirected.update(wheel if isinstance(wheel, dict) else {})
        else:
            kept.append(finder)
    if not redirected:
        return []
    sys.meta_path[:] = kept

    # Anything the finder already bound would be handed straight back by the
    # next import. At conftest time there is normally nothing; this says so
    # rather than assuming it.
    packages = {name.split(".")[0] for name in redirected}
    for name in list(sys.modules):
        if name.split(".")[0] in packages:
            del sys.modules[name]

    return sorted(redirected)
