"""Shared helpers for the fault-mesh examples."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow running the examples straight from a checkout, without installing.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MESH_PATH = Path(__file__).resolve().parents[1] / "meshes" / "fault_mesh.vtu"


class step:
    """Context manager that times and labels a stage of the run."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        print(f"  {self.label} ...", end="", flush=True)
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        print(f" {time.perf_counter() - self._t:7.1f} s")
        return False


def load_mesh(path: Path = MESH_PATH, box: tuple | None = None, validate: bool = True):
    """Read the fault mesh, optionally restricted to a centroid bounding box."""
    from mimetika.mesh.readers import check_orientation, read_vtu

    if not path.exists():
        raise SystemExit(f"mesh not found: {path}")

    with step(f"reading {path.name}"):
        mesh = read_vtu(path)

    if box is not None:
        with step("extracting subregion"):
            ids = mesh.cells_in_box(*box)
            if len(ids) == 0:
                raise SystemExit(f"no cells with centroid in {box}")
            mesh = mesh.subset(ids)

    with step("computing geometry"):
        mesh.geometry.measure(3)
        mesh.geometry.centroids(3)

    if validate:
        with step("validating topology"):
            if not mesh.complex.verify_complex():
                raise SystemExit("mesh failed the dd = 0 check")
            check_orientation(mesh)

    return mesh


def report_mesh(mesh) -> None:
    faces = np.array(
        [len(mesh.complex.facets_of(3, c)) for c in range(mesh.num_cells(3))]
    )
    p = mesh.geometry.points
    vol = mesh.geometry.measure(3)
    print(
        f"\n  cells {mesh.num_cells(3):>8d}   facets {mesh.num_cells(2):>8d}"
        f"   edges {mesh.num_cells(1):>8d}   vertices {mesh.num_cells(0):>8d}"
    )
    print(
        f"  faces/cell  mean {faces.mean():.2f}  max {faces.max()}"
        f"   |  cell volume  min {vol.min():.2e}  total {vol.sum():.6f}"
    )
    print(
        "  bounding box  "
        + " x ".join(f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(p.min(0), p.max(0)))
    )
    simplices = sum(mesh.complex.is_simplex(c) for c in range(mesh.num_cells(3)))
    print(f"  simplicial cells {simplices} / {mesh.num_cells(3)}")


def add_common_args(parser, default_vtu: str = "solution.vtu") -> None:
    parser.add_argument(
        "--backend",
        choices=["auto", "petsc", "scipy"],
        default="auto",
        help="linear-algebra backend (auto uses PETSc when petsc4py is importable)",
    )
    parser.add_argument(
        "--method",
        choices=["minres", "direct"],
        default="minres",
        help="direct LU, or MINRES with a Schur-complement block preconditioner",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-10, help="iterative solver tolerance"
    )
    parser.add_argument(
        "--pc",
        choices=["cpr", "schur"],
        default="cpr",
        help=(
            "MINRES preconditioner: 'cpr' applies AMG to the Schur (elliptic) "
            "block and a cheap factorisation to the leading block, both once; "
            "'schur' uses PETSc's defaults (an inner GMRES solve per block)."
        ),
    )
    parser.add_argument(
        "--petsc-opts",
        type=str,
        default=None,
        metavar="STR",
        help=(
            'PETSc options passed straight to the options database, e.g. '
            '--petsc-opts "-ksp_view -ksp_monitor" to have PETSc itself report '
            "the exact solver stack it is running."
        ),
    )
    parser.add_argument(
        "--vtu",
        type=Path,
        nargs="?",
        default=None,
        const=Path(default_vtu),
        metavar="PATH",
        help=(
            "write the post-processed solution to a .vtu file, with the "
            "polyhedral cells preserved.  Use the flag alone for the default "
            f"name ({default_vtu}), or give a path."
        ),
    )


def announce_backend(backend: str) -> str:
    from mimetika.assembly.backend import petsc_available

    resolved = backend
    if backend == "auto":
        resolved = "petsc" if petsc_available() else "scipy"
    if resolved == "petsc":
        from petsc4py import PETSc

        v = PETSc.Sys.getVersionInfo()
        print(f"  backend: PETSc {v['major']}.{v['minor']}.{v['subminor']}")
    else:
        if backend == "petsc":
            raise SystemExit("petsc4py is not importable; install it or use --backend scipy")
        print("  backend: scipy (petsc4py not available)")
    return resolved


def summarise(name: str, values: np.ndarray, per: int = 1) -> None:
    v = np.asarray(values).reshape(-1, per)
    mag = np.linalg.norm(v, axis=1) if per > 1 else np.abs(v).ravel()
    print(
        f"    {name:<26s} min {v.min():+.4e}  max {v.max():+.4e}  "
        f"|.|max {mag.max():.4e}"
    )
