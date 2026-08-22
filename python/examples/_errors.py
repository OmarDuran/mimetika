r"""The L2 error norms of a cell-wise constant field, and the table reporting them.

Both patch tests measure the same object. Every unknown either IS cell-wise
constant -- a cell moment divided by the measure -- or is reconstructed as a
cell average from the facet moments, and every exact field is affine or
constant, so the error

    e|_E = Pi_0(u - u_h)|_E ,     Pi_0 v|_E = |E|^-1 int_E v ,

is cell-wise constant too. Its norms are then exact sums and no quadrature
enters anywhere:

    ||e||_{L2(E)} = |E|^{1/2} |e_E| ,     ||e||_{L2(D)}^2 = sum_E |E| |e_E|^2 .

Pi_0 of an affine field is its value at the CENTROID, which is what
exokal::centroid returns -- the measure-weighted one, not the vertex mean -- so
the exact side needs no projection either.

WHY THE RELATIVE COLUMN TAKES ITS SCALE AS AN ARGUMENT. A row whose exact field
VANISHES has no scale of its own: the deviator of a pure dilation is zero, and
so is the rotation of a spin-free datum. Dividing by the row's own norm is then
0/0. Each caller therefore names the scale its row is to be read against -- the
stress rows against ||sigma||_D, so the three of them stay comparable -- and
states that choice in its own legend.
"""

import numpy as np


def l2_norms(volume, e):
    """(||e||_{L2(D)}, the per-cell ||e||_{L2(E)}) for a cell-wise constant e.

    `e` is one scalar per cell, or one row per cell, in which case |e_E| is the
    Euclidean norm of the row -- the Frobenius norm, where the row is a
    flattened tensor.
    """
    magnitude = np.abs(e) if e.ndim == 1 else np.linalg.norm(e, axis=1)
    local = np.sqrt(volume) * magnitude
    return float(np.linalg.norm(local)), local


def error_table(volume, rows, width=10):
    """One line per (name, e, scale); returns the relative errors in that order.

    S IS PRINTED, not only divided by. It is the norm of the field the row is
    measured against -- ||Pi_0 v||_{L2(D)}, or the fixed scale a vanishing
    reference borrows -- and without it the relative column cannot be checked
    against the absolute one, nor two runs on different meshes compared: both
    ||e||_D and S carry |D|^{1/2}, and only their ratio does not.

    The extremes are of the LOCAL norms ||e||_{L2(E)}, which carry |E|^{1/2}: on
    a graded mesh the smallest cells sit at the bottom of that column whatever
    their error, so the two ends bound the cells and not the field.
    """
    print(f"    {'':{width}}{'||e||_D':>13}{'S':>13}{'||e||_D / S':>13}"
          f"{'min_E ||e||_E':>15}{'max_E ||e||_E':>15}")
    relative = []
    for name, e, scale in rows:
        norm, local = l2_norms(volume, e)
        relative.append(norm / scale)
        print(f"    {name:{width}}{norm:>13.3e}{scale:>13.3e}{relative[-1]:>13.3e}"
              f"{local.min():>15.3e}{local.max():>15.3e}")
    print()
    return relative

