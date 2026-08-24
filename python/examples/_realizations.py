r"""The stress realizations the examples offer, product and formulation together.

A REALIZATION is a product together with the formulation it is solved in, and the
two are not independent.  The diagonal members exist only with the total pressure
-- the plain compliance couples the traction components through the trace and
cannot be diagonal -- and each blend inherits that demand.  So the formulation is
not a free dial beside the product; it is part of naming the method.

The BDM and VEM products admit both, and the two are DIFFERENT DISCRETIZATIONS
rather than one with a field appended:

    three fields    sigma, u, gamma.  The star carries the full compliance
                    C^-1 = (1/2mu)(sigma - a tr(sigma) I), a = lambda/(2mu + d lambda),
                    which stays bounded as nu -> 1/2 and goes singular on the
                    trace.
    four fields     sigma, u, gamma, p.  The trace moves to p, the star is built
                    at lambda = 0 and is lambda-free, and lambda lives in the
                    pressure row as c_p = d/(2mu) + 1/lambda.

They also solve differently.  Measured at nu = 0.3, 0.49, 0.499, 0.4999 on a
cartesian mesh: three fields runs 25, 33, 39, 54 and four fields 45, 94, 159, 394,
so the three-field forms are the ones to reach for near incompressibility.  Both
are h-robust and both are bounded under a jump in the material.

Hence `_total` in the name: `--product stabilized_bdm` is three fields and
`--product stabilized_bdm_total` is four, and there is no separate formulation
flag to contradict it.
"""

import mimetika_cxx as mk

W = mk.StressFormulation.weak_symmetry
WT = mk.StressFormulation.weak_symmetry_total
S = mk.StressFormulation.strong_symmetry
ST = mk.StressFormulation.strong_symmetry_total

#: name -> (product, formulation). The name IS the pair.
STRESS = {
    "derham_bdm": (mk.StressRealization.derham_bdm, W),
    "derham_bdm_total": (mk.StressRealization.derham_bdm, WT),
    # unisolvent as a space, and refused by the model: its weak-symmetry inf-sup
    # degenerates. Offered so the refusal is reachable rather than hidden.
    "derham_rt": (mk.StressRealization.derham_rt, W),
    "stabilized_bdm": (mk.StressRealization.stabilized_bdm, W),
    "stabilized_bdm_total": (mk.StressRealization.stabilized_bdm, WT),
    # the weak two-point star and its blend: four fields only
    "diagonal_afw": (mk.StressRealization.diagonal_afw, WT),
    "adaptive_afw": (mk.StressRealization.adaptive_afw, WT),
    # the strong family (Dassi-Lovadina-Visinoni), a 3D construction
    "stabilized_vem": (mk.StressRealization.stabilized_vem, S),
    "stabilized_vem_total": (mk.StressRealization.stabilized_vem, ST),
    "diagonal_vem": (mk.StressRealization.diagonal_vem, ST),
    "adaptive_vem": (mk.StressRealization.adaptive_vem, ST),
}

#: the ones whose star is diagonal, hence the per-cell selection and its blend
ADAPTIVE = ("adaptive_afw", "adaptive_vem")
TWO_POINT = ("diagonal_afw", "diagonal_vem")


def names():
    return sorted(STRESS)


def resolve(name):
    """(product, formulation) for a realization name."""
    return STRESS[name]


def fields(name):
    """How many fields the realization carries: 3 or 4."""
    return 4 if STRESS[name][1] in (WT, ST) else 3


def describe(name):
    """One line for the run's own report."""
    product, formulation = STRESS[name]
    return (f"{mk.stress_realization_name(product)}, "
            f"{mk.stress_formulation_name(formulation)} ({fields(name)} fields)")


def reject_formulation_flag(value):
    """--formulation is gone: the product names the pair.

    Raised rather than ignored, because a script passing the old flag meant
    something by it and would otherwise get a different discretization in
    silence.
    """
    if value is None:
        return
    total = str(value).endswith("_total")
    raise SystemExit(
        "--formulation was removed: the product names the formulation. Use "
        + ("--product <name>_total for four fields" if total
           else "--product <name> for three fields")
        + f", one of: {', '.join(names())}"
    )
