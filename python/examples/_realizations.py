r"""The stress realizations the examples offer, product and formulation together.

A REALIZATION is a product together with the formulation it is solved in, and the
two are not independent.  The diagonal members exist only in the deviatoric form
-- the plain compliance couples the traction components through the trace and
cannot be diagonal -- and each adaptive member inherits that demand.  So the
formulation is not a free dial beside the product; it is part of naming the
method.

The BDM and VEM products admit both, and the two are DIFFERENT DISCRETIZATIONS
rather than one with a field appended:

    three fields    sigma, u, gamma.  The star carries the full compliance
                    C^-1 = (1/2mu)(sigma - a tr(sigma) I), a = lambda/(2mu + d lambda),
                    which stays bounded as nu -> 1/2 and goes singular on the
                    trace.
    deviatoric      sigma, u, gamma, p.  The star drops to the deviatoric
                    compliance, (2mu)^-1 isotropically, and the hydrostatic
                    stress p closes the system through kappa T sigma = c_p |E| p,
                    kappa = deviatoric_reference and c_p = d/(2mu) + 1/lambda.
                    The deviatoric star is bounded, so lambda^-1 = 0 is
                    admissible -- that limit is Stokes.

They also solve differently.  Measured at nu = 0.3, 0.49, 0.499, 0.4999 on a
cartesian mesh: three fields runs 25, 33, 39, 54 and four fields 45, 94, 159, 394,
so the three-field forms are the ones to reach for near incompressibility.  Both
are h-robust and both are bounded under a jump in the material.

Hence `_deviatoric` in the name: `--product stabilized_vem` is the plain form and
`--product stabilized_vem_deviatoric` the deviatoric one, and there is no
separate formulation flag to contradict it.
"""

import mimetika_cxx as mk

W = mk.StressFormulation.weak_symmetry
WD = mk.StressFormulation.weak_symmetry_deviatoric
S = mk.StressFormulation.strong_symmetry
SD = mk.StressFormulation.strong_symmetry_deviatoric

STRESS = {
    "derham_bdm": (mk.StressRealization.derham_bdm, W),
    # the facet-frame sigma_dev/sigma_hyd split: the same three field in the
    # facet's own frame, so the normal and tangential tractions are separate
    # dofs and a datum reaches each of them
    "derham_bdm_deviatoric": (mk.StressRealization.derham_bdm, WD),
    # unisolvent as a space, and refused by the model: its weak-symmetry inf-sup
    # degenerates. Offered so the refusal is reachable rather than hidden.
    "derham_rt": (mk.StressRealization.derham_rt, W),
    "stabilized_bdm": (mk.StressRealization.stabilized_bdm, W),
    "stabilized_bdm_deviatoric": (mk.StressRealization.stabilized_bdm, WD),
    # the weak two-point star and its per-cell selection: deviatoric only
    "diagonal_afw": (mk.StressRealization.diagonal_afw, WD),
    "adaptive_afw": (mk.StressRealization.adaptive_afw, WD),
    # the strong family (Dassi-Lovadina-Visinoni), a 3D construction
    "stabilized_vem": (mk.StressRealization.stabilized_vem, S),
    "stabilized_vem_deviatoric": (mk.StressRealization.stabilized_vem, SD),
    "diagonal_vem": (mk.StressRealization.diagonal_vem, SD),
    "adaptive_vem": (mk.StressRealization.adaptive_vem, SD),
}

#: the strong family: symmetry in the space, q = d(d+1)/2 traction moments a
#: facet, so the two VEM stress reconstructions apply
STRONG = ("stabilized_vem", "stabilized_vem_deviatoric", "diagonal_vem", "adaptive_vem")

#: the ones whose star is diagonal, hence the per-cell selection eta in {0, 1}
ADAPTIVE = ("adaptive_afw", "adaptive_vem")
TWO_POINT = ("diagonal_afw", "diagonal_vem")

#: blend -> (reconstructed member, diagonal star): M(eta) = eta M_rec + (1 - eta)
#: M_diag per cell, entrywise over the layout the two share
ADAPTIVE_MEMBERS = {
    "adaptive_afw": ("stabilized_bdm", "diagonal_afw"),
    "adaptive_vem": ("stabilized_vem", "diagonal_vem"),
}


def blend_of(name):
    """The blend a diagonal star or a blend belongs to, or None."""
    for blend, (_, diag) in ADAPTIVE_MEMBERS.items():
        if name in (blend, diag):
            return blend
    return None


def add_adaptive_arguments(ap):
    """The two eta selectors, for every example that offers the blends."""
    ap.add_argument(
        "--degeneracy-percent",
        type=float,
        default=None,
        help="eta_E = 0 where |E| falls below this percent of the mean measure of "
             "its node star: an admissibility condition on the local moment "
             "problem, which loses rank as the measure collapses relative to its "
             "star. Independent of cond(M_E). exokal imposes its own such "
             "threshold whatever this says, so the set can only widen.",
    )
    ap.add_argument(
        "--cond-threshold",
        type=float,
        default=None,
        help="eta_E = 0 where the reconstructed member's block has cond(M_E) = "
             "lambda_max/lambda_min above this. Each selector only sets eta_E = 0, "
             "so the two commute and the selection is their union.",
    )


def apply_adaptive(model, name, args):
    """Hand the selectors to the model before it builds; inert off the blends."""
    if name not in ADAPTIVE:
        return
    if args.degeneracy_percent is not None:
        model.set_degeneracy_percent(args.degeneracy_percent)
    if args.cond_threshold is not None:
        model.set_cond_threshold(args.cond_threshold)


def adaptive_line(model, name, args):
    """Where the selection fell, after the build: the cells at each end."""
    rec, diag = ADAPTIVE_MEMBERS[name]
    n_diag = int((model.eta == 0.0).sum())
    pct = args.degeneracy_percent
    line = (f"  {name}: {n_diag} cell(s) on {diag}, {model.n_cells - n_diag} on {rec} "
            f"(threshold {'default' if pct is None else f'{pct}%'}")
    if args.cond_threshold is not None:
        line += f"; {model.n_ill_conditioned} switched by cond > {args.cond_threshold:g}"
    return line + ")"


def names():
    return sorted(STRESS)


def resolve(name):
    """(product, formulation) for a realization name."""
    return STRESS[name]


def fields(name):
    """How many fields the realization carries: 3 or 4."""
    return 4 if STRESS[name][1] in (WD, SD) else 3


def describe(name):
    """One line for the run's own report."""
    product, formulation = STRESS[name]
    return (f"{mk.stress_realization_name(product)}, "
            f"{mk.stress_formulation_name(formulation)} ({fields(name)} fields)")


def reject_formulation_flag(value):
    """--formulation is gone: the product names the pair.

    Raised rather than ignored: an ignored flag leaves the caller with a
    different discretization from the one it asked for.
    """
    if value is None:
        return
    deviatoric = str(value).endswith("_deviatoric")
    raise SystemExit(
        "--formulation was removed: the product names the formulation. Use "
        + ("--product <name>_deviatoric for four fields" if deviatoric
           else "--product <name> for three fields")
        + f", one of: {', '.join(names())}"
    )
