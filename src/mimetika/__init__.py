"""mimetika: efficient construction of mimetic operators for PDE.

Layered, single-responsibility architecture:

    topology   -> combinatorial complex, signed incidence (metric-free)
    geometry   -> coordinates and metric quantities
    mesh       -> topology + geometry container, generators/readers
    dof        -> numbering of differential-form degrees of freedom
    operators  -> exterior derivative (topology) + Hodge/mass (geometry)
    assembly   -> global operators, linear systems, PETSc/scipy backends
    solver     -> linear solve (PETSc KSP, scipy fallback), saddle point,
                  static condensation
    contact    -> fracture contact laws, algebraic contact map, driver
    simulation -> poromechanics solver, time stepping
    postprocess-> reconstruction, VTK/VTU export, error norms
"""

__version__ = "0.1.0"
