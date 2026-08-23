# Building

## Dependencies

    graphos    sibling checkout, or an installed package
    exokal     sibling checkout, or an installed package
    basix      the finite element basis engine (exokal requires it)
    PETSc      optional: the linear solver

## The conda environment

Both the C++ and the Python side live in the `mimetika` conda environment.

**It must match the architecture of the C++ build.** On Apple silicon the
native build is arm64, and a conda installed under Rosetta produces x86_64
packages that link against nothing — the symptom is a page of undefined PETSc
symbols "for architecture arm64" while `nm` shows the symbols present. The
environment is therefore created with the subdirectory forced:

    CONDA_SUBDIR=osx-arm64 conda create -n mimetika -c conda-forge \
        python=3.13 petsc petsc4py numpy scipy matplotlib pytest gmsh openpyxl

`PYTHONNOUSERSITE=1` is worth setting when running its Python: a user-site
`~/.local/lib/python3.13/site-packages` from another architecture shadows the
environment's own numpy and fails the same way.

## Configuring

PETSc is found through pkg-config, which every PETSc installation provides:

    P=$CONDA_PREFIX
    PKG_CONFIG_PATH=$P/lib/pkgconfig cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
    DYLD_LIBRARY_PATH=$P/lib ctest --test-dir build

PETSc's own pkg-config advertises only `-lpetsc`, but `PETSC_COMM_SELF`
expands to an MPI symbol, so the MPI it was built against is linked as well —
every PETSc is an MPI build even when it will only ever run on one rank.

Without PETSc the build still works: `-DMIMETIKA_USE_PETSC=OFF`, or simply not
having it. The operators can be assembled and compared without a solver.

## The solver

Direct, through MUMPS. These systems are saddle points with zero diagonal
blocks — the (u, u) and (gamma, gamma) blocks that make the form mixed — and
PETSc's built-in LU refuses them outright:

    mumps  -> converged, residual 0.0e+00
    petsc  -> "Matrix is missing diagonal entries", solve fails

MUMPS pivots symmetrically and handles them without a shift. An iterative
method is the same object with different options — `-ksp_type fgmres -pc_type
fieldsplit` — because the KSP reads them after every choice the code makes.
