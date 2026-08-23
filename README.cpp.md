# mimetika (C++)

mimetika is the **application**. It consumes exokal for discrete operators,
local automatic differentiation and assembly, and graphos for topology
beneath that; everything it adds is physics.

    graphos      topology, metric-free d
    exokal       discrete operators and assembly: ⋆, local AD, spaces, forms, time
    mimetika     models, closures, catalogue, drivers

## Layout

    include/mimetika/physics/package.hpp     Package, Requirements, Composition
    include/mimetika/physics/catalogue.hpp   named models, registered at static init
    include/mimetika/physics/flow.hpp        the Flow package
    include/mimetika/models/                 catalogue entries — declarations only
    apps/run.cpp                             the driver
    tests/                                   C++ tests beside the superseded Python suite

## The discipline

A catalogue entry is a **composition of packages**, a few lines long. If one
takes more than that, a package is missing and the right response is to write
the package. The catalogue grows as a product — physics times domain times
discretization — while the code grows as a sum.

Three axes must multiply by zero: poromechanics is flow ⊕ mechanics ⊕ a
coupling package, single-phase flow is compositional flow at one component,
and the domain type is a property of the mesh and the driver rather than of a
model. The reasons are in `docs/model_catalogue.md`; exokal's own boundary is
`../exokal/docs/scope.md`.

## Building

    cmake -B build && cmake --build build && ctest --test-dir build

exokal is found as an installed package or as the sibling checkout at
`../exokal`.

## The Python that is still here

`src/mimetika/` is the original Python research code. It is superseded as
porting proceeds; what should remain afterwards is a thin layer that
orchestrates models — never one that evaluates physics.
