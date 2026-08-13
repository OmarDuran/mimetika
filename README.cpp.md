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
discretization — while the code grows as a sum, and that gap is the only
thing that makes the catalogue survivable.

Three axes must multiply by zero, and each for a specific reason:

- **Poromechanics is not a physics.** It is flow ⊕ mechanics ⊕ a coupling
  package that reads a pressure and a displacement without caring how many
  components produced either.
- **Single-phase flow is compositional flow at one component.** A field
  count, not a different equation — provided a term takes its arity from the
  fluid rather than from its own type.
- **The domain type is not a model property.** Fixed, static
  mixed-dimensional and dynamic are properties of the mesh and the driver.
  exokal's stratified epoch already carries a term across every codimension
  it applies to. A package that branches on the domain has reintroduced the
  multiplication this layer exists to prevent.

Design notes: `docs/model_catalogue.md`. exokal's own boundary:
`../exokal/docs/scope.md`.

## Building

    cmake -B build && cmake --build build && ctest --test-dir build

exokal is found as an installed package or as the sibling checkout at
`../exokal`.

## The Python that is still here

`src/mimetika/` is the original Python research code. It is superseded as
porting proceeds; what should remain afterwards is a thin layer that
orchestrates models — never one that evaluates physics.
