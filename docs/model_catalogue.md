# The model catalogue

*mimetika's design. mimetika is the C++ application; it consumes exokal for
discrete operators, local AD and assembly, and owns everything below —
models, closures, drivers. exokal itself contains no physics and no Python.*

Python drives; the model is C++. The design problem is then how the supported
models are organized so that the catalogue can grow without the code growing
with it.

## The problem

The models we intend to support read as a product:

| axis | variants |
| --- | --- |
| flow | single-phase, compositional multiphase |
| mechanics | linear |
| poromechanics | single-phase, compositional multiphase |
| domain | fixed, static mixed-dimensional, dynamic mixed-dimensional |

Five physics rows against three domain types is fifteen models today, before
thermal and before the discretization choice. Written as fifteen
implementations this is unmaintainable.

**The product is the catalogue. The sum is the code.** Every entry below is a
composition of components that exist once. If adding a model requires writing
a new term, the axes were not orthogonal and that is a layering bug, not a
catalogue problem.

## Where the multiplication goes to zero

Three of the four axes must not multiply, and each has a specific reason.

**Poromechanics is not a physics.** It is flow, plus mechanics, plus the two
Biot coupling terms. `{single-phase poromechanics, compositional multiphase
poromechanics}` is *one* coupling package attached to whichever flow the
composition already contains. The coupling reads `p` and `u` and does not care
how many components produced the pressure. Two catalogue rows, zero
implementations.

**Single-phase flow is compositional flow at n_c = 1.** The distinction is a
field *count*, not a different equation. This only holds if a term takes its
arity from the fluid object rather than from its own type — the constraint
`FluidMobility` already imposes, where the number of composition fields is
instance data. A term that hard-codes "two phases" reintroduces the
multiplication.

**The domain axis is already orthogonal and already built.** `StratifiedEpoch`
gives one global numbering across strata; `On::all()` attaches a term to every
codimension it makes sense on; a face term whose stencil crosses a codimension
is the mixed-dimensional coupling. So:

- *fixed* — one stratum.
- *static mixed-dimensional* — several strata, plus the interface terms. Those
  interface terms are components, counted once, not per physics.
- *dynamic mixed-dimensional* — the above, plus an epoch rebuild and an
  `EntityMap` transfer between epochs. That loop lives in the **driver** and is
  shared by every model.

Nothing in a physics package mentions the domain type. If one ever does, that
is the bug.

**Discretization is orthogonal too**, and was designed that way:
`FluxHodge::Realization` selects the stabilized polytopal product or the de
Rham/RT one behind an identical interface, and the term is opaque to the
choice.

## The count

| component | covers |
| --- | --- |
| `Flow` package | single-phase and compositional multiphase, any n_c, thermal on or off |
| `Mechanics` package | linear mechanics |
| `PoroCoupling` package | both poromechanics rows |
| `Energy` package | the thermal axis |
| `Interface` package | mixed-dimensional coupling terms |

Five components for fifteen catalogue entries; the ratio improves as the
catalogue grows.

## The shape in C++

A package declares what it adds and what it needs, so a composition is
*validated*.

```cpp
namespace exokal::physics {

struct Requirements {
  std::vector<FieldSpec> fields;             // name, DofLayout, which codims
  std::vector<std::string> provides;         // capability tags: "pressure", "displacement"
  std::vector<std::string> needs;            // tags another package must provide
};

class Package {
 public:
  virtual ~Package() = default;
  virtual std::string name() const = 0;
  virtual Requirements requirements() const = 0;
  virtual void declare(SpaceBuilder&) const = 0;               // contribute fields
  virtual void attach(forms::Model&, const TermContext&) const = 0;  // contribute terms
};

}  // namespace exokal::physics
```

A composition builds the space from every package's fields, checks that each
`needs` tag is met by some `provides` tag, and then lets each package attach
its terms:

```cpp
Composition c;
c.add<Flow>({.fluid = &brine, .thermal = true});
c.add<Mechanics>({.material = &rock});
c.add<PoroCoupling>({.biot = 0.8});          // needs "pressure" and "displacement"
auto sim = c.build(mesh, Discretization::mimetic);
```

Composing `PoroCoupling` without `Mechanics` is then a named error at build
time — "PoroCoupling needs 'displacement', provided by no package" — rather
than a term indexing past the end of a stencil.

A catalogue entry is a declaration, not an implementation:

```cpp
// physics/models/compositional_poromechanics.cpp
static const RegisterModel entry{
    "compositional_poromechanics", [](const ModelOptions& o) {
      Composition c;
      c.add<Flow>({.fluid = o.fluid, .thermal = o.thermal});
      c.add<Mechanics>({.material = o.material});
      c.add<PoroCoupling>({.biot = o.biot});
      return c;
    }};
```

Ten lines per model. That is the budget; anything longer means a component is
missing.

## The Python surface

Driver only: it selects a compiled composition, supplies numbers and paths,
runs the loop, and reads results back. Python never defines a term, never
evaluates physics per cell, and never appears in an inner loop.

The closures need the builder in the next section. The composition itself is
one line:

```python
m = exokal.model("compositional_poromechanics", mesh="fractured.vtu",
                 discretization="mimetic", components=3, thermal=True)
# ... closures, per the next section ...
sim = m.build()
sim.solve(t_end=1e6, dt=1e3)
sim.write("out/")
```

## Constitutive slots

The composition above says *which equations*. It says nothing about the
closures, and those are most of what a user actually configures. Single-phase
flow alone requires a porosity law, a density law, a viscosity law, a
permeability field on every stratum, and a normal permeability on every
interface between strata. Each of those is its own catalogue. Flat keyword
arguments cannot express this.

### A slot is the mobility pattern, generalized

A constitutive model is a templated evaluator with a declared dependency —
exactly the shape `FluidMobility` already has:

```cpp
template <class T> T evaluate(const State<T>&) const;
std::vector<std::string> inputs() const;      // which fields it reads
```

Instantiated at `double` it is a residual contribution; at `ad::Local` its
derivatives are exact. So the constitutive layer introduces no new mechanism.
It generalizes one slot to many.

**The declaration is load-bearing.** A `poroelastic` porosity reads the
volumetric strain, so choosing it *creates* the (p, u) Jacobian block that a
`constant` porosity never touches. The choice of a closure changes the
sparsity pattern, and `inputs()` is how the assembler learns that before it
assembles anything.

### Three binding scopes

There are exactly three scopes, and every slot belongs to exactly one.

| scope | slots | bound to |
| --- | --- | --- |
| fluid | density, viscosity, enthalpy, fluid conductivity | the phase — one set per fluid |
| rock | porosity, permeability, rock compressibility, rock conductivity | **each stratum** |
| interface | normal permeability, normal conductivity, contact law | **each stratum pair** |

The out-of-plane permeability governing flow between a fracture and its
ambient matrix is an *interface* quantity. It is not a property of any cell,
which is why it has nowhere to sit in a per-cell coefficient and why it needs
the third scope to exist at all.

### A package declares its slots

Just as a package declares the fields it contributes, it declares the slots it
requires and at which scope:

```cpp
Requirements Flow::requirements() const {
  return {
    .fields = {{"p", DofLayout::cell_wise(dim)}, {"q", DofLayout::cochain(dim, dim - 1)}},
    .provides = {"pressure", "mass_balance"},
    .slots = {
      {"density",             Scope::fluid},
      {"viscosity",           Scope::fluid},
      {"porosity",            Scope::rock},
      {"permeability",        Scope::rock},
      {"normal_permeability", Scope::interface},   // only if the mesh is stratified
    },
  };
}
```

The composition then knows, before anything is built, the complete list of
what must be supplied and where.

### Slots have their own registries

One registry per slot name, populated at static init exactly as the term
registry is:

```cpp
inline const RegisterClosure<KozenyCarman> kc{"porosity", "kozeny_carman"};
inline const RegisterClosure<CubicLaw>     cl{"permeability", "cubic_law",
                                              /*needs=*/{"fracture"}};
```

Adding a porosity law is one file and one registration. It never touches the
packages, the terms, or the catalogue. Closures may carry `needs` tags of their
own, so a cubic-law permeability on a bulk stratum is rejected by name rather
than producing silent nonsense.

### The Python surface

A builder rather than one call: the configuration is nested, and an
incremental object can report what is still missing:

```python
m = exokal.model("flow", mesh="fractured.vtu", discretization="mimetic")

m.fluid.density("slightly_compressible", rho0=1000.0, c=4.5e-10, p_ref=1e5)
m.fluid.viscosity("constant", mu=1e-3)

m.rock["matrix"].porosity("exponential", phi0=0.2, c=1e-9)
m.rock["matrix"].permeability("from_array", k=perm)          # (ncells,) or (ncells,3,3)
m.rock["fracture"].porosity("constant", phi0=1.0)
m.rock["fracture"].permeability("cubic_law", aperture=b)

m.interface("matrix", "fracture").normal_permeability("constant", kn=1e-12)

sim = m.build()          # validates every slot on every stratum and interface
```

Every name resolves to a compiled C++ closure; every value is data.

### Introspection is required

With this many choices the surface is unusable unless it can describe itself,
and all of it is already in the registries:

```python
exokal.describe("flow")
# fluid     : density, viscosity
# rock      : porosity, permeability          (per stratum)
# interface : normal_permeability             (per stratum pair)

exokal.choices("porosity")            # ['constant', 'exponential', 'kozeny_carman', 'poroelastic']
exokal.params("porosity", "kozeny_carman")   # names, units, defaults
exokal.inputs("porosity", "poroelastic")     # ['p', 'u'] - and so the (p,u) block appears
```

`m.build()` reports what is missing by name and scope — "stratum 'fracture'
has no porosity model" — rather than failing later inside an assembly.

## Solver configuration

The same pattern, a third registry: time integrator, nonlinear solver, linear
solver and preconditioner are named choices with their own parameters. It is
not a new mechanism and not part of the model composition — a model is the
equations and their closures; how it is solved is bound separately, so the
same model can be run monolithically or with a fixed-stress split without
being redeclared.

## What blocks this today

**Terms index their fields positionally.** `MixedDarcyCell` reads
`st.view.blocks[1]` for the pressure, which is correct only because the space
happens to list `q` then `p`. The moment `Mechanics` contributes `u` to the
same space, every index shifts and every term breaks.

Positional indexing is the mechanism by which the axes would start to
multiply — each combination would need its own field ordering, and therefore
its own term. **Field lookup by name is a prerequisite for this catalogue.**
A term must resolve its field names against the composed space once at
construction, and cache the block indices it found.

## Testing

The code is a sum; the tests must be the product. Every catalogue entry gets a
cheap smoke assembly on a small mesh, asserting the space has the expected
fields, the model resolves against them, and the Jacobian's block structure
matches what the packages declared. Fifteen of those are seconds.
