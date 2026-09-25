#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exokal/forms/model.hpp"
#include "exokal/spaces/dof_layout.hpp"
#include "exokal/spaces/dof_map.hpp"
#include "exokal/spaces/product_space.hpp"

// A physics package: one set of equations, with the fields it introduces,
// the closures it must be given, and the terms it attaches.
//
// This is the unit the model catalogue is built from. The catalogue reads as a
// product — flow times mechanics times domain type — over a sum of packages
// that each exist once: poromechanics is flow, plus mechanics, plus a coupling
// package that reads the pressure and the displacement without caring how many
// components produced either.
//
// A package declares which capabilities it provides and which it needs, so a
// composition missing a package is rejected by name — "PoroCoupling needs
// 'displacement', provided by no package" — instead of a term indexing past the
// end of a stencil at assembly time. Closure slots are declared the same way,
// so the configuration surface of a model can be reported before it is built.
//
// A package may not mention the domain type. Whether the mesh is a single
// stratum, a static stratification, or one that changes between epochs is a
// property of the mesh and the driver, never of the equations — exokal's
// stratified epoch already carries a term across every codimension it makes
// sense on.

namespace mimetika::physics {

using exokal::spaces::DofLayout;
using exokal::spaces::DofMap;
using exokal::spaces::ProductSpace;

// Where a closure is bound. There are exactly three.
enum class Scope {
  fluid,      // a property of the phase: density, viscosity, enthalpy
  rock,       // a property of a stratum: porosity, permeability
  interface,  // a property of a stratum pair: the normal permeability
              // governing exchange across a codimension gap
};

inline const char* name_of(Scope s) {
  switch (s) {
    case Scope::fluid: return "fluid";
    case Scope::rock: return "rock";
    case Scope::interface: return "interface";
  }
  return "?";
}

struct SlotSpec {
  std::string name;
  Scope scope{Scope::rock};
};

// A field is named and given a layout; the layout decides which cochain
// degree it is supported on, and therefore which pairings it can enter.
struct FieldSpec {
  std::string name;
  DofLayout layout;
};

struct Requirements {
  std::vector<FieldSpec> fields;
  std::vector<std::string> provides;  // capability tags this package supplies
  std::vector<std::string> needs;     // tags another package must supply
  std::vector<SlotSpec> slots;        // closures the caller must bind
};

class Package {
 public:
  virtual ~Package() = default;
  virtual std::string name() const = 0;

  // The layouts depend on the dimension of the stratum the package will run
  // on, and the field names on its codimension, which is why this is a query
  // rather than a constant. A package declares the quantity; the stratum
  // decides what it is called there.
  virtual Requirements requirements(int dim, int codim = 0) const = 0;

  // Attach this package's terms to a model. The context carries the data a
  // named term needs — the discrete Hodge, the closures — which the driver
  // owns and provides.
  virtual void attach(exokal::forms::Model& model, const exokal::forms::TermContext& ctx) const = 0;
};

// A package confined to one codimension.
//
// MOST PACKAGES ARE CODIMENSION-AGNOSTIC ON PURPOSE. Flow in a fracture is the
// same two terms one codimension down, so `Flow` declares its fields at
// whatever codimension it is asked about and exokal evaluates the same terms
// there; that is the design, and it is why a package may not mention the
// domain type.
//
// Contact is the case that design does not cover. The bulk equations have no
// counterpart inside the fracture -- there is no stress there, only a traction
// on its two faces -- so a bulk package asked about codimension one would
// declare fields that no term ever writes the rows of, and the system would be
// singular by construction. This says which stratum a package governs.
//
// TERMS ARE NOT FILTERED HERE. exokal runs a term only on the strata that host
// its fields, so withholding the fields already withholds the equations, and
// the two can never disagree.
class AtCodim final : public Package {
 public:
  AtCodim(int codim, std::unique_ptr<Package> inner) : codim_(codim), inner_(std::move(inner)) {
    if (inner_ == nullptr) throw std::invalid_argument("AtCodim: null package");
  }

  std::string name() const override {
    return inner_->name() + " (codim " + std::to_string(codim_) + ")";
  }

  Requirements requirements(int dim, int codim = 0) const override {
    return codim == codim_ ? inner_->requirements(dim, codim) : Requirements{};
  }

  void attach(exokal::forms::Model& m, const exokal::forms::TermContext& ctx) const override {
    inner_->attach(m, ctx);
  }

 private:
  int codim_{0};
  std::unique_ptr<Package> inner_;
};

// Several packages, validated together: it takes packages that know nothing of
// each other and checks that what one needs another provides, before a single
// degree of freedom is numbered.
class Composition {
 public:
  Composition& add(std::unique_ptr<Package> p) {
    if (p == nullptr) throw std::invalid_argument("Composition: null package");
    packages_.push_back(std::move(p));
    return *this;
  }

  template <class P, class... Args>
  Composition& emplace(Args&&... args) {
    return add(std::make_unique<P>(std::forward<Args>(args)...));
  }

  // Confine every package added so far to one codimension. Written as a
  // postfix so a composition a catalogue built can be placed on a stratum
  // without the catalogue knowing there is more than one:
  //
  //     Composition c = Catalogue::instance().build("linear_elasticity", o);
  //     c.only_at(0).emplace<LinearContact>(co);
  Composition& only_at(int codim) {
    for (auto& p : packages_) p = std::make_unique<AtCodim>(codim, std::move(p));
    return *this;
  }

  std::size_t size() const { return packages_.size(); }
  const Package& at(std::size_t i) const { return *packages_.at(i); }
  Requirements requirements_of(std::size_t i, int dim, int codim = 0) const {
    return packages_.at(i)->requirements(dim, codim);
  }

  // Every capability some package needs must be provided by some package.
  // Reported by name, and naming the package that asked, because this is
  // the message a user sees when a composition is wrong.
  void validate(int dim, int codim = 0) const {
    std::map<std::string, std::string> provided;  // tag -> providing package
    for (const auto& p : packages_) {
      for (const std::string& t : p->requirements(dim, codim).provides) {
        const auto [it, fresh] = provided.emplace(t, p->name());
        if (!fresh) {
          throw std::invalid_argument("Composition: '" + t + "' is provided by both " + it->second +
                                      " and " + p->name());
        }
      }
    }
    for (const auto& p : packages_) {
      for (const std::string& t : p->requirements(dim, codim).needs) {
        if (provided.count(t) == 0) {
          throw std::invalid_argument("Composition: " + p->name() + " needs '" + t +
                                      "', provided by no package");
        }
      }
    }
  }

  // The product space every package contributes to. Field order is the order
  // packages were added, and no term depends on it: exokal resolves a term's
  // fields by name against whatever space it is given.
  ProductSpace space(const graphos::Complex& c, int cell_dim = -1, int codim = 0) const {
    const int dim = cell_dim < 0 ? c.dim() : cell_dim;
    validate(dim, codim);
    ProductSpace s;
    for (const auto& p : packages_) {
      for (const FieldSpec& f : p->requirements(dim, codim).fields) {
        if (s.has(f.name)) {
          throw std::invalid_argument("Composition: " + p->name() + " declares field '" + f.name +
                                      "', which another package already declared");
        }
        s.add(f.name, DofMap(c, f.layout, dim));
      }
    }
    return s;
  }

  // The complete configuration surface: every closure every package needs,
  // with the scope it is bound at. This is what a driver reports so a user
  // can see what must be supplied without reading the source.
  std::vector<SlotSpec> slots(int dim, int codim = 0) const {
    std::vector<SlotSpec> out;
    for (const auto& p : packages_) {
      for (const SlotSpec& s : p->requirements(dim, codim).slots) {
        bool seen = false;
        for (const SlotSpec& e : out) seen = seen || (e.name == s.name && e.scope == s.scope);
        if (!seen) out.push_back(s);
      }
    }
    return out;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext& ctx) const {
    for (const auto& p : packages_) p->attach(model, ctx);
  }

 private:
  std::vector<std::unique_ptr<Package>> packages_;
};

}  // namespace mimetika::physics
