#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mimetika/model/boundary.hpp"
#include "mimetika/model/constraints.hpp"

// Boundary conditions, one physics at a time.
//
// Poromechanics is two physics coupled, and their boundary conditions are
// separate descriptions. A face of a domain may be traction-loaded and sealed,
// or roller-supported and drained, or loaded and drained -- the mechanical and
// the hydraulic sets do not have to coincide and in a real problem they do not.
// Holding one list of conditions with a tag saying which physics each belongs
// to makes that accidental; holding two makes it structural.
//
//     MechanicsBoundary   acts on the stress field       s
//     FlowBoundary        acts on the flux and pressure  q, p
//
// A condition is three things, and a consumer needs all three:
//
//     the parameters   the numbers it carries -- a stress tensor, a datum
//     the form         the linear functional it imposes on the unknowns
//     the dofs         which unknowns it reaches, and the facets they came from
//
// Keeping the dofs is what makes a condition an object rather than a side
// effect. A driver reads the traction back off the wall it prescribed it on; a
// transient datum is updated in place without re-deriving the numbering; a
// diagnostic asks which unknowns a condition actually touched.

namespace mimetika {

// One facet's worth of a condition: the functional, and where it came from.
struct FacetForm {
  Index facet{0};
  std::vector<Index> dofs;
  std::vector<double> coeff;
  double value{0.0};
};

// The wrench layout, recognized from the space rather than told: the
// d(d+1)/2 rigid-motion moments of one scalar layout -- six in space, three
// in the plane -- is the signature no componentwise stress field has, those
// carrying d components. The strong family always, diagonal_afw on the weak
// axis; a condition serves all of them through the one space it is resolved
// against. In the plane the wrench is {t, n chi_0, n chi_1}: no rotation slot.
inline bool strong_layout(const FacetDofs& d) {
  return d.components == 1 && (d.moments == 6 || d.moments == 3);
}

// the tangent the wrench basis uses in the plane: the quarter turn of the
// canonical normal, as exokal's edge chart defines it
inline Point wrench_plane_tangent(const FacetFrame& fr) {
  return Point{-fr.normal[1], fr.normal[0], 0.0};
}

// The base every condition shares. `resolve` is what turns parameters plus a
// facet set into forms, and it needs the space because a form is a statement
// about unknowns.
class BoundaryCondition {
 public:
  virtual ~BoundaryCondition() = default;

  virtual std::string name() const = 0;
  // Strong conditions replace equations; natural ones are data a term reads.
  // Which is which follows from the mixed form and not from the caller: the
  // quantity carried as an unknown is imposed strongly.
  virtual bool strong() const = 0;
  virtual void resolve(const exokal::Mesh& mesh, int cell_dim,
                       const exokal::spaces::ProductSpace& space, Index offset) = 0;

  const std::vector<Index>& facets() const { return facets_; }
  const std::vector<FacetForm>& forms() const { return forms_; }

  // every unknown this condition reaches, once each
  std::vector<Index> dofs() const {
    std::vector<Index> out;
    for (const FacetForm& f : forms_) {
      for (const Index d : f.dofs) {
        bool seen = false;
        for (const Index e : out) seen = seen || e == d;
        if (!seen) out.push_back(d);
      }
    }
    return out;
  }

  // hand the forms to the constraint set, for the strong ones
  void impose(Constraints& c) const {
    if (!strong()) return;
    for (const FacetForm& f : forms_) c.constrain(f.dofs, f.coeff, f.value);
  }

 protected:
  std::vector<Index> facets_;
  std::vector<FacetForm> forms_;
};

// ------------------------------------------------------------- mechanics

// sigma n = g, from a stress tensor rather than a traction vector: the caller
// never has to know which way a facet's canonical normal points, and a vector
// assembled against the wrong one is silently sign-flipped.
class TractionBC final : public BoundaryCondition {
 public:
  TractionBC(std::vector<Index> facets, std::array<double, 9> stress) : stress_(stress) {
    facets_ = std::move(facets);
  }
  std::string name() const override { return "traction"; }
  bool strong() const override { return true; }

  const std::array<double, 9>& stress() const { return stress_; }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset) override {
    forms_.clear();
    // the coboundary is built once for the batch: per facet it is O(mesh),
    // and a resolve over the boundary of a large mesh then costs more than
    // the assembly it serves
    const std::vector<Index> cells = cofacets_of(mesh, cell_dim, facets_);
    for (std::size_t fi = 0; fi < facets_.size(); ++fi) {
      const Index f = facets_[fi];
      const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cells[fi], f);
      const FacetDofs d = facet_dofs(space, field_, cell_dim, f, offset);
      // the traction against the canonical normal, so the dof convention and
      // the datum agree on orientation without the caller knowing either
      std::array<double, 3> t{};
      for (int k = 0; k < cell_dim; ++k) {
        for (int j = 0; j < cell_dim; ++j) {
          t[static_cast<std::size_t>(k)] +=
              stress_[static_cast<std::size_t>(k * 3 + j)] * fr.normal[static_cast<std::size_t>(j)];
        }
      }
      if (strong_layout(d)) {
        // The wrench slots, in the facet's canonical frame: a uniform traction
        // meets only the resultant slots -- the rotation moment is centred and
        // the chart's higher functions have zero mean. In space slots 0, 1 and
        // 3 carry |f| times its frame components and 2, 4, 5 are zero; in the
        // plane slots 0 and 1 carry them and 2 is zero. The frame here is the
        // one the discrete basis uses.
        std::array<double, 6> value{};
        if (d.moments == 6) {
          value = {fr.measure * dot(t, fr.tangent[0]), fr.measure * dot(t, fr.tangent[1]), 0.0,
                   fr.measure * dot(t, fr.normal),     0.0,                                0.0};
        } else {
          value = {fr.measure * dot(t, wrench_plane_tangent(fr)), fr.measure * dot(t, fr.normal),
                   0.0, 0.0, 0.0, 0.0};
        }
        for (int b = 0; b < d.moments; ++b) {
          forms_.push_back(FacetForm{f, {d.at(0, b)}, {1.0}, value[static_cast<std::size_t>(b)]});
        }
        continue;
      }
      for (int k = 0; k < cell_dim; ++k) {
        // a uniform traction lands entirely on the constant moment, scaled by
        // the measure it is integrated against; the higher moments are zero
        for (int b = 0; b < d.moments; ++b) {
          forms_.push_back(FacetForm{
              f, {d.at(k, b)}, {1.0}, b == 0 ? t[static_cast<std::size_t>(k)] * fr.measure : 0.0});
        }
      }
    }
  }

 private:
  static double dot(const std::array<double, 3>& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  std::array<double, 9> stress_;
  std::string field_{"s_0"};
};

// t_a . (sigma n) = 0 on every tangent: free slip, the strong half of a roller.
// The vanishing normal displacement is the natural half and is imposed by
// leaving the normal traction free, so that its own equation reads u.n = 0.
class FreeSlipBC final : public BoundaryCondition {
 public:
  explicit FreeSlipBC(std::vector<Index> facets) { facets_ = std::move(facets); }
  std::string name() const override { return "free slip"; }
  bool strong() const override { return true; }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset) override {
    forms_.clear();
    const std::vector<Index> cells = cofacets_of(mesh, cell_dim, facets_);
    for (std::size_t fi = 0; fi < facets_.size(); ++fi) {
      const Index f = facets_[fi];
      const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cells[fi], f);
      const FacetDofs d = facet_dofs(space, field_, cell_dim, f, offset);
      if (strong_layout(d)) {
        // the tangential traction is carried whole by the tangential slots --
        // in space 0, 1 and 2, the two mean components and the in-plane
        // rotation moment; in the plane slot 0 alone -- so free slip pins
        // exactly those and leaves the normal slots free
        const int n_tangential = d.moments == 6 ? 3 : 1;
        for (int b = 0; b < n_tangential; ++b) {
          forms_.push_back(FacetForm{f, {d.at(0, b)}, {1.0}, 0.0});
        }
        continue;
      }
      for (int a = 0; a < fr.n_tangents; ++a) {
        const Point& t = fr.tangent[static_cast<std::size_t>(a)];
        for (int b = 0; b < d.moments; ++b) {
          std::vector<Index> dofs;
          std::vector<double> coeff;
          for (int k = 0; k < d.components; ++k) {
            if (t[static_cast<std::size_t>(k)] == 0.0) continue;
            dofs.push_back(d.at(k, b));
            coeff.push_back(t[static_cast<std::size_t>(k)]);
          }
          if (!dofs.empty()) forms_.push_back(FacetForm{f, std::move(dofs), std::move(coeff), 0.0});
        }
      }
    }
  }

 private:
  std::string field_{"s_0"};
};

// ------------------------------------------------------------------ flow

// q . n = g on a scalar facet field. Zero is a sealed facet.
class NormalFluxBC final : public BoundaryCondition {
 public:
  NormalFluxBC(std::vector<Index> facets, double value = 0.0) : value_(value) {
    facets_ = std::move(facets);
  }
  std::string name() const override { return "normal flux"; }
  bool strong() const override { return true; }
  double value() const { return value_; }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset) override {
    forms_.clear();
    const std::vector<Index> cells = cofacets_of(mesh, cell_dim, facets_);
    for (std::size_t fi = 0; fi < facets_.size(); ++fi) {
      const Index f = facets_[fi];
      const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cells[fi], f);
      const FacetDofs d = facet_dofs(space, field_, cell_dim, f, offset);
      for (int b = 0; b < d.moments; ++b) {
        for (int k = 0; k < d.components; ++k) {
          forms_.push_back(FacetForm{f, {d.at(k, b)}, {1.0}, b == 0 ? value_ * fr.measure : 0.0});
        }
      }
    }
  }

 private:
  double value_;
  std::string field_{"q_0"};
};

// p = g on the facet: natural in the mixed form, so it is data a term reads
// rather than an equation replaced. It still resolves its dofs -- the flux
// moments the datum will reach -- because a caller asking "which unknowns does
// this condition touch" deserves the same answer whichever kind it is.
class PressureBC final : public BoundaryCondition {
 public:
  PressureBC(std::vector<Index> facets, double value) : value_(value) {
    facets_ = std::move(facets);
  }
  std::string name() const override { return "pressure"; }
  bool strong() const override { return false; }
  double value() const { return value_; }

  // The datum as it enters the row is the value itself.
  //
  // It goes into the flux row, against -B^T p, whose entries are the incidence
  // and not the measure -- the flux unknown is already the measure-weighted
  // moment int_f (q.n) chi, so the pairing carries no further factor. Scaling
  // the datum by the measure imposes the condition at a strength that varies
  // facet by facet on a graded mesh, an error a steady Darcy annulus exposes
  // and a uniform column does not.
  void fill(BoundaryData& data, const exokal::Mesh& mesh, int cell_dim) const {
    (void)mesh;
    (void)cell_dim;
    for (const Index f : facets_) data.set({f}, value_);
  }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset) override {
    forms_.clear();
    for (const Index f : facets_) {
      const FacetDofs d = facet_dofs(space, field_, cell_dim, f, offset);
      // the datum reaches the constant moment of the flux on this facet
      forms_.push_back(FacetForm{f, {d.at(0, 0)}, {1.0}, value_});
    }
    (void)mesh;
  }

 private:
  double value_;
  std::string field_{"q_0"};
};

// a (q.n) + b p_E = g: a Robin condition, one form over two fields. This is
// what forces the constraint layer to take forms rather than values: no
// per-dof pinning expresses a condition that couples a facet unknown to a cell
// unknown.
class RobinBC final : public BoundaryCondition {
 public:
  RobinBC(std::vector<Index> facets, double a, double b, double g) : a_(a), b_(b), g_(g) {
    facets_ = std::move(facets);
  }
  std::string name() const override { return "robin"; }
  bool strong() const override { return true; }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset) override {
    forms_.clear();
    const std::size_t ci = space.index_of(cell_field_);
    const exokal::spaces::DofMap& cmap = space.map(ci);
    const std::vector<Index> cells = cofacets_of(mesh, cell_dim, facets_);
    for (std::size_t fi = 0; fi < facets_.size(); ++fi) {
      const Index f = facets_[fi];
      const Index cell = cells[fi];
      const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cell, f);
      const FacetDofs d = facet_dofs(space, field_, cell_dim, f, offset);
      const Index p = space.offset(ci) + cmap.global(cell_dim, cell, 0, 0) + offset;
      for (int b = 0; b < d.moments; ++b) {
        for (int k = 0; k < d.components; ++k) {
          if (b == 0) {
            forms_.push_back(FacetForm{f, {d.at(k, b), p}, {a_, b_}, g_ * fr.measure});
          } else {
            forms_.push_back(FacetForm{f, {d.at(k, b)}, {a_}, 0.0});
          }
        }
      }
    }
  }

 private:
  double a_, b_, g_;
  std::string field_{"q_0"}, cell_field_{"p_0"};
};

// ------------------------------------------------------- one set per physics

class BoundarySet {
 public:
  template <class C, class... Args>
  C& emplace(Args&&... args) {
    auto c = std::make_unique<C>(std::forward<Args>(args)...);
    C& ref = *c;
    conditions_.push_back(std::move(c));
    return ref;
  }

  std::size_t size() const { return conditions_.size(); }
  const BoundaryCondition& at(std::size_t i) const { return *conditions_[i]; }

  void resolve(const exokal::Mesh& mesh, int cell_dim, const exokal::spaces::ProductSpace& space,
               Index offset = 0) {
    for (auto& c : conditions_) c->resolve(mesh, cell_dim, space, offset);
  }

  void impose(Constraints& c) const {
    for (const auto& bc : conditions_) bc->impose(c);
  }

  // the natural pressure data of this set, if any
  bool fill_pressure(BoundaryData& data, const exokal::Mesh& mesh, int cell_dim) const {
    bool any = false;
    for (const auto& bc : conditions_) {
      if (const auto* p = dynamic_cast<const PressureBC*>(bc.get())) {
        p->fill(data, mesh, cell_dim);
        any = true;
      }
    }
    return any;
  }

 private:
  std::vector<std::unique_ptr<BoundaryCondition>> conditions_;
};

// Named for what they are, so a driver reads as the problem does.
using MechanicsBoundary = BoundarySet;
using FlowBoundary = BoundarySet;

}  // namespace mimetika
