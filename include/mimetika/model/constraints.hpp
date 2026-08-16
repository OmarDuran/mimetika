#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/forms/model.hpp"

// ESSENTIAL CONSTRAINTS, as LINEAR FORMS on the degrees of freedom.
//
//     sum_j a_j x_{d_j} = g
//
// A single pinned unknown is the one-term case, and it is not the general one.
// In a mixed method a boundary condition is a statement about a QUANTITY, and
// the quantity is a form on the facet's unknowns rather than one of them:
//
//     n . (sigma n) = g      the normal traction   sum_k n_k x_{f,k,b}
//     t . (sigma n) = 0      no shear              sum_k t_k x_{f,k,b}
//     q . n = 0              a sealed facet        x_{f,b}
//     a (q.n) + b p = c      a Robin condition, coupling a facet to a cell
//
// Only on an axis-aligned facet does the first collapse to pinning a single
// component. On a borehole wall it does not, and a code that can only pin
// components either refuses the mesh or silently imposes a different
// condition. Carrying the form is what makes the same statement mean the same
// thing on any mesh in any dimension.
//
// SUBSTITUTION, NOT PENALTY. The form replaces the equation of ONE of the
// unknowns it involves -- its LEADING dof, chosen by largest coefficient, as a
// pivot is chosen. The residual becomes the discrepancy of the form, the
// tangent's row becomes the form itself, and the action of the tangent along a
// direction is the form applied to the direction. The constraint is then EXACT:
// a penalty would leave an error scaling with the penalty parameter, and a
// benchmark checked against a closed form would be measuring the penalty
// rather than the discretization.
//
// The columns of the constrained unknowns are left alone. That makes the
// system unsymmetric, which is deliberate: eliminating them means moving their
// contribution into the right-hand side, and in a matrix-free setting that is
// an extra operator apply per assembly.
//
// THE ROW CARRIES THE SCALE OF THE EQUATION IT REPLACES.
//
// The form and any nonzero multiple of it are the same constraint, so the
// scale is free and the solution does not depend on it. It is not free for the
// FACTORIZATION. The replaced row is a constitutive relation carrying that
// relation's factors -- a traction moment has A_ii ~ 1/(2 mu), around 1e-9 for
// rock; a flux moment has A_ii ~ 1/(k dt), which can be 1e14 for a small step.
// A unit-scaled row sits many orders of magnitude away from everything around
// it: the system is exactly as well posed, but a direct factorization pivots
// on the wrong entries and reports a zero pivot somewhere else entirely (MUMPS
// returns DIVERGED_PC_FAILED on a problem with a perfectly good solution).
//
// It must be the same scale on all three paths, and then
//
//     s a^T dx = -s (a^T x - g)   =>   a^T (x + dx) = g
//
// exactly, whatever s is, and a Krylov method sees one consistent operator.

namespace mimetika {

using exokal::forms::Index;

class Constraints {
 public:
  // A LINEAR FORM over the global numbering, and the value it takes.
  struct Form {
    std::vector<Index> dofs;
    std::vector<double> coeff;
    double value{0.0};
  };

  // The general case.
  void constrain(std::vector<Index> dofs, std::vector<double> coeff, double value) {
    if (dofs.empty() || dofs.size() != coeff.size()) {
      throw std::invalid_argument(
          "Constraints::constrain: a form needs matching dofs and "
          "coefficients, and at least one term");
    }
    forms_.push_back(Form{std::move(dofs), std::move(coeff), value});
    final_ = false;
  }

  // The one-term case, which is what a sealed facet or a clamped moment is.
  void pin(Index dof, double value) { constrain({dof}, {1.0}, value); }

  void pin_all(const std::vector<Index>& dofs, double value) {
    for (const Index d : dofs) pin(d, value);
  }

  std::size_t size() const { return forms_.size(); }
  bool empty() const { return forms_.empty(); }
  const std::vector<Form>& forms() const { return forms_; }

  // MOVE A FORM'S VALUE, leaving its structure alone.
  //
  // This is the affine decomposition made operational. finalize() assigns each
  // form the equation it replaces by partial pivoting on the COEFFICIENTS, and
  // the assembled operator depends on the dofs and the coefficients alone; the
  // value enters only the right-hand side. So a caller that solves the same
  // problem repeatedly with a different datum -- an outer iteration on a
  // prescribed traction, which is what contact is -- moves the value here and
  // keeps the factorization. Changing dofs or coefficients would need
  // finalize() again, and constrain() marks that by clearing the final flag;
  // this deliberately does not.
  void set_value(std::size_t form, double value) {
    if (form >= forms_.size()) throw std::out_of_range("Constraints::set_value: form index");
    forms_[form].value = value;
  }

  // ASSIGN EACH FORM THE EQUATION IT REPLACES, and build the lookups an
  // assembly needs. The leading dof is the unclaimed one with the largest
  // coefficient -- partial pivoting, and for the orthonormal facet frames the
  // boundary forms are written in it always succeeds. Two forms that want the
  // same single unknown are a genuine conflict and are refused; the ambiguous
  // middle, where a form is left with nothing to lead, is refused too rather
  // than silently dropped.
  void finalize(std::size_t n_dofs) {
    mask_.assign(n_dofs, 0);
    leader_of_.assign(n_dofs, -1);
    scale_.clear();
    scaled_ = false;

    // strongest form first, so a form with only one usable unknown is not
    // starved by one that had a choice
    std::vector<std::size_t> order(forms_.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return peak(forms_[a]) > peak(forms_[b]);
    });

    for (const std::size_t fi : order) {
      const Form& f = forms_[fi];
      std::size_t best = f.dofs.size();
      double best_c = 0.0;
      for (std::size_t j = 0; j < f.dofs.size(); ++j) {
        const auto d = static_cast<std::size_t>(f.dofs[j]);
        if (d >= n_dofs) throw std::out_of_range("Constraints: dof outside the numbering");
        if (mask_[d] != 0) continue;  // already leads another form
        if (std::abs(f.coeff[j]) > best_c) {
          best_c = std::abs(f.coeff[j]);
          best = j;
        }
      }
      if (best == f.dofs.size() || best_c <= 0.0) {
        // THE SAME CONDITION TWICE IS REDUNDANT, NOT A CONFLICT. Two facet sets
        // that overlap, or a driver that names a facet in two selections, both
        // produce a form already imposed; dropping it is right and refusing it
        // would make selections order-dependent. A form asking for the same
        // unknowns with a DIFFERENT equation is a genuine contradiction and is
        // still refused.
        bool redundant = false;
        for (const Index dj : f.dofs) {
          const auto d = static_cast<std::size_t>(dj);
          if (mask_[d] == 0 || leader_of_[d] < 0) continue;
          if (same_equation(forms_[static_cast<std::size_t>(leader_of_[d])], f)) {
            redundant = true;
            break;
          }
        }
        if (redundant) continue;
        throw std::invalid_argument(
            "Constraints: a form has no unknown left to impose it on; two conditions "
            "constrain the same degrees of freedom with different equations");
      }
      const auto d = static_cast<std::size_t>(f.dofs[best]);
      mask_[d] = 1;
      leader_of_[d] = static_cast<long long>(fi);
      lead_index_.resize(forms_.size(), 0);
      lead_index_[fi] = best;
    }
    final_ = true;
  }

  bool pinned(std::size_t d) const { return d < mask_.size() && mask_[d] != 0; }
  const std::vector<char>& mask() const { return mask_; }

  // The form led by a constrained dof, and which of its terms that dof is.
  const Form& form_at(std::size_t d) const { return forms_[index_at(d)]; }
  std::size_t lead_term_at(std::size_t d) const { return lead_index_[index_at(d)]; }

  // THE RIGHT-HAND SIDE of the form a dof leads: what a driver assembling
  // A x = b directly must place in that row, since the row itself is s a^T.
  double rhs_at(std::size_t d) const { return form_at(d).value; }

  // A one-term form's value, which is the solution's value at that dof.
  // Refused on a genuine form, where "the value of dof d" is not a thing that
  // exists -- only the form has a value.
  double value_at(std::size_t d) const {
    const Form& f = form_at(d);
    if (f.dofs.size() != 1) {
      throw std::invalid_argument(
          "Constraints::value_at: dof " + std::to_string(d) +
          " leads a multi-term form; use form_at() and impose the form itself");
    }
    return f.value / f.coeff[0];
  }

  // ---- the scale of each replaced equation ------------------------------

  double scale_at(std::size_t d) const { return scale_.empty() ? 1.0 : scale_[d]; }
  bool scaled() const { return scaled_; }

  // MEASURED LAZILY, AND SO const. The scale is not part of what the constraint
  // MEANS -- the form holds whatever it is multiplied by -- it is a
  // representation chosen to keep the replaced row in scale with the matrix
  // around it. It can therefore be read off the first tangent that gets
  // assembled rather than paid for with an assembly of its own.
  //
  // Rows the terms left empty have no scale of their own and fall back to the
  // mean of the rest, so an unknown no equation reached does not become the
  // pivot the factorization trips on.
  void set_scales(const std::vector<double>& diagonal) const {
    require_final();
    if (diagonal.size() != mask_.size()) {
      throw std::invalid_argument("Constraints::set_scales: size");
    }
    double sum = 0.0;
    std::size_t count = 0;
    for (const double v : diagonal) {
      if (std::abs(v) > 0.0) {
        sum += std::abs(v);
        ++count;
      }
    }
    const double reference = count > 0 ? sum / static_cast<double>(count) : 1.0;
    scale_.assign(mask_.size(), 1.0);
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] == 0) continue;
      scale_[d] = std::abs(diagonal[d]) > 0.0 ? std::abs(diagonal[d]) : reference;
    }
    scaled_ = true;
  }

  // ---- the three paths --------------------------------------------------

  // A STARTING POINT consistent with the forms: solve each for its leading
  // unknown, holding the others. It is only a starting point -- the constraint
  // is made exact by the step, not by this -- but it costs nothing and it
  // makes the very first residual small.
  void apply_to_state(std::vector<double>& x) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] == 0) continue;
      const Form& f = form_at(d);
      const std::size_t lead = lead_term_at(d);
      double rest = 0.0;
      for (std::size_t j = 0; j < f.dofs.size(); ++j) {
        if (j != lead) rest += f.coeff[j] * x[static_cast<std::size_t>(f.dofs[j])];
      }
      x[d] = (f.value - rest) / f.coeff[lead];
    }
  }

  // r_lead <- s (a^T x - g)
  void apply_to_residual(const std::vector<double>& x, std::vector<double>& r) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] == 0) continue;
      r[d] = scale_at(d) * (evaluate(form_at(d), x) - form_at(d).value);
    }
  }

  // y_lead <- s a^T v: the form applied to the direction, which is what the
  // tangent's row does to it
  void apply_to_action(const std::vector<double>& v, std::vector<double>& y) const {
    require_final();
    for (std::size_t d = 0; d < mask_.size(); ++d) {
      if (mask_[d] == 0) continue;
      y[d] = scale_at(d) * evaluate(form_at(d), v);
    }
  }

  static double evaluate(const Form& f, const std::vector<double>& x) {
    double acc = 0.0;
    for (std::size_t j = 0; j < f.dofs.size(); ++j) {
      acc += f.coeff[j] * x[static_cast<std::size_t>(f.dofs[j])];
    }
    return acc;
  }

 private:
  // The same equation up to a nonzero multiple: a x = g and (c a) x = (c g)
  // constrain identically, so they are compared after normalizing by the peak
  // coefficient rather than entry by entry.
  static bool same_equation(const Form& a, const Form& b) {
    if (a.dofs.size() != b.dofs.size()) return false;
    const double pa = peak(a), pb = peak(b);
    if (!(pa > 0.0) || !(pb > 0.0)) return false;
    double sign = 0.0;
    for (std::size_t j = 0; j < a.dofs.size(); ++j) {
      if (a.dofs[j] != b.dofs[j]) return false;
      const double ca = a.coeff[j] / pa, cb = b.coeff[j] / pb;
      if (sign == 0.0 && std::abs(ca) > 1e-12) sign = (ca * cb >= 0.0) ? 1.0 : -1.0;
      if (std::abs(ca - (sign == 0.0 ? 1.0 : sign) * cb) > 1e-12) return false;
    }
    const double s = sign == 0.0 ? 1.0 : sign;
    return std::abs(a.value / pa - s * b.value / pb) <= 1e-12 * (1.0 + std::abs(a.value / pa));
  }

  static double peak(const Form& f) {
    double m = 0.0;
    for (const double c : f.coeff) m = std::max(m, std::abs(c));
    return m;
  }

  std::size_t index_at(std::size_t d) const {
    require_final();
    if (d >= leader_of_.size() || leader_of_[d] < 0) {
      throw std::invalid_argument("Constraints: dof " + std::to_string(d) +
                                  " does not lead a form");
    }
    return static_cast<std::size_t>(leader_of_[d]);
  }

  void require_final() const {
    if (!final_) throw std::logic_error("Constraints: finalize() before use");
  }

  std::vector<Form> forms_;
  std::vector<char> mask_;
  std::vector<long long> leader_of_;     // dof -> form it leads, or -1
  std::vector<std::size_t> lead_index_;  // form -> which of its terms leads
  mutable std::vector<double> scale_;
  mutable bool scaled_{false};
  bool final_{false};
};

}  // namespace mimetika
