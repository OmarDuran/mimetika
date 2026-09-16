#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "exokal/forms/assemble.hpp"
#include "mimetika/linear_solver/linear.hpp"

// The approximate Schur complement of the mixed flow saddle point, assembled.
//
// The system the Darcy term writes is
//
//     [ M   -D^T ] [ q ]   [ f ]
//     [ D    0   ] [ p ] = [ g ],
//
// with M the flux Hodge, cell-local and dense a cell, and D the divergence --
// topological, one entry per facet a cell, +/-1 against the cell's outward
// orientation, pairing the CONSTANT moment alone. Eliminating q leaves the
// exact Schur complement
//
//     S = D M^-1 D^T,
//
// which is SPD and dense: M^-1 couples every facet of a cell to every other, so
// S reaches every cell sharing a facet-neighbour's facet. Replacing M by its
// diagonal gives
//
//     S~ = D diag(M)^-1 D^T,
//
// which is the same operator with the cell-local coupling dropped: one entry a
// facet, hence a WEIGHTED GRAPH LAPLACIAN on the cell adjacency graph, with the
// transmissibility of a facet the reciprocal of its own mass. That is the
// two-point flux matrix -- exact where the star is already diagonal, and the
// two-point approximation of its own Schur complement everywhere else.
//
// Pazner, Kolev & Vassilevski (SIAM J. Sci. Comput. 46 (2024) B179) build the
// same object for high-order H(div): their eq. (15) carries a W~^-1 term as
// well, which is the mass of the pressure space. Here the pressure is one
// constant a cell and the (2, 2) block is empty, so that term is absent and S~
// is the Laplacian alone. Their D is topological only in the
// interpolation-histopolation basis; at lowest order it is topological outright.
//
// S~ is an M-matrix: every off-diagonal is -1/m_f with m_f > 0, and the
// diagonal is the sum of the facet's own terms over the cell's facets, so the
// row sums vanish on the interior and the matrix is weakly diagonally dominant.
// An algebraic multigrid cycle preconditions it at once.

namespace mimetika::solver {

// Where an unknown sits in a block, or -1: the inverse of an ascending index
// list, sized to the system so a lookup is one load.
inline std::vector<int> slot_of(std::size_t n, const std::vector<Index>& block) {
  std::vector<int> at(n, -1);
  for (std::size_t i = 0; i < block.size(); ++i) {
    at[static_cast<std::size_t>(block[i])] = static_cast<int>(i);
  }
  return at;
}

// S~ = D diag(M)^-1 D^T, in the pressure numbering the block induces.
//
// One pass over A collects diag(M) and the rows of D; the product is then taken
// a FLUX dof at a time, because a flux dof is read by the two cells sharing its
// facet and by no others. Each contributes the outer product of a column with
// at most two entries, so the result carries at most one off-diagonal per facet
// of a cell and nothing has to be searched for.
inline SparseSystem approximate_schur(const SparseSystem& A, const std::vector<Index>& flux,
                                      const std::vector<Index>& pressure) {
  const std::vector<int> in_flux = slot_of(A.n, flux);
  const std::vector<int> in_pressure = slot_of(A.n, pressure);

  std::vector<double> mass(flux.size(), 0.0);
  // the column of D over a flux dof: (pressure row, value), two at most
  std::vector<std::vector<std::pair<int, double>>> column(flux.size());

  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto r = static_cast<std::size_t>(A.row[k]);
    const auto c = static_cast<std::size_t>(A.col[k]);
    const int fr = in_flux[r], fc = in_flux[c];
    if (fr >= 0 && fr == fc) {
      mass[static_cast<std::size_t>(fr)] += A.value[k];
    } else if (fc >= 0 && in_pressure[r] >= 0) {
      column[static_cast<std::size_t>(fc)].emplace_back(in_pressure[r], A.value[k]);
    }
  }

  for (std::size_t i = 0; i < mass.size(); ++i) {
    if (!(mass[i] > 0.0)) {
      throw std::invalid_argument(
          "approximate_schur: the flux mass has a non-positive diagonal, so diag(M)^-1 is not a "
          "norm and S~ is not an M-matrix; the star is invalid on this mesh");
    }
  }

  SparseSystem S;
  S.n = pressure.size();
  for (std::size_t f = 0; f < flux.size(); ++f) {
    const auto& col = column[f];
    if (col.empty()) continue;
    const double w = 1.0 / mass[f];
    for (const auto& [i, vi] : col) {
      for (const auto& [j, vj] : col) {
        S.row.push_back(static_cast<Index>(i));
        S.col.push_back(static_cast<Index>(j));
        S.value.push_back(w * vi * vj);
      }
    }
  }
  S.add_structural_diagonal();
  return S;
}

// The preconditioner's data, assembled WITHOUT the operator.
//
// S~ needs two things from the tangent: the diagonal of the flux block and the
// divergence. Both are a vanishing fraction of it -- one number a flux dof, one
// entry a facet a cell -- and forming the whole saddle point to read them off
// would cost the memory the matrix-free route exists to save. So this sink runs
// the same term kernels the assembly does and keeps only those two, which also
// means the divergence is the one the operator applies rather than a second
// derivation of it that could drift.
//
// Want::jacobian, because a tangent is what carries them; the cost is one
// assembly's arithmetic and none of its storage.
class SchurSink final : public exokal::forms::Sink {
 public:
  // `pinned` is the constraint mask over the global numbering, or empty. A
  // PINNED FLUX IS NOT AN ADJACENCY: the symmetric elimination zeroes that
  // column of D, so the facet carries no coupling between its two cells and
  // belongs in neither the operator's Schur complement nor this approximation
  // of it. Keeping it would precondition a sealed face as though it were open,
  // which on a mostly sealed domain is a Dirichlet Laplacian standing in for a
  // Neumann one: measured on a box sealed but for two faces, 12, 15, 24 and 31
  // outer iterations over a ladder of 3^3 to 8^3 cells against 33, 42, 65 and
  // 82 with the pinned facets left in. The mass is still collected -- the
  // pinned block of the operator is diag(s_d) = diag(M_dd), so 1/M_dd inverts
  // it exactly.
  SchurSink(std::vector<int> in_flux, std::vector<int> in_pressure, std::size_t n_flux,
            std::vector<char> pinned = {})
      : in_flux_(std::move(in_flux)),
        in_pressure_(std::move(in_pressure)),
        pinned_(std::move(pinned)),
        mass(n_flux, 0.0),
        column(n_flux) {}

  std::vector<double> mass;                                   // diag(M), a flux dof
  std::vector<std::vector<std::pair<int, double>>> column;    // D, a flux dof

  void scatter(const exokal::forms::Stencil& st,
               const exokal::ad::LocalSystem& sys) override {
    const auto& dofs = st.view.dofs;
    const exokal::ad::LocalSpace& sp = sys.space();
    for (std::size_t bi = 0; bi < sp.n_blocks(); ++bi) {
      for (std::size_t bj = 0; bj < sp.n_blocks(); ++bj) {
        if (!sys.has_block(bi, bj)) continue;
        const exokal::numerics::Dense& blk = sys.block(bi, bj);
        for (std::size_t i = 0; i < blk.rows(); ++i) {
          const auto r = static_cast<std::size_t>(dofs[sp.begin(bi) + i]);
          for (std::size_t j = 0; j < blk.cols(); ++j) {
            const double v = blk(i, j);
            if (v == 0.0) continue;
            const auto c = static_cast<std::size_t>(dofs[sp.begin(bj) + j]);
            if (r == c && in_flux_[r] >= 0) {
              mass[static_cast<std::size_t>(in_flux_[r])] += v;
            } else if (in_pressure_[r] >= 0 && in_flux_[c] >= 0 && !is_pinned(c)) {
              column[static_cast<std::size_t>(in_flux_[c])].emplace_back(in_pressure_[r], v);
            }
          }
        }
      }
    }
  }

 private:
  bool is_pinned(std::size_t d) const { return !pinned_.empty() && pinned_[d] != 0; }

  std::vector<int> in_flux_, in_pressure_;
  std::vector<char> pinned_;
};

// S~ from what the sink kept, identical to approximate_schur's product but with
// no assembled system to read it from.
inline SparseSystem schur_of(const SchurSink& sink, std::size_t n_pressure) {
  for (std::size_t i = 0; i < sink.mass.size(); ++i) {
    if (!(sink.mass[i] > 0.0)) {
      throw std::invalid_argument(
          "schur_of: the flux mass has a non-positive diagonal, so diag(M)^-1 is not a norm");
    }
  }
  SparseSystem S;
  S.n = n_pressure;
  for (std::size_t f = 0; f < sink.column.size(); ++f) {
    const auto& col = sink.column[f];
    if (col.empty()) continue;
    const double w = 1.0 / sink.mass[f];
    for (const auto& [i, vi] : col) {
      for (const auto& [j, vj] : col) {
        S.row.push_back(static_cast<Index>(i));
        S.col.push_back(static_cast<Index>(j));
        S.value.push_back(w * vi * vj);
      }
    }
  }
  S.add_structural_diagonal();
  return S;
}

}  // namespace mimetika::solver
