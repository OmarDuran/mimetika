#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/linear_solver/linear.hpp"

// STATIC CONDENSATION OF A DIAGONAL BLOCK.
//
// A diagonal star -- exokal's diagonal_tpfa and diagonal_tpsa -- makes the
// first block of the saddle point diagonal, and a diagonal block is inverted by
// division. So the flux or the stress leaves the system entirely:
//
//     [ M  A01 ] [ x0 ]   [ b0 ]            S = C - A10 M^-1 A01
//     [ A10 C  ] [ y  ] = [ b1 ]            s = b1 - A10 M^-1 b0
//
//     x0 = M^-1 (b0 - A01 y)                once y is known
//
// and what is handed to a solver is S, which for those two products is the
// finite volume method itself: the pressure alone for TPFA, seven entries a row
// in space; the displacement, rotation and total pressure for TPSA, which is
// Nordbotten & Keilegavlen's Eq. (3.9). Both are symmetric, both have the
// two-point stencil, and S is smaller than M alone -- 27 unknowns against 135
// on a 3^3 mesh of hexahedra, 189 against 513.
//
// WHY IT IS WORTH DOING BEYOND THE SIZE. The saddle point is indefinite and its
// first block is what a Riesz map spends its effort on; S has no such block.
// And the elimination is rank-revealing where a factorization is not: handed
// the Kuhn tetrahedra, on which diagonal_tpsa carries one spurious rotation per
// interior cube face, MUMPS reports CONVERGED and returns 1e16, while the
// condensation runs out of pivots and says so.
//
// THE COST IS ONE PASS. Each eliminated unknown contributes the outer product
// of its own row with its own column, and that row has the entries of the two
// cells sharing its facet -- fourteen of them for TPSA in space -- so the whole
// term is a few hundred multiplications per facet and no fill anywhere else.

namespace mimetika::solver {

using graphos::Index;

// The eliminated unknowns, the system that is left, and what is needed to put
// the eliminated field back.
struct Condensation {
  std::vector<int> first;  // eliminated, ascending
  std::vector<int> rest;   // kept, ascending
  std::vector<int> slot;   // global index -> position in `rest`, or -1
  std::vector<double> inv;  // 1 / M_gg, one per eliminated unknown

  SparseSystem reduced;
  std::vector<double> rhs;

  // the rows A(g, rest) of the eliminated unknowns, in `rest` positions
  std::vector<Index> row_begin;
  std::vector<int> row_col;
  std::vector<double> row_val;

  std::size_t size() const { return rest.size(); }

  // THE ELIMINATED FIELD, BACK OUT OF ITS OWN ROWS. One division per unknown:
  // no solve, and no communication, because the row of a facet unknown reaches
  // only the two cells that share it.
  std::vector<double> expand(const std::vector<double>& y, const std::vector<double>& b) const {
    if (y.size() != rest.size()) {
      throw std::invalid_argument("Condensation::expand: solution size");
    }
    std::vector<double> x(b.size(), 0.0);
    for (std::size_t i = 0; i < rest.size(); ++i) {
      x[static_cast<std::size_t>(rest[i])] = y[i];
    }
    for (std::size_t k = 0; k < first.size(); ++k) {
      double acc = b[static_cast<std::size_t>(first[k])];
      for (Index e = row_begin[k]; e < row_begin[k + 1]; ++e) {
        acc -= row_val[static_cast<std::size_t>(e)] * y[static_cast<std::size_t>(
                   row_col[static_cast<std::size_t>(e)])];
      }
      x[static_cast<std::size_t>(first[k])] = acc * inv[k];
    }
    return x;
  }
};

// IS THE BLOCK DIAGONAL? The question the specialization turns on, and it is
// asked of the assembled matrix rather than of the product's name: a star that
// stops being diagonal, for any reason, must stop being condensed here.
inline bool block_is_diagonal(const SparseSystem& A, const std::vector<int>& first) {
  std::vector<char> mine(A.n, 0);
  for (const int i : first) mine[static_cast<std::size_t>(i)] = 1;
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    if (A.row[k] == A.col[k] || A.value[k] == 0.0) continue;
    if (mine[static_cast<std::size_t>(A.row[k])] != 0 &&
        mine[static_cast<std::size_t>(A.col[k])] != 0) {
      return false;
    }
  }
  return true;
}

// Build S and s. Throws rather than returning a wrong system: a caller asking
// for the condensation of a block that is not diagonal, or that has a zero on
// its diagonal, has named the wrong field.
inline Condensation condense(const SparseSystem& A, const std::vector<double>& b,
                             std::vector<int> first) {
  Condensation c;
  c.first = std::move(first);
  c.slot.assign(A.n, -1);
  std::vector<char> mine(A.n, 0);
  for (const int i : c.first) {
    if (i < 0 || static_cast<std::size_t>(i) >= A.n) {
      throw std::invalid_argument("condense: eliminated index out of range");
    }
    mine[static_cast<std::size_t>(i)] = 1;
  }
  for (std::size_t i = 0; i < A.n; ++i) {
    if (mine[i] == 0) {
      c.slot[i] = static_cast<int>(c.rest.size());
      c.rest.push_back(static_cast<int>(i));
    }
  }

  // where each eliminated unknown sits in `first`, and its diagonal
  std::vector<int> place(A.n, -1);
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    place[static_cast<std::size_t>(c.first[k])] = static_cast<int>(k);
  }
  std::vector<double> diag(c.first.size(), 0.0);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto r = static_cast<std::size_t>(A.row[k]);
    const auto col = static_cast<std::size_t>(A.col[k]);
    if (mine[r] == 0) continue;
    if (r == col) {
      diag[static_cast<std::size_t>(place[r])] += A.value[k];
    } else if (mine[col] != 0 && A.value[k] != 0.0) {
      throw std::invalid_argument("condense: the block named is not diagonal");
    }
  }
  c.inv.resize(c.first.size());
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    if (diag[k] == 0.0) {
      throw std::invalid_argument("condense: a zero on the diagonal of the eliminated block");
    }
    c.inv[k] = 1.0 / diag[k];
  }

  // the rows A(g, rest), counted then filled: one pass each, no map
  c.row_begin.assign(c.first.size() + 1, 0);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto r = static_cast<std::size_t>(A.row[k]);
    if (mine[r] == 0 || mine[static_cast<std::size_t>(A.col[k])] != 0) continue;
    ++c.row_begin[static_cast<std::size_t>(place[r]) + 1];
  }
  for (std::size_t k = 0; k < c.first.size(); ++k) c.row_begin[k + 1] += c.row_begin[k];
  c.row_col.resize(static_cast<std::size_t>(c.row_begin.back()));
  c.row_val.resize(c.row_col.size());
  {
    std::vector<Index> at(c.row_begin.begin(), c.row_begin.end() - 1);
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto r = static_cast<std::size_t>(A.row[k]);
      if (mine[r] == 0 || mine[static_cast<std::size_t>(A.col[k])] != 0) continue;
      const auto put = static_cast<std::size_t>(at[static_cast<std::size_t>(place[r])]++);
      c.row_col[put] = c.slot[static_cast<std::size_t>(A.col[k])];
      c.row_val[put] = A.value[k];
    }
  }

  // and the columns A(rest, g), which the outer product needs on the left
  std::vector<Index> col_begin(c.first.size() + 1, 0);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto col = static_cast<std::size_t>(A.col[k]);
    if (mine[col] == 0 || mine[static_cast<std::size_t>(A.row[k])] != 0) continue;
    ++col_begin[static_cast<std::size_t>(place[col]) + 1];
  }
  for (std::size_t k = 0; k < c.first.size(); ++k) col_begin[k + 1] += col_begin[k];
  std::vector<int> col_row(static_cast<std::size_t>(col_begin.back()));
  std::vector<double> col_val(col_row.size());
  {
    std::vector<Index> at(col_begin.begin(), col_begin.end() - 1);
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto col = static_cast<std::size_t>(A.col[k]);
      if (mine[col] == 0 || mine[static_cast<std::size_t>(A.row[k])] != 0) continue;
      const auto put = static_cast<std::size_t>(at[static_cast<std::size_t>(place[col])]++);
      col_row[put] = c.slot[static_cast<std::size_t>(A.row[k])];
      col_val[put] = A.value[k];
    }
  }

  // S = C - A10 M^-1 A01, as triplets. C is copied entry by entry and the
  // correction is one dense outer product per eliminated unknown -- fourteen
  // by fourteen for a stress facet in space, and nothing is materialized in
  // between.
  c.reduced.n = c.rest.size();
  std::size_t entries = 0;
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    entries += static_cast<std::size_t>(col_begin[k + 1] - col_begin[k]) *
               static_cast<std::size_t>(c.row_begin[k + 1] - c.row_begin[k]);
  }
  c.reduced.row.reserve(entries + A.nnz());
  c.reduced.col.reserve(entries + A.nnz());
  c.reduced.value.reserve(entries + A.nnz());
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    if (mine[static_cast<std::size_t>(A.row[k])] != 0 ||
        mine[static_cast<std::size_t>(A.col[k])] != 0) {
      continue;
    }
    c.reduced.row.push_back(c.slot[static_cast<std::size_t>(A.row[k])]);
    c.reduced.col.push_back(c.slot[static_cast<std::size_t>(A.col[k])]);
    c.reduced.value.push_back(A.value[k]);
  }
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    for (Index a = col_begin[k]; a < col_begin[k + 1]; ++a) {
      const double left = col_val[static_cast<std::size_t>(a)] * c.inv[k];
      for (Index e = c.row_begin[k]; e < c.row_begin[k + 1]; ++e) {
        c.reduced.row.push_back(col_row[static_cast<std::size_t>(a)]);
        c.reduced.col.push_back(c.row_col[static_cast<std::size_t>(e)]);
        c.reduced.value.push_back(-left * c.row_val[static_cast<std::size_t>(e)]);
      }
    }
  }

  c.rhs.assign(c.rest.size(), 0.0);
  for (std::size_t i = 0; i < c.rest.size(); ++i) {
    c.rhs[i] = b[static_cast<std::size_t>(c.rest[i])];
  }
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    const double weighted = c.inv[k] * b[static_cast<std::size_t>(c.first[k])];
    for (Index a = col_begin[k]; a < col_begin[k + 1]; ++a) {
      c.rhs[static_cast<std::size_t>(col_row[static_cast<std::size_t>(a)])] -=
          col_val[static_cast<std::size_t>(a)] * weighted;
    }
  }
  return c;
}

}  // namespace mimetika::solver
