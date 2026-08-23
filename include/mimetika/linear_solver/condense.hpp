#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "mimetika/linear_solver/linear.hpp"

// Static condensation of a diagonal block.
//
// A diagonal star -- exokal's diagonal_tpfa and diagonal_afw -- makes the
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
// Why it is worth doing beyond the size. The saddle point is indefinite and its
// first block is what a Riesz map spends its effort on; S has no such block.
// The elimination is also rank-revealing where a factorization is not: handed
// the Kuhn tetrahedra, on which diagonal_afw carries one spurious rotation per
// interior cube face, MUMPS reports CONVERGED and returns 1e16, while the
// condensation runs out of pivots and says so.
//
// The cost is one pass. Each eliminated unknown contributes the outer product
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

  // The eliminated field, back out of its own rows. One division per unknown:
  // no solve and no communication, because the row of a facet unknown reaches
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
      if (inv[k] == 0.0) continue;  // not this rank's row to recover
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

// Is the block diagonal? The question the specialization turns on, asked of the
// assembled matrix rather than of the product's name: a star that stops being
// diagonal, for any reason, must stop being condensed here.
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
//
// On several processes, `owners` names who emits what. Distributed assembly
// gives a rank complete rows for the unknowns it owns and for its halo, so:
//
//   a reduced row is emitted by the rank that owns it, once. Row i of S sums
//   A(i,g) M_gg^-1 A(g,.) over the eliminated unknowns g in row i, and those
//   are the facets of an owned cell -- halo, therefore complete here. No
//   contribution ever lands on another rank's row, so nothing is stashed.
//
//   the eliminated field is recovered wherever its row is complete, which is
//   owned and halo: exactly the unknowns a cell's stress reconstruction reads.
//
// Serially `owners` is null and every row is this one's.
inline Condensation condense(const SparseSystem& A, const std::vector<double>& b,
                             std::vector<int> first, const std::vector<int>* owners = nullptr,
                             int rank = 0) {
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
  // A zero diagonal is a row this rank does not hold, not a singular star.
  // Serially there is no such row and a zero is an error; distributed, an
  // unknown outside this rank's owned and halo sets has no entries here at
  // all, and is not this rank's to eliminate.
  c.inv.assign(c.first.size(), 0.0);
  std::vector<char> usable(c.first.size(), 0);
  for (std::size_t k = 0; k < c.first.size(); ++k) {
    if (diag[k] == 0.0) {
      if (owners == nullptr) {
        throw std::invalid_argument("condense: a zero on the diagonal of the eliminated block");
      }
      continue;
    }
    c.inv[k] = 1.0 / diag[k];
    usable[k] = 1;
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

  // The couplings of each kept row, gathered by row: its direct (rest, rest)
  // entries and its (rest, g) reaches into the eliminated block.
  //
  // S = C - A10 M^-1 A01, and the outer-product form of that correction emits
  // one triplet per (row, g, column) triple -- 196 per eliminated stress slot,
  // gigabytes of duplicates on a few hundred thousand cells, for a matrix
  // whose true row holds a few dozen entries. So S is formed row by row
  // instead: a scratch the size of the reduced system, stamped per row,
  // accumulates every contribution in place and each row is emitted once,
  // already merged. The peak is the merged matrix, not the algebra's
  // intermediate.
  const std::size_t n_rest = c.rest.size();
  std::vector<Index> direct_begin(n_rest + 1, 0), to_g_begin(n_rest + 1, 0);
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto r = static_cast<std::size_t>(A.row[k]);
    if (mine[r] != 0) continue;
    const auto col = static_cast<std::size_t>(A.col[k]);
    ++(mine[col] != 0 ? to_g_begin : direct_begin)[static_cast<std::size_t>(c.slot[r]) + 1];
  }
  for (std::size_t i = 0; i < n_rest; ++i) {
    direct_begin[i + 1] += direct_begin[i];
    to_g_begin[i + 1] += to_g_begin[i];
  }
  std::vector<int> direct_col(static_cast<std::size_t>(direct_begin[n_rest]));
  std::vector<double> direct_val(direct_col.size());
  std::vector<int> to_g(static_cast<std::size_t>(to_g_begin[n_rest]));
  std::vector<double> to_g_val(to_g.size());
  {
    std::vector<Index> at_d(direct_begin.begin(), direct_begin.end() - 1);
    std::vector<Index> at_g(to_g_begin.begin(), to_g_begin.end() - 1);
    for (std::size_t k = 0; k < A.nnz(); ++k) {
      const auto r = static_cast<std::size_t>(A.row[k]);
      if (mine[r] != 0) continue;
      const auto col = static_cast<std::size_t>(A.col[k]);
      const auto sr = static_cast<std::size_t>(c.slot[r]);
      if (mine[col] != 0) {
        const auto put = static_cast<std::size_t>(at_g[sr]++);
        to_g[put] = place[col];
        to_g_val[put] = A.value[k];
      } else {
        const auto put = static_cast<std::size_t>(at_d[sr]++);
        direct_col[put] = c.slot[col];
        direct_val[put] = A.value[k];
      }
    }
  }

  c.reduced.n = n_rest;
  const auto ours = [&](std::size_t reduced_row) {
    return owners == nullptr ||
           (*owners)[static_cast<std::size_t>(c.rest[reduced_row])] == rank;
  };
  std::vector<double> acc(n_rest, 0.0);
  std::vector<int> last(n_rest, -1);
  std::vector<int> touched;
  c.rhs.assign(n_rest, 0.0);
  for (std::size_t r = 0; r < n_rest; ++r) {
    if (!ours(r)) continue;
    touched.clear();
    const auto put = [&](int cc, double v) {
      const auto ci = static_cast<std::size_t>(cc);
      if (last[ci] != static_cast<int>(r)) {
        last[ci] = static_cast<int>(r);
        acc[ci] = 0.0;
        touched.push_back(cc);
      }
      acc[ci] += v;
    };
    for (Index k = direct_begin[r]; k < direct_begin[r + 1]; ++k) {
      put(direct_col[static_cast<std::size_t>(k)], direct_val[static_cast<std::size_t>(k)]);
    }
    double rhs_r = b[static_cast<std::size_t>(c.rest[r])];
    for (Index k = to_g_begin[r]; k < to_g_begin[r + 1]; ++k) {
      const auto g = static_cast<std::size_t>(to_g[static_cast<std::size_t>(k)]);
      if (usable[g] == 0) {
        throw std::invalid_argument(
            "condense: an eliminated unknown reaches a row this rank owns, and its own row is "
            "not here -- the halo does not cover the facets of an owned cell");
      }
      const double left = to_g_val[static_cast<std::size_t>(k)] * c.inv[g];
      for (Index e = c.row_begin[g]; e < c.row_begin[g + 1]; ++e) {
        put(c.row_col[static_cast<std::size_t>(e)], -left * c.row_val[static_cast<std::size_t>(e)]);
      }
      rhs_r -= left * b[static_cast<std::size_t>(c.first[g])];
    }
    c.rhs[r] = rhs_r;
    for (const int cc : touched) {
      c.reduced.row.push_back(static_cast<Index>(r));
      c.reduced.col.push_back(static_cast<Index>(cc));
      c.reduced.value.push_back(acc[static_cast<std::size_t>(cc)]);
    }
  }
  return c;
}

}  // namespace mimetika::solver
