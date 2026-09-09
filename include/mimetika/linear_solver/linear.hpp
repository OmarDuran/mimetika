#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exokal/forms/assemble.hpp"

// The linear solve behind an interface: a solver takes an assembled system and
// a right-hand side and returns a solution, and is handed neither the model,
// the mesh nor the fields.
//
// A direct factorization validates the operator with no preconditioner in
// between; an iterative method is what makes a large saddle point finish.

namespace mimetika::solver {

using exokal::forms::Index;

// An assembled system in coordinate form — what a TripletSink already holds,
// so nothing is converted before it reaches a backend that wants its own
// format anyway.
struct SparseSystem {
  std::size_t n{0};
  std::vector<Index> row, col;
  std::vector<double> value;

  std::size_t nnz() const { return value.size(); }

  static SparseSystem from(const exokal::forms::TripletSink& s) {
    SparseSystem out;
    out.n = s.residual.size();
    out.row = s.row;
    out.col = s.col;
    out.value = s.value;
    out.add_structural_diagonal();
    return out;
  }

  // The same, moving the triplets rather than copying them: a 22k-cell
  // polyhedral mesh emits about 10^8 of them, and a copy doubles the peak.
  static SparseSystem from(exokal::forms::TripletSink&& s) {
    SparseSystem out;
    out.n = s.residual.size();
    out.row = std::move(s.row);
    out.col = std::move(s.col);
    out.value = std::move(s.value);
    out.add_structural_diagonal();
    return out;
  }

  // The diagonal, structurally present, zero or not. The (p, p) block of a
  // mixed form is empty, not small, and a factorization that indexes the
  // diagonal refuses such a matrix outright while a fieldsplit cannot address
  // a block it cannot find. One zero triplet per row; it changes no product.
  void add_structural_diagonal() {
    row.reserve(row.size() + n);
    col.reserve(col.size() + n);
    value.reserve(value.size() + n);
    for (std::size_t i = 0; i < n; ++i) {
      row.push_back(static_cast<Index>(i));
      col.push_back(static_cast<Index>(i));
      value.push_back(0.0);
    }
  }
};

struct SolveReport {
  bool converged{false};
  int iterations{0};
  double residual{0.0};  // ||Ax - b|| relative to ||b||
  std::string reason;

  // Where the time went, kept apart: the matrix build is linear in the
  // assembly, the preconditioner setup decides whether a mesh is reachable at
  // all, and the iteration is what the preconditioner shortens.
  double assembly_seconds{0.0};  // the Jacobian: the model's own build
  double matrix_seconds{0.0};
  double preconditioner_seconds{0.0};
  double solve_seconds{0.0};
  // the share of the matrix whose columns another process owns: zero on one
  // process, and on several the price of the layout
  double off_rank_fraction{0.0};
  // Which solver the Riesz block got. It is chosen from the size of that
  // block, so a sweep over mesh sizes crosses riesz_exact_limit somewhere and
  // two rows of one table are then two methods.
  std::string block_solver;

  // Whether the first field was eliminated before the solve, and how much was
  // left. A condensed run and a saddle-point run of one model agree in the
  // answer and in nothing else -- different matrix, method and iteration
  // count -- so which ran is reported rather than inferred from the timings.
  bool condensed{false};
  std::size_t condensed_dofs{0};  // the size of S, when it was formed
};

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;
  virtual std::string name() const = 0;

  // x is resized and filled; the report carries whether it converged, in how
  // many iterations, and the relative residual it leaves.
  virtual SolveReport solve(const SparseSystem& A, const std::vector<double>& b,
                            std::vector<double>& x) = 0;
};

// The residual a caller should check, computed from the system rather than
// from whatever the backend chose to report about its own iteration.
inline double true_residual(const SparseSystem& A, const std::vector<double>& b,
                            const std::vector<double>& x) {
  std::vector<double> r = b;
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    r[static_cast<std::size_t>(A.row[k])] -= A.value[k] * x[static_cast<std::size_t>(A.col[k])];
  }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < r.size(); ++i) {
    num += r[i] * r[i];
    den += b[i] * b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace mimetika::solver
