#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/forms/assemble.hpp"

// THE LINEAR SOLVE, behind an interface, because the choice of solver is not
// a property of the model.
//
// A saddle point of this kind is solved one way while a method is being
// validated and another way when it is being run: a direct factorization
// answers "is the operator right" without a preconditioner standing between
// the question and the answer, and an iterative method is what makes a large
// problem finish. Both are legitimate and neither should be wired into the
// physics — which is why Simulation produces operators and stops.
//
// THE INTERFACE IS DELIBERATELY NARROW. A solver is handed an assembled
// system and a right-hand side and returns a solution; it is not handed the
// model, the mesh, or the fields. That keeps a matrix-free operator and an
// assembled one interchangeable behind it later, and it means a solver can be
// tested against a matrix nobody assembled from physics at all.

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
    return out;
  }
};

struct SolveReport {
  bool converged{false};
  int iterations{0};
  double residual{0.0};  // ||Ax - b|| relative to ||b||
  std::string reason;
};

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;
  virtual std::string name() const = 0;

  // x is resized and filled; the return value says what happened, and a
  // caller that ignores it gets a throw rather than a silent wrong answer.
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
