#pragma once

// THE CONDITIONING OF ONE CELL'S INNER PRODUCT, and the selection it drives.
//
// The metric-degeneracy scan judges a cell by its measure against its node
// star, which catches a collapsed cell and misses a sliver of ordinary volume:
// a tetrahedron whose flux block has lambda_min ~ 1e-8 against lambda_max ~ 1e2
// sits well above the collapse threshold and still breaks every factorization
// built on the stabilized product. Its block says so directly. cond() is the
// exact 2-norm condition number of the cell block in the facet-flux basis --
// the basis the assembled matrix and its preconditioner live in -- from a
// Jacobi eigensolve, to roundoff; infinite where the block is not positive.
//
// cond_selection() zeroes eta where the stabilized member's conditioning
// exceeds a threshold. It composes with the scan's selection rather than
// replacing it: a cell either selector flags takes the diagonal star.

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "exokal/numerics/dense.hpp"

namespace mimetika {

inline double block_conditioning(const exokal::numerics::Dense& M) {
  if (M.rows() == 0) return 0.0;
  const exokal::numerics::SymmetricEigen eig = exokal::numerics::symmetric_eigen(M);
  const double lo = eig.values.front(), hi = eig.values.back();
  return lo > 0.0 ? hi / lo : std::numeric_limits<double>::infinity();
}

// eta[e] = 0 wherever block(e) -- the STABILIZED member's block, built with
// ones everywhere -- has cond > threshold; a cell already at eta = 0 is
// left alone. Returns how many cells this selector switched. An empty block
// (a cell this process does not own) is not judged.
inline std::size_t cond_selection(std::vector<double>& eta, double threshold,
                                  const std::function<const exokal::numerics::Dense&(std::size_t)>& block) {
  std::size_t switched = 0;
  for (std::size_t e = 0; e < eta.size(); ++e) {
    if (eta[e] == 0.0) continue;
    const exokal::numerics::Dense& M = block(e);
    if (M.rows() == 0) continue;
    if (block_conditioning(M) > threshold) {
      eta[e] = 0.0;
      ++switched;
    }
  }
  return switched;
}

}  // namespace mimetika
