#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exokal/forms/assemble.hpp"
#include "exokal/spaces/product_space.hpp"
#include "graphos/core/complex.hpp"
#include "mimetika/linear_solver/linear.hpp"

// A proper spectral equivalence for the stress Hodge of the weak-symmetry
// products on simplices, and the pieces the block-triangular solve needs.
//
// WHY NOT diag(M). Pazner, Kolev & Vassilevski take the (1, 1) block to be its
// own diagonal, and that is earned by their basis: one localized flux unknown a
// subelement face. The facet moments of a BDM stress are not that. Measured,
// kappa(Mt^-1 M) for stabilized_bdm in 3D, over refinement:
//
//                        diag(M)   facet block   cell additive Schwarz
//     Kuhn tetrahedra       56          46               14.7
//     cubes                  7.5         7.4              3.9
//
// flat in h for each. The facet block barely helps -- the coupling is across
// the facets OF A CELL, not within a facet, which is the cell-level bound
// tests/solver/test_matrix_free.cpp states -- and the cell blocks overlap, a
// facet belonging to two cells. Hence additive Schwarz over cells:
//
//     Mt^-1 = sum_E R_E^T (M|_E)^-1 R_E,
//
// M|_E the principal block of the ASSEMBLED operator on E's stress unknowns.
// No coarse space is needed and none is used: M is zeroth order, so it has no
// near-kernel a coarse level would have to carry.
//
// B IS KEPT SPARSE. The block-triangular preconditioner applies B^T once an
// iteration, and through the kernels that would be a second operator apply.
// Assembled, it is a vanishing fraction of the tangent -- a stress unknown
// reaches the cell unknowns of at most two cells -- which is how PKV hold their
// divergence (their Figure 4).
//
// CONSTRAINTS. The operator the Krylov method sees is Simulation::
// apply_symmetric's: a pinned stress unknown keeps its diagonal and loses the
// rest of its row and column. The sink reproduces exactly that -- a pinned
// unknown's off-diagonal couplings are dropped from the cell blocks and its
// column is dropped from B -- so the preconditioner approximates the operator
// actually applied, not the raw tangent. Cell unknowns are never pinned in this
// family; the route refuses a model where one is.
//
// SIGNS. A stencil's view carries boundary signs alongside its unknowns, and
// the kernels have already applied them: TripletSink emits blk(i, j) as it
// stands, and so does this sink, so the two agree entry for entry.

namespace mimetika::solver {

// Which stress unknowns each cell holds, which cell unknowns, and who its face
// neighbours are -- all read off the space and the complex, not assumed.
struct CellStressLayout {
  std::vector<std::vector<int>> stress_of_cell;  // cell -> stress-local, facet order
  std::vector<std::vector<int>> rest_of_cell;    // cell -> rest-local
  std::vector<std::vector<Index>> neighbours;    // cell -> face neighbours
  // stress-local k is held by owner_cell[2k] and owner_cell[2k + 1] (-1 when
  // absent: a boundary facet has one cell), at positions owner_pos[2k], ...
  std::vector<int> owner_cell, owner_pos;

  static CellStressLayout build(const graphos::Complex& topology, int dim,
                                const exokal::spaces::ProductSpace& space,
                                const std::string& stress_field, const std::vector<int>& stress_slot,
                                const std::vector<int>& rest_slot, Index offset = 0) {
    const std::size_t sfi = space.index_of(stress_field);
    const auto ncells = static_cast<std::size_t>(topology.count(dim));
    const auto nfacets = static_cast<std::size_t>(topology.count(dim - 1));
    const graphos::BoundaryOperator& bnd = topology.boundary(dim);

    CellStressLayout L;
    L.stress_of_cell.resize(ncells);
    L.rest_of_cell.resize(ncells);
    L.neighbours.resize(ncells);

    // the stress field lives on facets alone, or a cell block is not the union
    // of its facets' unknowns
    const exokal::spaces::DofMap& smap = space.map(sfi);
    const exokal::spaces::DofLayout& slay = smap.layout();
    for (int k = 0; k <= dim; ++k) {
      if (k != dim - 1 && slay.on(k) != 0) {
        throw std::invalid_argument(
            "CellStressLayout: the stress field carries unknowns off the facets; the cell "
            "Schwarz blocks are built from facet moments alone");
      }
    }
    // a facet's stress unknowns, as facet_dofs lays them out
    std::vector<std::vector<int>> of_facet(nfacets);
    for (std::size_t f = 0; f < nfacets; ++f) {
      for (int l = 0; l < slay.on(dim - 1); ++l) {
        for (int cp = 0; cp < slay.components; ++cp) {
          const Index g = space.offset(sfi) + smap.global(dim - 1, static_cast<Index>(f), l, cp) + offset;
          const int k = stress_slot[static_cast<std::size_t>(g)];
          if (k < 0) {
            throw std::invalid_argument(
                "CellStressLayout: a facet unknown of the stress field is not in the stress block");
          }
          of_facet[f].push_back(k);
        }
      }
    }

    const auto ns = static_cast<std::size_t>(
        std::count_if(stress_slot.begin(), stress_slot.end(), [](int s) { return s >= 0; }));
    L.owner_cell.assign(2 * ns, -1);
    L.owner_pos.assign(2 * ns, -1);
    std::vector<std::vector<Index>> cells_of_facet(nfacets);
    for (std::size_t e = 0; e < ncells; ++e) {
      auto& set = L.stress_of_cell[e];
      for (auto p = bnd.offsets[e]; p < bnd.offsets[e + 1]; ++p) {
        const auto f = static_cast<std::size_t>(bnd.indices[static_cast<std::size_t>(p)]);
        cells_of_facet[f].push_back(static_cast<Index>(e));
        for (const int k : of_facet[f]) {
          const auto kk = static_cast<std::size_t>(k);
          const std::size_t slot = L.owner_cell[2 * kk] < 0 ? 0 : 1;
          if (L.owner_cell[2 * kk + slot] >= 0) {
            throw std::invalid_argument("CellStressLayout: a stress unknown in more than two cells");
          }
          L.owner_cell[2 * kk + slot] = static_cast<int>(e);
          L.owner_pos[2 * kk + slot] = static_cast<int>(set.size());
          set.push_back(k);
        }
      }
    }
    for (std::size_t k = 0; k < ns; ++k) {
      if (L.owner_cell[2 * k] < 0) {
        throw std::invalid_argument("CellStressLayout: a stress unknown belongs to no cell");
      }
    }
    for (std::size_t f = 0; f < nfacets; ++f) {
      if (cells_of_facet[f].size() == 2) {
        const Index a = cells_of_facet[f][0], b = cells_of_facet[f][1];
        L.neighbours[static_cast<std::size_t>(a)].push_back(b);
        L.neighbours[static_cast<std::size_t>(b)].push_back(a);
      }
    }

    // every other field: cell unknowns only
    for (std::size_t fi = 0; fi < space.n_fields(); ++fi) {
      if (fi == sfi) continue;
      const exokal::spaces::DofMap& map = space.map(fi);
      const exokal::spaces::DofLayout& lay = map.layout();
      for (int k = 0; k < dim; ++k) {
        if (lay.on(k) != 0) {
          throw std::invalid_argument(
              "CellStressLayout: a field other than the stress carries unknowns below the cells; "
              "the Schur complement pattern assumes cell unknowns");
        }
      }
      for (std::size_t e = 0; e < ncells; ++e) {
        for (int l = 0; l < lay.on(dim); ++l) {
          for (int cp = 0; cp < lay.components; ++cp) {
            const Index g = space.offset(fi) + map.global(dim, static_cast<Index>(e), l, cp) + offset;
            const int r = rest_slot[static_cast<std::size_t>(g)];
            if (r < 0) throw std::invalid_argument("CellStressLayout: a cell unknown outside the rest block");
            L.rest_of_cell[e].push_back(r);
          }
        }
      }
    }
    return L;
  }
};

// One assembly pass, two products: each cell's principal stress block, and B.
class CellSchwarzSink final : public exokal::forms::Sink {
 public:
  CellSchwarzSink(const CellStressLayout& layout, std::vector<int> stress_slot,
                  std::vector<int> rest_slot, std::vector<char> pinned = {})
      : blocks(layout.stress_of_cell.size()),
        column(layout.owner_cell.size() / 2),
        layout_(&layout),
        stress_slot_(std::move(stress_slot)),
        rest_slot_(std::move(rest_slot)),
        pinned_(std::move(pinned)) {
    for (std::size_t e = 0; e < blocks.size(); ++e) {
      const std::size_t q = layout.stress_of_cell[e].size();
      blocks[e].assign(q * q, 0.0);
    }
  }

  std::vector<std::vector<double>> blocks;                  // cell -> M|_E, row-major
  std::vector<std::vector<std::pair<int, double>>> column;  // stress-local -> (rest-local, B)

  void scatter(const exokal::forms::Stencil& st, const exokal::ad::LocalSystem& sys) override {
    const auto& dofs = st.view.dofs;
    const exokal::ad::LocalSpace& sp = sys.space();
    const auto& owner = layout_->owner_cell;
    const auto& pos = layout_->owner_pos;
    for (std::size_t bi = 0; bi < sp.n_blocks(); ++bi) {
      for (std::size_t bj = 0; bj < sp.n_blocks(); ++bj) {
        if (!sys.has_block(bi, bj)) continue;
        const exokal::numerics::Dense& blk = sys.block(bi, bj);
        for (std::size_t i = 0; i < blk.rows(); ++i) {
          const auto gr = static_cast<std::size_t>(dofs[sp.begin(bi) + i]);
          const int sr = stress_slot_[gr];
          for (std::size_t j = 0; j < blk.cols(); ++j) {
            const double v = blk(i, j);
            if (v == 0.0) continue;
            const auto gc = static_cast<std::size_t>(dofs[sp.begin(bj) + j]);
            const int sc = stress_slot_[gc];
            if (sr >= 0 && sc >= 0) {
              // what apply_symmetric does to a pinned unknown: its diagonal
              // stays, every other entry of its row and column goes
              if (gr != gc && (is_pinned(gr) || is_pinned(gc))) continue;
              // every cell holding BOTH unknowns sees the entry: two for a pair
              // on one facet, one for a pair on two facets of the same cell
              const auto r2 = 2 * static_cast<std::size_t>(sr), c2 = 2 * static_cast<std::size_t>(sc);
              for (std::size_t a = 0; a < 2; ++a) {
                const int e = owner[r2 + a];
                if (e < 0) continue;
                for (std::size_t b = 0; b < 2; ++b) {
                  if (owner[c2 + b] != e) continue;
                  const std::size_t q = layout_->stress_of_cell[static_cast<std::size_t>(e)].size();
                  blocks[static_cast<std::size_t>(e)][static_cast<std::size_t>(pos[r2 + a]) * q +
                                                      static_cast<std::size_t>(pos[c2 + b])] += v;
                }
              }
            } else if (sc >= 0 && rest_slot_[gr] >= 0 && !is_pinned(gc)) {
              column[static_cast<std::size_t>(sc)].emplace_back(rest_slot_[gr], v);
            }
          }
        }
      }
    }
  }

 private:
  bool is_pinned(std::size_t g) const { return !pinned_.empty() && pinned_[g] != 0; }

  const CellStressLayout* layout_;
  std::vector<int> stress_slot_, rest_slot_;
  std::vector<char> pinned_;
};

// Mt^-1 = sum_E R_E^T (M|_E)^-1 R_E, the inverses explicit: an application is
// then a dense product a cell, which PKV measure as fastest at low order.
struct CellSchwarz {
  const CellStressLayout* layout{nullptr};
  std::vector<std::vector<double>> inverse;  // cell -> (M|_E)^-1, row-major
  std::size_t n_stress{0};

  // Inverts the blocks IN PLACE and takes them: the peak is one set of cell
  // matrices and one cell's scratch, not two sets.
  static CellSchwarz from(const CellStressLayout& layout, std::vector<std::vector<double>>&& blocks) {
    CellSchwarz out;
    out.layout = &layout;
    out.n_stress = layout.owner_cell.size() / 2;
    std::vector<double> scratch;
    for (std::size_t e = 0; e < blocks.size(); ++e) {
      const auto q = layout.stress_of_cell[e].size();
      if (!invert(blocks[e], q, scratch)) {
        throw std::runtime_error("CellSchwarz: the stress block of cell " + std::to_string(e) +
                                 " is singular, so M is not positive definite there");
      }
    }
    out.inverse = std::move(blocks);
    return out;
  }

  // y = (scale Mt)^-1 x, over the stress-local numbering
  void apply(const double* x, double* y, double scale) const {
    std::fill(y, y + n_stress, 0.0);
    for (std::size_t e = 0; e < inverse.size(); ++e) {
      const auto& set = layout->stress_of_cell[e];
      const auto& inv = inverse[e];
      const std::size_t q = set.size();
      for (std::size_t i = 0; i < q; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < q; ++j) s += inv[i * q + j] * x[static_cast<std::size_t>(set[j])];
        y[static_cast<std::size_t>(set[i])] += s;
      }
    }
    const double w = 1.0 / scale;
    for (std::size_t k = 0; k < n_stress; ++k) y[k] *= w;
  }

 private:
  // Gauss-Jordan with partial pivoting; `a` is replaced by its inverse
  static bool invert(std::vector<double>& a, std::size_t n, std::vector<double>& e) {
    e.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) e[i * n + i] = 1.0;
    for (std::size_t c = 0; c < n; ++c) {
      std::size_t p = c;
      for (std::size_t r = c + 1; r < n; ++r) {
        if (std::fabs(a[r * n + c]) > std::fabs(a[p * n + c])) p = r;
      }
      if (!(std::fabs(a[p * n + c]) > 0.0)) return false;
      if (p != c) {
        for (std::size_t k = 0; k < n; ++k) {
          std::swap(a[c * n + k], a[p * n + k]);
          std::swap(e[c * n + k], e[p * n + k]);
        }
      }
      const double d = a[c * n + c];
      for (std::size_t k = 0; k < n; ++k) {
        a[c * n + k] /= d;
        e[c * n + k] /= d;
      }
      for (std::size_t r = 0; r < n; ++r) {
        if (r == c) continue;
        const double f = a[r * n + c];
        if (f == 0.0) continue;
        for (std::size_t k = 0; k < n; ++k) {
          a[r * n + k] -= f * a[c * n + k];
          e[r * n + k] -= f * e[c * n + k];
        }
      }
    }
    a.swap(e);
    return true;
  }
};

// B, rest x stress, held by stress column with duplicates merged.
struct SparseColumns {
  std::vector<std::size_t> start;  // stress-local k owns [start[k], start[k + 1])
  std::vector<int> row;            // rest-local
  std::vector<double> value;

  // Takes the sink's lists and releases each as it is copied, so the two copies
  // never coexist in full.
  static SparseColumns from(std::vector<std::vector<std::pair<int, double>>>&& column) {
    SparseColumns out;
    out.start.assign(column.size() + 1, 0);
    std::size_t total = 0;
    for (auto& col : column) {
      std::sort(col.begin(), col.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      int last = -1;
      for (const auto& entry : col) {
        if (entry.first != last) {
          ++total;
          last = entry.first;
        }
      }
    }
    out.row.reserve(total);
    out.value.reserve(total);
    for (std::size_t k = 0; k < column.size(); ++k) {
      out.start[k] = out.row.size();
      int last = -1;
      for (const auto& [r, v] : column[k]) {
        if (r != last) {
          out.row.push_back(r);
          out.value.push_back(v);
          last = r;
        } else {
          out.value.back() += v;
        }
      }
      std::vector<std::pair<int, double>>().swap(column[k]);
    }
    out.start[column.size()] = out.row.size();
    return out;
  }

  // out -= B^T y
  void subtract_transpose(const double* y_rest, double* out_stress) const {
    for (std::size_t k = 0; k + 1 < start.size(); ++k) {
      double s = 0.0;
      for (std::size_t p = start[k]; p < start[k + 1]; ++p) {
        s += value[p] * y_rest[static_cast<std::size_t>(row[p])];
      }
      out_stress[k] -= s;
    }
  }
};

// Compressed rows with sorted columns: what a PETSc AIJ matrix takes without a copy.
struct Csr {
  int n{0};
  std::vector<int> ptr, col;
  std::vector<double> val;
  std::size_t nnz() const { return val.size(); }
};

// S~ = sum_E (B R_E^T) (M|_E)^-1 (R_E B^T).
//
// THE PATTERN IS KNOWN BEFORE THE VALUES. Cell E's stress reaches the cell
// unknowns of E and its face neighbours -- its patch -- so a row of cell a
// couples to every cell within face distance two of a. That is laid out first
// and the values are added into it, so no triplet list the size of the
// contributions is ever held: on a hybrid mesh of 10^5 cells that list would be
// ~10^8 entries against ~6 x 10^7 in the result.
inline Csr schwarz_schur(const CellStressLayout& L, const CellSchwarz& mt, const SparseColumns& b,
                         std::size_t n_rest) {
  const std::size_t ncells = L.stress_of_cell.size();
  std::vector<int> cell_of_rest(n_rest, -1);
  for (std::size_t e = 0; e < ncells; ++e) {
    for (const int r : L.rest_of_cell[e]) cell_of_rest[static_cast<std::size_t>(r)] = static_cast<int>(e);
  }
  for (std::size_t r = 0; r < n_rest; ++r) {
    if (cell_of_rest[r] < 0) throw std::invalid_argument("schwarz_schur: a rest unknown owned by no cell");
  }

  // cells within face distance two, per cell, sorted
  std::vector<std::vector<Index>> reach(ncells);
  for (std::size_t a = 0; a < ncells; ++a) {
    auto& out = reach[a];
    out.push_back(static_cast<Index>(a));
    for (const Index e : L.neighbours[a]) {
      out.push_back(e);
      for (const Index c : L.neighbours[static_cast<std::size_t>(e)]) out.push_back(c);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }

  // symbolic: one row pattern a cell, shared by that cell's unknowns
  Csr S;
  S.n = static_cast<int>(n_rest);
  S.ptr.assign(n_rest + 1, 0);
  for (std::size_t r = 0; r < n_rest; ++r) {
    std::size_t len = 0;
    for (const Index c : reach[static_cast<std::size_t>(cell_of_rest[r])]) {
      len += L.rest_of_cell[static_cast<std::size_t>(c)].size();
    }
    S.ptr[r + 1] = S.ptr[r] + static_cast<int>(len);
  }
  S.col.resize(static_cast<std::size_t>(S.ptr[n_rest]));
  S.val.assign(S.col.size(), 0.0);
  std::vector<int> pattern;
  for (std::size_t a = 0; a < ncells; ++a) {
    pattern.clear();
    for (const Index c : reach[a]) {
      for (const int r : L.rest_of_cell[static_cast<std::size_t>(c)]) pattern.push_back(r);
    }
    std::sort(pattern.begin(), pattern.end());
    for (const int r : L.rest_of_cell[a]) {
      std::copy(pattern.begin(), pattern.end(), S.col.begin() + S.ptr[static_cast<std::size_t>(r)]);
    }
  }
  std::vector<std::vector<Index>>().swap(reach);

  // numeric: a dense product a cell, added into the known rows
  std::vector<int> slot(n_rest, -1), touch;
  std::vector<double> Bl, BI;
  for (std::size_t e = 0; e < ncells; ++e) {
    const auto& set = L.stress_of_cell[e];
    const auto& inv = mt.inverse[e];
    const std::size_t q = set.size();
    touch.clear();
    for (const int k : set) {
      const auto kk = static_cast<std::size_t>(k);
      for (std::size_t p = b.start[kk]; p < b.start[kk + 1]; ++p) {
        const auto r = static_cast<std::size_t>(b.row[p]);
        if (slot[r] < 0) {
          slot[r] = static_cast<int>(touch.size());
          touch.push_back(b.row[p]);
        }
      }
    }
    const std::size_t t = touch.size();
    Bl.assign(t * q, 0.0);
    for (std::size_t j = 0; j < q; ++j) {
      const auto k = static_cast<std::size_t>(set[j]);
      for (std::size_t p = b.start[k]; p < b.start[k + 1]; ++p) {
        Bl[static_cast<std::size_t>(slot[static_cast<std::size_t>(b.row[p])]) * q + j] += b.value[p];
      }
    }
    BI.assign(t * q, 0.0);
    for (std::size_t i = 0; i < t; ++i) {
      for (std::size_t j = 0; j < q; ++j) {
        const double bij = Bl[i * q + j];
        if (bij == 0.0) continue;
        for (std::size_t k = 0; k < q; ++k) BI[i * q + k] += bij * inv[j * q + k];
      }
    }
    for (std::size_t i = 0; i < t; ++i) {
      const auto ri = static_cast<std::size_t>(touch[i]);
      const auto lo = S.col.begin() + S.ptr[ri];
      const auto hi = S.col.begin() + S.ptr[ri + 1];
      for (std::size_t j = 0; j < t; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < q; ++k) s += BI[i * q + k] * Bl[j * q + k];
        if (s == 0.0) continue;
        const auto it = std::lower_bound(lo, hi, touch[j]);
        if (it == hi || *it != touch[j]) {
          throw std::logic_error("schwarz_schur: a coupling outside the face-distance-two pattern");
        }
        S.val[static_cast<std::size_t>(it - S.col.begin())] += s;
      }
    }
    for (const int r : touch) slot[static_cast<std::size_t>(r)] = -1;
  }

  // compress: the pattern is a superset, and an explicit zero is not a coupling
  int w = 0;
  for (std::size_t r = 0; r < n_rest; ++r) {
    const int lo = S.ptr[r], hi = S.ptr[r + 1];
    S.ptr[r] = w;
    for (int p = lo; p < hi; ++p) {
      if (S.val[static_cast<std::size_t>(p)] == 0.0) continue;
      S.col[static_cast<std::size_t>(w)] = S.col[static_cast<std::size_t>(p)];
      S.val[static_cast<std::size_t>(w)] = S.val[static_cast<std::size_t>(p)];
      ++w;
    }
  }
  S.ptr[n_rest] = w;
  S.col.resize(static_cast<std::size_t>(w));
  S.val.resize(static_cast<std::size_t>(w));
  S.col.shrink_to_fit();
  S.val.shrink_to_fit();
  return S;
}

}  // namespace mimetika::solver
