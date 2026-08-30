#pragma once

#include <cstddef>
#include <vector>

// The contract between a model and whichever library inverts its Riesz map.
//
// This header carries no solver and depends on neither PETSc nor hypre.
// linear_solver/petsc.hpp reads these fields and so does
// linear_solver/hypre.hpp; keeping the norm here is what stops the two paths
// drifting apart in what they mean by it.

namespace mimetika::solver {

// The norm of the product space. P is its Gram matrix, and nothing else.
//
// A maps X to its dual, so a Krylov method -- which needs an operator X -> X --
// requires a map X' -> X. The canonical one is the Riesz map of the inner
// product of X, and with it P^{-1}A has a condition number bounded by the
// inf-sup and continuity constants alone: independent of h. P is therefore not
// an approximation of A. Write the norm and P is determined.
//
//   FLOW      X = H(div) x L^2
//             ||q||^2 = (K^{-1} q, q) + ||div q||^2 ,   ||p||^2 = ||p||_{L2}^2
//
//   ELASTICITY  X = H(div; M) x L^2(R^d) x L^2(skew)
//             ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 ,  A = C^{-1}
//             ||u||^2 = ||u||_{L2}^2 ,   ||r||^2 = ||r||_{L2}^2
//
// Both are the same statement: the first factor carries the material inner
// product plus the graph term of its differential, and every factor after it
// carries plain L^2. The multiplier for weak symmetry is an L^2 factor like the
// displacement -- it is not special, and giving it anything else is a different
// preconditioner.
//
// How each term reads in this dof basis:
//
//   (A sigma, sigma)  is the assembled (0,0) block. The discrete Hodge is that
//                     form, so it is taken rather than rebuilt.
//
//   ||div sigma||^2   is not B^T B. The facet dof is the measure-weighted
//                     moment, so (B sigma)_E = int_E div sigma, the integral;
//                     div sigma is constant on the cell, so its square
//                     integrates to (B sigma)_E^2 / |E| and the term is
//                     B^T diag(1/|E|) B. On a uniform mesh that is a constant
//                     factor and passes for a tuning knob; on a graded one it
//                     varies cell by cell and no constant repairs it.
//
//   ||u||^2           is diag(|E|): a cell dof is the value on the cell.
//
// So one quantity -- the cell measure -- fixes every block.
struct SpaceNorm {
  // the factors of X, as index sets, first factor first
  std::vector<std::vector<int>> factors;
  // the L2 weight of every unknown of factors 1.., one vector per such factor:
  // the measure of the cell that unknown belongs to.
  std::vector<std::vector<double>> l2_weight;

  // The graph term is not stated separately: it is B^T W^{-1} B with W the
  // multiplier block above, which is the whole content of
  //
  //     P = diag( M + B^T W^{-1} B ,  W ) .
  //
  // W is the L2 mass in the dof basis, and that basis differs between the two.
  // Flow's cell unknown is the value of p on the cell, so ||p||^2 = sum p^2 |E|
  // and W = |E|. Elasticity's cell unknowns are moments -- u_dof = int_E u, as
  // CauchyMechanicsModel::displacement shows by dividing by the measure to
  // report a mean -- so ||u||^2 = sum (u_dof/|E|)^2 |E| = sum u_dof^2 / |E| and
  // W = 1/|E|. The rotation is a moment likewise.
  //
  // Measured on the Lame annulus, cond(P^{-1}A) with W = |E| is 8e2 and rising
  // with refinement; with W = 1/|E| it is 3.2, 3.4, 3.5 over the same three
  // meshes -- flat.

  // A constrained unknown is not in the space. Its row of A is the constraint,
  // scale * e_i^T, not a form; leaving the norm's entries there preconditions an
  // equation that is not the one being solved, and the iteration count starts
  // growing with h again. P carries the same row, so those unknowns contribute
  // the identity to P^{-1}A and drop out of the Krylov space.
  // Which multipliers contribute a graph term: the differential constraint does
  // (factor 1), an algebraic one does not. AFW's inf-sup is proved with
  // ||sigma||^2 = (A sigma, sigma) + ||div sigma||^2 -- skw is bounded
  // L^2 -> L^2, so the rotation adds nothing to the stress norm.
  std::size_t differential_factors{1};
  bool carries_graph_term(std::size_t f) const { return f >= 1 && f <= differential_factors; }

  std::vector<int> pinned;
  std::vector<double> pinned_diagonal;

  // Rank-one additions to the first block: each contributes w (t . x)^2, with t
  // supported on `rank_one_dofs`. What the three-field stress norm needs to stay
  // lambda-free; see CauchyMechanicsModel::norm_trace_terms.
  std::vector<std::vector<int>> rank_one_dofs;
  std::vector<std::vector<double>> rank_one_row;
  std::vector<double> rank_one_weight;

  // The de Rham maps an auxiliary-space solver needs.
  //
  // ADS preconditions an H(div) operator by splitting it along the complex --
  // a field becomes a vector potential in H(curl) plus a part carried by the
  // vertex spaces -- and the maps that take it there are the discrete gradient
  // (edges x vertices) and curl (faces x edges). Those are not a new
  // construction: they are the boundary operators of the complex as stored, so
  // a library built on a chain complex hands them over instead of
  // reconstructing them from an element table.
  //
  // Empty means no auxiliary solver is possible and a factorization is used.
  struct Incidence {
    int rows{0}, cols{0};
    std::vector<int> row, col;
    std::vector<double> value;

    bool empty() const { return value.empty(); }
  };
  Incidence discrete_gradient;  // d_1 : vertices -> edges
  Incidence discrete_curl;      // d_2 : edges -> faces

  // ADS also needs the vertex coordinates, one row of `space_dim` per column of
  // the gradient. They are not redundant with the maps above: the complex is
  // metric-free, and the auxiliary spaces the solver builds are spaces of
  // piecewise linear fields -- it recovers their interpolation by applying the
  // maps to the coordinate functions x, y, z. This is the whole metric content
  // ADS asks for.
  std::vector<double> vertex_coordinates;  // row-major, n_vertices x space_dim
  int space_dim{3};

  // Which process owns each entity the two maps address -- vertices, edges,
  // faces -- by the same rule and the same partition that owns the unknowns.
  //
  // The maps above are stated in the complex's own numbering, which is a
  // serial numbering. Distributed, hypre needs each of the three spaces laid
  // out across the ranks, and the faces must be laid out exactly as the block
  // is: its row i has to be that block's row i. That lets the solver renumber
  // all three consistently, without asking the complex to be distributed.
  //
  // Empty on one process, where there is nothing to lay out.
  std::vector<std::vector<int>> entity_owner;  // [k][entity]

  // The lowest-order subspace of the first factor, when the first factor is
  // not itself lowest order.
  //
  // ADS is written for one unknown per facet. Flow's RT space is that already;
  // the AFW stress space is not -- a facet carries d traction components, each
  // measured against the d functions of the facet P_1 basis, so d^2 unknowns
  // sit on it. The auxiliary-space argument still applies, and this is the map
  // it applies through: the injection of the facet-constant moments, which are
  // a subset of the degrees of freedom rather than a computed interpolation,
  // because the space is defined by its moments and the constant one is one of
  // them.
  //
  // Columns are ordered component-major -- all facets of component 0, then
  // component 1 -- so that each component is a contiguous run of the coarse
  // space and ADS can be given it as the scalar H(div) problem it expects.
  //
  // Rows are global unknowns; build_riesz maps them into the block.
  Incidence lowest_order;
  int lowest_order_components{1};

  // THE INTERPOLATIONS, FOR A SPACE ADS CANNOT BUILD THEM FOR.
  //
  // ADS forms Pi -- the map from a vector nodal field into H(div) -- from the
  // vertex coordinates, and that construction is the LOWEST-ORDER one: it
  // assumes a facet carries a single unknown. Given a BDM facet, whose three
  // moments the coordinates say nothing about, the two are supplied here
  // instead and reach HYPRE_ADSSetInterpolations, the documented hook.
  //
  //   rt_interpolation  n_flux x 3 n_vertices, the BDM dofs of a vertex hat
  //   nd_interpolation  n_circ x 3 n_vertices, its circulation dofs
  //
  // Both empty is the lowest-order case and ADS builds its own. Supplying them
  // also constrains the cycle types to the monolithic ones, below 10.
  Incidence rt_interpolation;
  Incidence nd_interpolation;

  // Who owns each vector nodal unknown -- the interpolations' columns, three
  // to a vertex. A fourth partition, alongside entity_owner's three: the other
  // spaces are the complex's and this one is what interpolates into it.
  std::vector<int> interpolation_owner;

  bool empty() const { return factors.empty(); }
};

}  // namespace mimetika::solver
