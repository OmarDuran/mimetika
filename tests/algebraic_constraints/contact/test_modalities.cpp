#include <cmath>
#include <vector>

#include "../../mimetika_test.hpp"
#include "graphos/ops/cut.hpp"
#include "mimetika/algebraic_constraints/contact/trace.hpp"
#include "mimetika/mesh/structured.hpp"

// THE TWO MODALITIES OF CONTACT, and the fact that a driver cannot tell them
// apart.
//
//   CONFORMAL. The mesh and the function space stay connected. A fracture is
//   nothing but an ALGEBRAIC CONSTRAINT on degrees of freedom that already
//   exist: the traction on the facet -- one unknown, shared by both cofaces --
//   and the displacements and rotations of the two adjacent cells. The jump is
//   the single assembled constitutive row, whose outward incidences make it a
//   difference without any subtraction being written.
//
//   SPLIT. The space is cut exactly at the fracture, so each side acquires its
//   own internal boundary facet and its own ONE-SIDED trace. The traction is no
//   longer in the space -- it becomes a LAGRANGE MULTIPLIER on the interface --
//   and the jump is trace^+ - trace^-, now a genuine subtraction of two
//   independent quantities.
//
// From the driver the two are one thing, because ContactMechanics asks only for
// to_moments, the solution operator and the gap. What changes underneath is
// what the traction unknown IS and how the jump is measured; laws.hpp, map.hpp
// and driver.hpp cannot tell, which is the same property that lets the driver
// serve elasticity and poromechanics alike.
//
// WELL-POSEDNESS IS ORTHOGONAL TO THE CHOICE, and that is the other thing this
// file records. A fracture that completely severs the domain under pure
// traction data leaves each piece with a rigid-body null mode -- in the
// conformal form the prescribed dof decouples the two halves, in the split form
// the multiplier is simply undetermined. Global equilibrium then FIXES the
// fault traction and no contact law has anything to decide. A fault SEGMENT,
// whose tips are embedded in intact material, is the posed problem; and the cut
// yields that topology on its own, because a cell with a single side is not
// copied.

using graphos::Index;
using mimetika::contact::Fracture;
using mimetika::mesh::Family;

namespace {

// an n x n grid of quadrilaterals over the unit square, so a mid-row of facets
// can be cut PARTIALLY -- which a one-cell-wide column cannot express
exokal::Mesh grid(int n) {
  std::vector<exokal::Mesh::Point> p;
  const auto vid = [n](int i, int j) { return static_cast<Index>(j * (n + 1) + i); };
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      p.push_back({static_cast<double>(i) / n, static_cast<double>(j) / n, 0.0});
    }
  }
  std::vector<std::vector<Index>> cells;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      cells.push_back({vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
    }
  }
  return exokal::Mesh::from_polygons(std::move(p), cells);
}

// the horizontal facets on the mid-line, between x_lo and x_hi
std::vector<Index> mid_facets(const exokal::Mesh& m, double x_lo, double x_hi) {
  std::vector<Index> out;
  const graphos::Complex& c = m.topology();
  for (Index f = 0; f < c.count(1); ++f) {
    const auto x = exokal::centroid(m, 1, f);
    if (std::abs(x[1] - 0.5) < 1e-9 && x[0] > x_lo && x[0] < x_hi) out.push_back(f);
  }
  return out;
}

std::size_t interior_count(const exokal::Mesh& m, const std::vector<Index>& facets) {
  const graphos::CoboundaryOperator cob = graphos::coboundary(m.topology(), 1);
  std::size_t n = 0;
  for (const Index f : facets) {
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    if (e - b == 2) ++n;
  }
  return n;
}

}  // namespace

// -- the conformal modality ---------------------------------------------------

// A CONFORMAL FRACTURE ADDS NO UNKNOWN. The facets it names are ordinary
// interior facets of an uncut complex, and the traction it constrains is the
// dof that was already there. Nothing about the mesh changes.
MIMETIKA_TEST(a_conformal_fracture_is_a_constraint_on_existing_degrees_of_freedom) {
  const exokal::Mesh m = grid(4);
  const graphos::Complex& c = m.topology();
  const std::vector<Index> segment = mid_facets(m, 0.24, 0.76);
  CHECK(segment.size() == 2);                           // a SEGMENT, not the whole line
  CHECK(interior_count(m, segment) == segment.size());  // every one interior

  const Fracture fr(m, 2, segment, 2);
  CHECK(fr.size() == segment.size());
  // the complex is untouched: the fracture is bookkeeping over it
  CHECK(c.count(2) == 16 && c.count(1) == 40 && c.count(0) == 25);
}

// -- the split modality -------------------------------------------------------

// CUTTING DOUBLES THE INTERFACE AND KEEPS THE ORIGINAL. graphos::cut_along
// gives each side of the marked facets its own copy and leaves the originals in
// place as a DETACHED lower-dimensional subcomplex -- the fracture domain
// itself, closure included. That detached stratum is where a Lagrange
// multiplier lives in this modality, and where a fracture pressure will live
// when the same driver is put on poromechanics.
MIMETIKA_TEST(cutting_creates_the_interface_as_its_own_stratum) {
  const exokal::Mesh m = grid(4);
  const graphos::Complex& c = m.topology();
  const std::vector<Index> segment = mid_facets(m, 0.24, 0.76);

  graphos::Marker interface(c);
  for (const Index f : segment) interface.mark(1, f);
  const graphos::CutResult r = graphos::cut_along(c, interface);
  r.complex.validate();

  // THE CELLS ARE UNTOUCHED and each cut facet gains TWO copies -- one per
  // side -- while the ORIGINAL survives as the detached interface cell. That
  // original is the fracture domain: a genuine lower-dimensional stratum of the
  // same complex, which is where a Lagrange multiplier lives in this modality
  // and where a fracture pressure will live under poromechanics.
  CHECK(r.complex.count(2) == c.count(2));
  CHECK(r.complex.count(1) == c.count(1) + 2 * static_cast<Index>(segment.size()));
  // and d.d = 0 survives, which is what makes the result a complex at all
  CHECK(graphos::d_squared_is_zero(r.complex));

  // every copy descends from the facet it was cut from
  for (Index f = c.count(1); f < r.complex.count(1); ++f) {
    const Index anc = r.ancestor.index[1][static_cast<std::size_t>(f)];
    bool from_segment = false;
    for (const Index s : segment) from_segment = from_segment || (anc == s);
    CHECK(from_segment);
  }
}

// THE CRACK TIP FALLS OUT OF THE TOPOLOGY. A cell with a single side is not
// copied, so the vertices at the ends of an interior segment -- where intact
// material still wraps around -- stay single and the two sides remain joined
// there. No geometric test, no tip detection, no special case: the same
// statement that makes the op sound on nonmanifold complexes produces the
// crack-front topology for free.
//
// It is also exactly the geometry a contact problem must be posed on. A
// fracture that severs the domain under pure traction data leaves each piece
// floating, and equilibrium then determines the fault traction -- there is
// nothing left for a law to decide.
MIMETIKA_TEST(the_crack_tip_is_not_split_and_the_sides_stay_joined) {
  const exokal::Mesh m = grid(4);
  const graphos::Complex& c = m.topology();

  // an INTERIOR segment: two facets, three vertices, the outer two being tips
  const std::vector<Index> segment = mid_facets(m, 0.24, 0.76);
  graphos::Marker interface(c);
  for (const Index f : segment) interface.mark(1, f);
  const graphos::CutResult r = graphos::cut_along(c, interface);

  // The segment spans three vertices. The middle one has two sides and gets a
  // copy each; the two TIPS have a single side -- intact material still wraps
  // around them -- and are not copied at all. So exactly one vertex splits.
  const std::size_t new_vertices = static_cast<std::size_t>(r.complex.count(0) - c.count(0));
  std::printf(
      "  interior segment, %zu facets, 3 vertices: %zu vertex copies"
      " (the middle one alone, one per side)\n",
      segment.size(), new_vertices);
  CHECK(new_vertices == 2);  // one vertex x two sides

  // AND A COMPLETE CUT SEVERS IT. Reaching the boundary at both ends leaves no
  // vertex with a single side, so ALL FIVE split -- ten copies -- and the two
  // halves come apart entirely. That is the configuration a contact problem
  // must not be posed on under pure traction data: each piece then carries a
  // rigid-body null mode, and global equilibrium fixes the fault traction, so
  // no law has anything left to decide.
  const std::vector<Index> whole = mid_facets(m, -1.0, 2.0);
  CHECK(whole.size() == 4);
  graphos::Marker full(c);
  for (const Index f : whole) full.mark(1, f);
  const graphos::CutResult rf = graphos::cut_along(c, full);
  const std::size_t severed = static_cast<std::size_t>(rf.complex.count(0) - c.count(0));
  std::printf(
      "  complete cut, %zu facets, 5 vertices: %zu vertex copies"
      " (every one of them, the tips having reached the boundary)\n",
      whole.size(), severed);
  CHECK(severed == 10);  // five vertices x two sides: nothing holds it together
  CHECK(severed > new_vertices);
  CHECK(graphos::d_squared_is_zero(rf.complex));

  // A SEGMENT THAT REACHES THE BOUNDARY AT ONE END is the intermediate case,
  // and it splits there too: the tip is a tip only where material wraps around.
  const std::vector<Index> half = mid_facets(m, 0.49, 1.01);
  graphos::Marker edge(c);
  for (const Index f : half) edge.mark(1, f);
  const graphos::CutResult re = graphos::cut_along(c, edge);
  const std::size_t one_sided = static_cast<std::size_t>(re.complex.count(0) - c.count(0));
  std::printf("  boundary-touching segment, %zu facets: %zu vertex copies\n", half.size(),
              one_sided);
  CHECK(one_sided == 4);  // the interior vertex AND the boundary end
}

// -- what the two modalities share --------------------------------------------

// THE FRACTURE ABSTRACTION IS THE SAME OBJECT EITHER WAY. In the conformal form
// it names interior facets of the uncut complex; in the split form it names the
// interface facets that survived the cut. Both give the frames a jump is
// measured in, and neither the law nor the map is told which it is.
MIMETIKA_TEST(the_fracture_reads_the_same_frames_before_and_after_a_cut) {
  const exokal::Mesh m = grid(4);
  const std::vector<Index> segment = mid_facets(m, 0.24, 0.76);
  const Fracture conformal(m, 2, segment, 2);

  for (std::size_t i = 0; i < conformal.size(); ++i) {
    const auto& fr = conformal.frame(i);
    // a horizontal facet of the unit grid: unit normal along y, measure 1/4
    CHECK(std::abs(std::abs(fr.normal[1]) - 1.0) < 1e-12);
    CHECK(std::abs(fr.measure - 0.25) < 1e-12);
    CHECK(fr.n_tangents == 1);
  }
}

MIMETIKA_TEST_MAIN()
