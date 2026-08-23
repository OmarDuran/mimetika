#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"
#include "mimetika/model/flow_model.hpp"
#include "mimetika/linear_solver/fields.hpp"

// The factors of the product space, as index sets.
//
// A block preconditioner is only a preconditioner for the whole operator if its
// blocks partition the unknowns: one left out of every block is never
// corrected, one in two blocks is corrected twice, and neither shows up as
// anything but a convergence rate nobody can explain. So the property under
// test is the partition itself, on the spaces the models actually build.

using mimetika::solver::blocks_partition;
using mimetika::solver::field_blocks;
using mimetika::solver::FieldBlock;
using mimetika::solver::stratum_field_blocks;

namespace {

// the sizes a block split must reproduce: the flux carries `moments` per facet
// per component, the cell unknown one per cell
std::size_t total(const std::vector<FieldBlock>& b) {
  std::size_t n = 0;
  for (const FieldBlock& f : b) n += f.size();
  return n;
}

const FieldBlock& named(const std::vector<FieldBlock>& b, const std::string& name) {
  for (const FieldBlock& f : b) {
    if (f.name == name) return f;
  }
  throw std::runtime_error("no block named " + name);
}

}  // namespace

// The blocks tile the vector: every unknown in exactly one.
MIMETIKA_TEST(the_flow_blocks_partition_the_unknowns) {
  const auto m = mimetika::mesh::annulus(6, 3, 2, mimetika::mesh::Family::simplex, 1.0, 10.0, 1.0);
  mimetika::FlowModel model(m, 2, 1.0, exokal::hodge::FluxOperators::Realization::derham_rt);
  model.build();
  const auto& e = model.simulation().epoch();

  const auto atoms = stratum_field_blocks(e);
  const auto blocks = field_blocks(e);
  CHECK(blocks_partition(atoms, model.simulation().n_dofs()));
  CHECK(blocks_partition(blocks, model.simulation().n_dofs()));
  CHECK(total(atoms) == model.simulation().n_dofs());
  CHECK(total(blocks) == total(atoms));
}

// One factor per field, and the cell unknown is one per cell -- the sizes are
// what the Riesz map will be built against, so they are asserted rather than
// assumed.
MIMETIKA_TEST(the_flow_blocks_are_the_flux_and_the_pressure) {
  const auto m = mimetika::mesh::annulus(6, 3, 2, mimetika::mesh::Family::simplex, 1.0, 10.0, 1.0);
  mimetika::FlowModel model(m, 2, 1.0, exokal::hodge::FluxOperators::Realization::derham_rt);
  model.build();
  const auto blocks = field_blocks(model.simulation().epoch());

  CHECK(blocks.size() == 2);
  // the pressure: one unknown per 2-cell
  CHECK(named(blocks, "p_0").size() == static_cast<std::size_t>(m.count(2)));
  // the flux: one moment per facet on RT
  CHECK(named(blocks, "q_0").size() == static_cast<std::size_t>(m.count(1)));
}

// A single stratum gives each field one run; the atoms and the grouped blocks
// then agree, which is the case every fixed-dimensional problem is in.
MIMETIKA_TEST(one_stratum_gives_each_field_a_single_run) {
  const auto m = mimetika::mesh::annulus(4, 2, 2, mimetika::mesh::Family::simplex, 1.0, 10.0, 1.0);
  mimetika::CauchyMechanicsModel model(m, 2, mimetika::ElasticMaterial(1.0, 1.0),
                                        exokal::hodge::StressOperators::Realization::derham_bdm);
  model.build();
  const auto& e = model.simulation().epoch();
  CHECK(e.n_strata() == 1);

  const auto atoms = stratum_field_blocks(e);
  const auto blocks = field_blocks(e);
  CHECK(atoms.size() == blocks.size());
  for (const FieldBlock& f : blocks) CHECK(f.ranges.size() == 1);
  CHECK(blocks_partition(blocks, model.simulation().n_dofs()));

  // the atoms carry the stratum in the name, the blocks do not
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    CHECK(atoms[i].name == "ambient." + blocks[i].name);
    CHECK(atoms[i].ranges[0].begin == blocks[i].ranges[0].begin);
    CHECK(atoms[i].ranges[0].end == blocks[i].ranges[0].end);
  }
}

// The runs are ascending and abut: the layout is contiguous per field, which is
// what lets a strided IS stand in for an explicit index array.
MIMETIKA_TEST(the_runs_are_contiguous_and_ordered) {
  const auto m = mimetika::mesh::annulus(4, 2, 3, mimetika::mesh::Family::simplex, 1.0, 10.0, 1.0);
  mimetika::CauchyMechanicsModel model(m, 3, mimetika::ElasticMaterial(1.0, 1.0),
                                        exokal::hodge::StressOperators::Realization::derham_bdm);
  model.build();
  const auto atoms = stratum_field_blocks(model.simulation().epoch());

  std::size_t at = 0;
  for (const FieldBlock& f : atoms) {
    CHECK(f.ranges.size() == 1);
    CHECK(f.ranges[0].begin == at);  // no gap, no overlap
    CHECK(f.ranges[0].end > f.ranges[0].begin);
    at = f.ranges[0].end;
  }
  CHECK(at == model.simulation().n_dofs());

  // and indices() materializes exactly that run
  const auto idx = atoms.front().indices();
  CHECK(idx.size() == atoms.front().size());
  CHECK(idx.front() == 0);
  CHECK(static_cast<std::size_t>(idx.back()) == atoms.front().ranges[0].end - 1);
}

// A doubly-claimed or unclaimed unknown is refused, which is the whole value of
// the check: it is what a hand-written split gets wrong.
MIMETIKA_TEST(a_split_that_is_not_a_partition_is_refused) {
  CHECK(blocks_partition({FieldBlock{"a", {{0, 4}}}, FieldBlock{"b", {{4, 10}}}}, 10));
  CHECK(!blocks_partition({FieldBlock{"a", {{0, 4}}}, FieldBlock{"b", {{3, 10}}}}, 10));  // overlap
  CHECK(!blocks_partition({FieldBlock{"a", {{0, 4}}}, FieldBlock{"b", {{5, 10}}}}, 10));  // gap
  CHECK(!blocks_partition({FieldBlock{"a", {{0, 4}}}}, 10));                              // short
  CHECK(!blocks_partition({FieldBlock{"a", {{0, 12}}}}, 10));                             // past n
}

MIMETIKA_TEST_MAIN()
