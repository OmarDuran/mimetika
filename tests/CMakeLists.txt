# tests mirror include/mimetika/: one test file per header
set(MIMETIKA_UNIT_TESTS
  mesh/test_box
  physics/test_package
  physics/test_flow
  physics/test_darcy_term
  physics/test_immiscible
  physics/test_upwind
  model/test_simulation
  model/test_poroelasticity
  model/test_boundary
  algebraic_constraints/contact/test_laws
  algebraic_constraints/contact/test_associative
  algebraic_constraints/contact/test_map
  algebraic_constraints/contact/test_driver
  algebraic_constraints/contact/test_trace
  algebraic_constraints/contact/test_modalities
  solver/test_bdm_complex
)

foreach(t ${MIMETIKA_UNIT_TESTS})
  string(REPLACE "/" "." name ${t})
  add_executable(${name} ${t}.cpp)
  target_link_libraries(${name} PRIVATE mimetika::mimetika)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

# BENCHMARKS ARE NOT UNIT TESTS. Each solves a full problem -- seconds or
# minutes, not milliseconds -- so they are built and registered only on request
# and carry the `benchmark` label:
#
#     cmake -DMIMETIKA_BUILD_BENCHMARKS=ON ..
#     ctest -L benchmark
#
# A plain `ctest` stays the unit suite. Declared here rather than inside a
# solver branch because both PETSc and hypre have benchmarks under it.
option(MIMETIKA_BUILD_BENCHMARKS "Build and register the benchmark tests" OFF)

# tests that need a solver, built only when there is one
if(MIMETIKA_USE_PETSC)
  foreach(t solver/test_fields solver/test_petsc model/test_confined_compression model/test_diagonal_condensation model/test_hybrid_interface model/test_flow_model model/test_cauchy_mechanics_model model/test_global_patch model/test_hybrid_flux algebraic_constraints/contact/test_jump algebraic_constraints/contact/test_cauchy_mechanics)
    string(REPLACE "/" "." name ${t})
    add_executable(${name} ${t}.cpp)
    target_link_libraries(${name} PRIVATE mimetika::mimetika mimetika_petsc)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()

  # the contact-mechanics benchmarks: a full fault-slip simulation each
  if(MIMETIKA_BUILD_BENCHMARKS)
    foreach(t benchmarks/test_contact_mechanics_0 benchmarks/test_contact_mechanics_1)
      string(REPLACE "/" "." name ${t})
      add_executable(${name} ${t}.cpp)
      target_link_libraries(${name} PRIVATE mimetika::mimetika mimetika_petsc)
      add_test(NAME ${name} COMMAND ${name})
      set_tests_properties(${name} PROPERTIES LABELS benchmark)
    endforeach()
    # benchmarks 2 and 3 state their geometry as a shape complex and mesh it with gmsh
    if(EXOKAL_USE_GMSH)
      add_compile_definitions(MIMETIKA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/_data")
      foreach(t benchmarks/test_contact_mechanics_2 benchmarks/test_contact_mechanics_3)
        string(REPLACE "/" "." name ${t})
        add_executable(${name} ${t}.cpp)
        target_link_libraries(${name} PRIVATE mimetika::mimetika mimetika_petsc exokal::gmsh)
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES LABELS benchmark)
      endforeach()
    endif()
  endif()
endif()

# The direct BDM path, which needs hypre rather than PETSc: ADS on the degree-2
# complex against the facet-constant subspace cycle, on the same problem.
if(MIMETIKA_USE_HYPRE AND MIMETIKA_BUILD_BENCHMARKS)
  add_executable(benchmarks.test_bdm_ads_direct benchmarks/test_bdm_ads_direct.cpp)
  target_link_libraries(benchmarks.test_bdm_ads_direct PRIVATE mimetika::mimetika mimetika_hypre)
  add_test(NAME benchmarks.test_bdm_ads_direct COMMAND benchmarks.test_bdm_ads_direct)
  set_tests_properties(benchmarks.test_bdm_ads_direct PROPERTIES LABELS benchmark)

  add_executable(benchmarks.test_bdm_assembly benchmarks/test_bdm_assembly.cpp)
  target_link_libraries(benchmarks.test_bdm_assembly PRIVATE mimetika::mimetika)
  add_test(NAME benchmarks.test_bdm_assembly COMMAND benchmarks.test_bdm_assembly)
  set_tests_properties(benchmarks.test_bdm_assembly PROPERTIES LABELS benchmark)
endif()
