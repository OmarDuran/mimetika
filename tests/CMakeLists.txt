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
)

foreach(t ${MIMETIKA_UNIT_TESTS})
  string(REPLACE "/" "." name ${t})
  add_executable(${name} ${t}.cpp)
  target_link_libraries(${name} PRIVATE mimetika::mimetika)
  add_test(NAME ${name} COMMAND ${name})
endforeach()

# tests that need a solver, built only when there is one
if(MIMETIKA_USE_PETSC)
  foreach(t solver/test_fields solver/test_petsc model/test_confined_compression model/test_diagonal_condensation model/test_hybrid_interface model/test_single_phase model/test_cauchy_elasticity model/test_global_patch model/test_hybrid_flux algebraic_constraints/contact/test_jump algebraic_constraints/contact/test_cauchy_mechanics)
    string(REPLACE "/" "." name ${t})
    add_executable(${name} ${t}.cpp)
    target_link_libraries(${name} PRIVATE mimetika::mimetika mimetika_petsc)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()

  # THE CONTACT-MECHANICS BENCHMARKS ARE NOT UNIT TESTS. Each one runs a full
  # fault-slip simulation -- minutes, not seconds -- so they are built and
  # registered only on request, and carry the `benchmark` label so they can be
  # run alone:
  #
  #     cmake -DMIMETIKA_BUILD_BENCHMARKS=ON ..
  #     ctest -L benchmark
  #
  # A plain `ctest` stays the unit suite.
  option(MIMETIKA_BUILD_BENCHMARKS "Build and register the contact-mechanics benchmark tests" OFF)
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
