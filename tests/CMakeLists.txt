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
  # benchmark 2 states its geometry as a shape complex and meshes it with gmsh
  if(EXOKAL_USE_GMSH)
    add_compile_definitions(MIMETIKA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/_data")
    add_executable(benchmarks.test_contact_mechanics_2 benchmarks/test_contact_mechanics_2.cpp)
    target_link_libraries(benchmarks.test_contact_mechanics_2
      PRIVATE mimetika::mimetika mimetika_petsc exokal::gmsh)
    add_test(NAME benchmarks.test_contact_mechanics_2
             COMMAND benchmarks.test_contact_mechanics_2)
    add_executable(benchmarks.test_contact_mechanics_3 benchmarks/test_contact_mechanics_3.cpp)
    target_link_libraries(benchmarks.test_contact_mechanics_3
      PRIVATE mimetika::mimetika mimetika_petsc exokal::gmsh)
    add_test(NAME benchmarks.test_contact_mechanics_3
             COMMAND benchmarks.test_contact_mechanics_3)
  endif()

  foreach(t solver/test_fields solver/test_petsc model/test_confined_compression model/test_diagonal_condensation model/test_single_phase model/test_cauchy_elasticity model/test_global_patch algebraic_constraints/contact/test_jump algebraic_constraints/contact/test_cauchy_mechanics benchmarks/test_contact_mechanics_0 benchmarks/test_contact_mechanics_1)
    string(REPLACE "/" "." name ${t})
    add_executable(${name} ${t}.cpp)
    target_link_libraries(${name} PRIVATE mimetika::mimetika mimetika_petsc)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
endif()
