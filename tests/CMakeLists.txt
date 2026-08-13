# tests mirror include/mimetika/: one test file per header
set(MIMETIKA_UNIT_TESTS
  physics/test_package
  physics/test_flow
  model/test_simulation
  model/test_poroelasticity
  model/test_boundary
)

foreach(t ${MIMETIKA_UNIT_TESTS})
  string(REPLACE "/" "." name ${t})
  add_executable(${name} ${t}.cpp)
  target_link_libraries(${name} PRIVATE mimetika::mimetika)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
