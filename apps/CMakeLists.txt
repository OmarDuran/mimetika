add_executable(mimetika-run run.cpp)
target_link_libraries(mimetika-run PRIVATE mimetika::mimetika)

add_executable(mimetika-bench-column bench_column.cpp)
target_link_libraries(mimetika-bench-column PRIVATE mimetika::mimetika)

if(MIMETIKA_USE_PETSC)

  add_executable(mimetika-poroelastic poroelastic.cpp)
  target_link_libraries(mimetika-poroelastic PRIVATE mimetika::mimetika mimetika_petsc)
endif()
