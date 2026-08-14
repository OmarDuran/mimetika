add_executable(mimetika-run run.cpp)
target_link_libraries(mimetika-run PRIVATE mimetika::mimetika)

add_executable(mimetika-export export_system.cpp)
target_link_libraries(mimetika-export PRIVATE mimetika::mimetika)

add_executable(mimetika-bench-column bench_column.cpp)
target_link_libraries(mimetika-bench-column PRIVATE mimetika::mimetika)

add_executable(mimetika-bench-poro bench_poro.cpp)
target_link_libraries(mimetika-bench-poro PRIVATE mimetika::mimetika)

add_executable(mimetika-export-column export_column.cpp)
target_link_libraries(mimetika-export-column PRIVATE mimetika::mimetika)

if(MIMETIKA_USE_PETSC)
  add_executable(mimetika-confined confined.cpp)
  target_link_libraries(mimetika-confined PRIVATE mimetika::mimetika mimetika_petsc)

  add_executable(mimetika-poroelastic poroelastic.cpp)
  target_link_libraries(mimetika-poroelastic PRIVATE mimetika::mimetika mimetika_petsc)

  add_executable(mimetika-terzaghi terzaghi.cpp)
  target_link_libraries(mimetika-terzaghi PRIVATE mimetika::mimetika mimetika_petsc)
endif()
