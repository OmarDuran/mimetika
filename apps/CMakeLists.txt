add_executable(mimetika-run run.cpp)
target_link_libraries(mimetika-run PRIVATE mimetika::mimetika)

add_executable(mimetika-bench-column bench_column.cpp)
target_link_libraries(mimetika-bench-column PRIVATE mimetika::mimetika)

if(MIMETIKA_USE_PETSC)

  add_executable(mimetika-poroelastic poroelastic.cpp)
  target_link_libraries(mimetika-poroelastic PRIVATE mimetika::mimetika mimetika_petsc)
endif()

# The hypre path, which must NOT link PETSc: both libraries export the same
# HYPRE_* symbols and a binary holding two copies resolves by load order.
if(MIMETIKA_USE_HYPRE)
  add_executable(mimetika-hypre-flow hypre_flow.cpp)
  target_link_libraries(mimetika-hypre-flow PRIVATE mimetika::mimetika mimetika_hypre)
endif()

# Does the conforming complex's Lambda^2 agree with the flow model's flux space?





if(MIMETIKA_USE_HYPRE)
endif()
