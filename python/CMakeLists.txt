include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG stable)
FetchContent_MakeAvailable(pybind11)

# _core IS THE PETSc MODULE. bindings.cpp includes linear_solver/petsc.hpp, so
# it does not build without PETSc -- a tree configured for the hypre path alone
# (MIMETIKA_USE_PETSC=OFF) must not try, which it did until this guard.
if(MIMETIKA_USE_PETSC)
  pybind11_add_module(_core bindings.cpp)
  target_link_libraries(_core PRIVATE mimetika::mimetika mimetika_petsc)
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mimetika_cxx)
endif()

if(DEFINED SKBUILD AND MIMETIKA_USE_PETSC)
  set(_core_rpath "")
  if(EXOKAL_BASIX_ROOT)
    list(APPEND _core_rpath "${EXOKAL_BASIX_ROOT}/lib")
  endif()
  foreach(_p IN LISTS CMAKE_PREFIX_PATH)
    list(APPEND _core_rpath "${_p}/lib")
  endforeach()
  set_target_properties(_core PROPERTIES INSTALL_RPATH "${_core_rpath}")
  install(TARGETS _core DESTINATION mimetika_cxx)
endif()

# The direct hypre path, as its own module.
#
# It CANNOT be part of _core: that links PETSc, PETSc links its own libHYPRE,
# and two copies with the same symbol names resolve by load order. The two are
# kept apart by linkage AND by visibility -- the HYPRE_* and hypre_* names this
# module pulls out of libHYPRE.a are marked unexported, so importing both
# mimetika_cxx and mimetika_hypre in one interpreter cannot interpose one
# library's ADS on the other's call.
if(MIMETIKA_USE_HYPRE)
  pybind11_add_module(_hypre hypre_bindings.cpp)
  target_link_libraries(_hypre PRIVATE mimetika::mimetika mimetika_hypre)
  set_target_properties(_hypre PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mimetika_hypre)
  if(APPLE)
    target_link_options(_hypre PRIVATE
      "LINKER:-unexported_symbol,_HYPRE_*"
      "LINKER:-unexported_symbol,_hypre_*"
      "LINKER:-unexported_symbol,__hypre_*"
      "LINKER:-unexported_symbol,_Euclid_*")
  else()
    target_link_options(_hypre PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  if(DEFINED SKBUILD)
    install(TARGETS _hypre DESTINATION mimetika_hypre)
  endif()
endif()

# The pytest suite is the Python-side mirror of tests/model: the same problems,
# the same closed forms, the same tolerances, driven through the bindings.
if(BUILD_TESTING)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  foreach(t test_confined_compression test_cauchy_mechanics test_poroelasticity test_flow
           test_riesz test_flow_ads test_flow_hypre_ads test_mechanics_hypre_ads
           test_hybrid_patch test_flow_riesz_patch
           test_mechanics_riesz_patch)
    add_test(NAME python.${t}
             COMMAND ${Python_EXECUTABLE} -m pytest -q
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixed_dimensional/${t}.py)
    set_tests_properties(python.${t} PROPERTIES
      # PYTHONNOUSERSITE keeps ~/.local out of the interpreter: a package
      # installed there for another architecture or another build shadows the
      # environment's, and the failure reads as a broken extension rather than
      # as the wrong numpy.
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};PYTHONNOUSERSITE=1")
  endforeach()

  # THE SAME ANSWER ON SEVERAL PROCESSES. The solver distributes its algebra
  # over MPI_COMM_WORLD when there is more than one rank, and the property that
  # matters is that nothing about the answer -- including the iteration count --
  # depends on how many there were. Skipped where there is no mpiexec to launch
  # them with.
  find_program(MIMETIKA_MPIEXEC NAMES mpirun mpiexec)
  if(MIMETIKA_MPIEXEC)
    add_test(NAME python.mpi_layout
             COMMAND ${CMAKE_COMMAND}
                     -DPYTHON=${Python_EXECUTABLE}
                     -DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/tests/parallel/mpi_layout.py
                     -DMPIEXEC=${MIMETIKA_MPIEXEC}
                     -DRANKS=2
                     -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/parallel/run_mpi_layout.cmake)
    set_tests_properties(python.mpi_layout PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};PYTHONNOUSERSITE=1"
      SKIP_REGULAR_EXPRESSION "SKIP:"
      PROCESSORS 2)
  endif()
endif()
