cmake_minimum_required(VERSION 3.20)
project(mimetika VERSION 0.1.0 LANGUAGES CXX)

# mimetika is the APPLICATION. exokal supplies the discrete operators,
# the local automatic differentiation and the assembly; graphos supplies
# the topology beneath it. Everything mimetika adds is physics: models,
# their closures, and the drivers that run them.
if(NOT TARGET exokal::exokal)
  find_package(exokal CONFIG QUIET)
  if(NOT exokal_FOUND)
    set(MIMETIKA_EXOKAL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../exokal"
        CACHE PATH "Path to an exokal source checkout")
    if(NOT EXISTS ${MIMETIKA_EXOKAL_SOURCE}/CMakeLists.txt)
      message(FATAL_ERROR "exokal not found: install it or set MIMETIKA_EXOKAL_SOURCE")
    endif()
    add_subdirectory(${MIMETIKA_EXOKAL_SOURCE} exokal EXCLUDE_FROM_ALL)
  endif()
endif()

add_library(mimetika INTERFACE)
add_library(mimetika::mimetika ALIAS mimetika)
target_include_directories(mimetika INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(mimetika INTERFACE cxx_std_20)
target_link_libraries(mimetika INTERFACE exokal::exokal)

# PETSC, for the linear algebra. Optional so the operators can be built and
# compared without it — the port's checkpoint is that the assembled system is
# the same, and that needs no solver. Found through pkg-config, which is what
# every PETSc installation provides however it was built.
option(MIMETIKA_USE_PETSC "Solve with PETSc" ON)
if(MIMETIKA_USE_PETSC)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PETSC IMPORTED_TARGET PETSc)
  endif()
  if(PETSC_FOUND)
    message(STATUS "mimetika: PETSc ${PETSC_VERSION}")
    add_library(mimetika_petsc INTERFACE)
    target_link_libraries(mimetika_petsc INTERFACE PkgConfig::PETSC)
    target_compile_definitions(mimetika_petsc INTERFACE MIMETIKA_HAVE_PETSC=1)
    # PETSc's own pkg-config advertises only -lpetsc, but PETSC_COMM_SELF
    # expands to an MPI symbol, so the MPI it was built against has to be on
    # the link line too. Every PETSc is an MPI build even when it will only
    # ever run on one rank.
    pkg_check_modules(OMPI IMPORTED_TARGET ompi-c)
    if(OMPI_FOUND)
      target_link_libraries(mimetika_petsc INTERFACE PkgConfig::OMPI)
      message(STATUS "mimetika: MPI ${OMPI_VERSION}")
    else()
      find_package(MPI QUIET COMPONENTS C)
      if(MPI_C_FOUND)
        target_link_libraries(mimetika_petsc INTERFACE MPI::MPI_C)
      endif()
    endif()
  else()
    message(STATUS "mimetika: PETSc not found; the solver is not built")
    set(MIMETIKA_USE_PETSC OFF)
  endif()
endif()

option(MIMETIKA_BUILD_TESTS "Build mimetika tests" ON)
if(MIMETIKA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(apps)
