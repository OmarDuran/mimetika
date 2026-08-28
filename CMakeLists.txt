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
    # PETSc's HEADERS MUST BE SEARCHED BEFORE ANY SYSTEM DIRECTORY.
    #
    # An imported target carries its includes as SYSTEM ones, so they arrive as
    # -isystem. PETSc's prefix is a whole environment -- Python, zlib and basix
    # live there too -- so that directory reaches the command line as -isystem
    # whatever PETSc itself is marked as, and it then sits in the same chain as
    # /usr/local/include. A machine with an older Open MPI installed there
    # supplies mpi.h, and PETSc, which checks the version it was configured
    # against, refuses to compile.
    #
    # Adding the directory back as -I does NOT fix it: a compiler drops an -I
    # that duplicates an -isystem. The classification itself has to go, which
    # is what this variable is for -- imported targets then contribute plain
    # -I, searched ahead of every system directory.
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)
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

# HYPRE, reached DIRECTLY rather than through PETSc.
#
# PETSc offers hypre only through PCHYPRE, and that surface is narrower than
# hypre's own: -pc_hypre_ads_amg_theta, -pc_hypre_ads_ams_theta and
# -pc_hypre_ads_ams_cycle_type are registered by PETSc and never queried, so
# the strength threshold of the auxiliary hierarchies -- the one parameter a
# jumping coefficient asks to change -- cannot be set at all. Calling hypre
# directly restores HYPRE_ADSSetAMGOptions and HYPRE_ADSSetAMSOptions.
#
# THE TWO PATHS MUST NOT MEET IN ONE BINARY. PETSc links its own libHYPRE (3.1
# in the environment this was written against) and a second copy with the same
# symbol names resolves by load order, not by intent. So a target linking this
# one must not link mimetika_petsc, and nothing here is added to the mimetika
# interface itself.
option(MIMETIKA_USE_HYPRE "Solve with hypre directly, without PETSc" OFF)
if(MIMETIKA_USE_HYPRE)
  set(MIMETIKA_HYPRE_ROOT "" CACHE PATH "Prefix of a hypre install (lib/ and include/)")
  find_library(MIMETIKA_HYPRE_LIB NAMES HYPRE
               HINTS ${MIMETIKA_HYPRE_ROOT}/lib NO_DEFAULT_PATH)
  find_path(MIMETIKA_HYPRE_INCLUDE NAMES HYPRE_parcsr_ls.h
            HINTS ${MIMETIKA_HYPRE_ROOT}/include NO_DEFAULT_PATH)
  if(MIMETIKA_HYPRE_LIB AND MIMETIKA_HYPRE_INCLUDE)
    file(STRINGS ${MIMETIKA_HYPRE_INCLUDE}/HYPRE_config.h MIMETIKA_HYPRE_VER
         REGEX "HYPRE_RELEASE_VERSION")
    message(STATUS "mimetika: hypre ${MIMETIKA_HYPRE_LIB} (${MIMETIKA_HYPRE_VER})")
    add_library(mimetika_hypre INTERFACE)
    target_link_libraries(mimetika_hypre INTERFACE ${MIMETIKA_HYPRE_LIB})
    target_include_directories(mimetika_hypre INTERFACE ${MIMETIKA_HYPRE_INCLUDE})
    target_compile_definitions(mimetika_hypre INTERFACE MIMETIKA_HAVE_HYPRE=1)
    # the SAME MPI PETSc was built against, or the two disagree on MPI_Comm
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(OMPI_H IMPORTED_TARGET ompi-c)
    endif()
    if(OMPI_H_FOUND)
      target_link_libraries(mimetika_hypre INTERFACE PkgConfig::OMPI_H)
    else()
      find_package(MPI QUIET COMPONENTS C)
      if(MPI_C_FOUND)
        target_link_libraries(mimetika_hypre INTERFACE MPI::MPI_C)
      else()
        message(FATAL_ERROR "mimetika: hypre needs MPI and none was found")
      endif()
    endif()
  else()
    message(STATUS "mimetika: hypre not found under '${MIMETIKA_HYPRE_ROOT}'; set MIMETIKA_HYPRE_ROOT")
    set(MIMETIKA_USE_HYPRE OFF)
  endif()
endif()

option(MIMETIKA_BUILD_TESTS "Build mimetika tests" ON)
if(MIMETIKA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# The Python interface to this application. Off by default: it fetches
# pybind11, and the C++ build must not need a network to succeed.
option(MIMETIKA_BUILD_PYTHON "Build the Python bindings" OFF)
if(MIMETIKA_BUILD_PYTHON)
  add_subdirectory(python)
endif()

add_subdirectory(apps)
