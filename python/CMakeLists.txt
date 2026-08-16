include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG stable)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core bindings.cpp)
target_link_libraries(_core PRIVATE mimetika::mimetika)
if(MIMETIKA_USE_PETSC)
  target_link_libraries(_core PRIVATE mimetika_petsc)
endif()
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mimetika_cxx)

if(DEFINED SKBUILD)
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

# The pytest suite is the Python-side mirror of tests/model: the same problems,
# the same closed forms, the same tolerances, driven through the bindings.
if(BUILD_TESTING)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  foreach(t test_confined_compression test_cauchy_elasticity test_poroelasticity test_single_phase)
    add_test(NAME python.${t}
             COMMAND ${Python_EXECUTABLE} -m pytest -q
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixed_dimensional/${t}.py)
    set_tests_properties(python.${t} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}")
  endforeach()
endif()
