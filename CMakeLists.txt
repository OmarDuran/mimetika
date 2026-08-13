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

option(MIMETIKA_BUILD_TESTS "Build mimetika tests" ON)
if(MIMETIKA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(apps)
