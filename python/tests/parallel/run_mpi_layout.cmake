# Solve once on one process, then again under MPI, and compare.
#
# Two runs are the test: a reference the same build produced sequentially, and
# the same problems distributed. Neither number is written down here -- what is
# asserted is that they agree, which is the property that survives changing the
# mesh, the tolerance or the machine.
#
# Invoked by ctest with -DPYTHON, -DSCRIPT, -DMPIEXEC, -DRANKS, -DWORKDIR.

set(reference "${WORKDIR}/mpi_layout_serial.json")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env PYTHONNOUSERSITE=1 ${PYTHON} ${SCRIPT} --write ${reference}
  RESULT_VARIABLE serial_result
  OUTPUT_VARIABLE serial_output
  ERROR_VARIABLE serial_output)
if(NOT serial_result EQUAL 0)
  message(FATAL_ERROR "the sequential reference failed:\n${serial_output}")
endif()

# THE LAUNCHER MUST MATCH THE MPI THE EXTENSION LOADS, and on a machine with
# more than one Open MPI installed it often does not -- MPI_Init then fails, or
# worse, the ranks come up as SINGLETONS, each solving the whole problem alone
# and agreeing with the reference perfectly while testing nothing. That is why
# the script is given --expect-ranks.
#
# So the launcher beside the interpreter is tried first, and whatever is on
# PATH after it. If none of them can start the ranks, the environment is broken
# rather than the solver, and the test says SKIP instead of failing.
set(candidates "${MPIEXEC}")
find_program(path_mpiexec NAMES mpirun mpiexec PATHS ENV PATH NO_DEFAULT_PATH)
if(path_mpiexec AND NOT path_mpiexec STREQUAL MPIEXEC)
  list(APPEND candidates "${path_mpiexec}")
endif()

foreach(launcher IN LISTS candidates)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env PYTHONNOUSERSITE=1
            ${launcher} -n ${RANKS} ${PYTHON} ${SCRIPT} --check ${reference}
            --expect-ranks ${RANKS}
    RESULT_VARIABLE mpi_result
    OUTPUT_VARIABLE mpi_output
    ERROR_VARIABLE mpi_output)
  if(mpi_result EQUAL 0)
    message(STATUS "${RANKS} ranks under ${launcher}:\n${mpi_output}")
    return()
  endif()
  if(NOT mpi_output MATCHES "MPI_Init|MPI_INIT|PML add procs|expected ${RANKS}")
    # it launched and the answers differ: that is the failure this test is for
    message(FATAL_ERROR "${RANKS} processes disagree with one:\n${mpi_output}")
  endif()
  message(STATUS "${launcher} cannot launch ${RANKS} working ranks here")
endforeach()

# the test's SKIP_REGULAR_EXPRESSION picks this line up
message(STATUS "SKIP: no MPI launcher on this machine starts ${RANKS} ranks against the "
               "MPI the extension loads")
