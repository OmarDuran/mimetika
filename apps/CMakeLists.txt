add_executable(mimetika-run run.cpp)
target_link_libraries(mimetika-run PRIVATE mimetika::mimetika)

add_executable(mimetika-export export_system.cpp)
target_link_libraries(mimetika-export PRIVATE mimetika::mimetika)
