cmake_minimum_required (VERSION 3.5)

project(mylib)

add_library(mylib STATIC src/mylib.c)

target_include_directories(mylib PRIVATE include)
target_include_directories(mylib PRIVATE src)

set (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

install(TARGETS mylib
    LIBRARY DESTINATION ${LIBRARY_OUTPUT_PATH}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})