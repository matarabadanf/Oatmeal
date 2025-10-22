cmake_minimum_required(VERSION 3.10)
project(c_libs C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall")

# Find all .c files in the directory
file(GLOB SRC_FILES *.c)

# Build one shared library per .c file
foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(LIB_NAME ${SRC_FILE} NAME_WE)
    add_library(${LIB_NAME} SHARED ${SRC_FILE})
    set_target_properties(${LIB_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()

# Add a custom clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake_clean.cmake
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_SOURCE_DIR}/*.so
    COMMENT "Cleaning all build files and shared libraries"
)
