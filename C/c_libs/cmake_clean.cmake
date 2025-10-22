file(GLOB FILES "${CMAKE_CURRENT_BINARY_DIR}/*")
foreach(FILE ${FILES})
    file(REMOVE_RECURSE ${FILE})
endforeach()