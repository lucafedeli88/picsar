
set(name amrex_example)

add_executable(${name} "${name}.cpp")

# Move AMReX example in amrex_example subdirectory
set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/amrex_example)

# Require C++14 or newer
target_compile_features(${name} PUBLIC cxx_std_14)
set_target_properties(${name} PROPERTIES CXX_EXTENSIONS OFF)

# Enable warnings
if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic)
endif()
