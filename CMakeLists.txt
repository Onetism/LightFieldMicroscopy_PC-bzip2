project(lfm  LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.0)

include(cmake/git-versioning.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_STANDARD   99)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-w)
add_subdirectory(src)

set(build_tests ON)
if (build_tests)
    enable_testing()
    add_subdirectory(test)
endif()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)

include(InstallRequiredSystemLibraries)
include(CPack)