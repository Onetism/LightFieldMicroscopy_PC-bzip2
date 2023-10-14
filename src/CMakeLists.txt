
# cmake_policy( SET CMP0104 OLD )

add_subdirectory(external)

file(GLOB SRCS *.cpp *.cu)
file(GLOB HDRS *.h)

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)

set(tgt lfm_static)
add_library(${tgt} STATIC ${SRCS} ${HDRS})
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})
target_link_libraries(${tgt} Threads::Threads bzip2 zlibstatic cudart_static OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${TIFF_LIBRARIES})

set(tgt lfm)
if(${MSVC})
    # MSVC_VERSION is only defined, and thus the expresion only resolves, 
    # if MSVC is defined.
    if(${MSVC_VERSION} EQUAL 1700)
        add_definitions(-D_VARIADIC_MAX=10)
    endif()
endif()
add_library(${tgt} SHARED ${SRCS} ${HDRS})
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR}) 
target_compile_definitions(${tgt} PRIVATE COMPILE_SHARED_LIBRARY)
target_link_libraries(${tgt} Threads::Threads bzip2 zlibstatic cudart ${OpenCV_LIBS} ${TIFF_LIBRARIES})

option(BINDINGS_JAVA "Build bindings for Java" ON)
if(BINDINGS_JAVA)
    add_subdirectory(jni)
endif()

# Only setup install targets if lfm is the project being built.
# Projects that use lfm via add_subdirectory() probably don't
# want the install targets.  They can always set that up themselves.
if(${CMAKE_PROJECT_NAME} STREQUAL lfm)
    install(FILES ${HDRS} DESTINATION include)
    install(TARGETS lfm RUNTIME  DESTINATION lib)
    install(TARGETS lfm_static ARCHIVE DESTINATION lib)
else()
    message("Not installing lfm libraries. CMAKE_PROJECT_NAME is ${CMAKE_PROJECT_NAME}")
endif()
