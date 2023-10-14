# cmake_policy( SET CMP0104 OLD )

find_package(Threads REQUIRED)
find_package(TIFF REQUIRED)

#executable to test lfmlib library for I/O purposes
set(tgt test_lfmIO)
add_executable(${tgt} mainTest_lfmIO.cxx)
target_link_libraries (${tgt} lfm_static ${TIFF_LIBRARIES})
target_include_directories(${tgt} PUBLIC ${TIFF_INCLUDE_DIR})
set_target_properties(${tgt} PROPERTIES FOLDER test)
add_test(test-${tgt} ${tgt})

