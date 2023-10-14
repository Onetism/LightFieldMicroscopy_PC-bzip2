
cmake_policy( SET CMP0104 OLD )

set(SRCS
	cuda_crctable.c
	cuda_randtable.c
	cuda_compress.c
	cuda_decompress.c
	cuda_bzlib.c
	cuda_blocksort.c
    cuda_huffman.c
	# bzip2.c
	gpuBWTSort.cu)

file(GLOB HDRS *.h)
set_source_files_properties(
	cuda_crctable.c
	cuda_randtable.c													
	cuda_blocksort.c
	cuda_huffman.c 	
	cuda_bzlib.c
	cuda_decompress.c
	# bzip2.c		
	cuda_compress.c
    PROPERTIES LANGUAGE CXX )

set_source_files_properties(gpuBWTSort.cu PROPERTIES LANGUAGE CUDA )

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(tgt cuda_bzip2)
add_library(${tgt} STATIC ${SRCS} ${HDRS})
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries (${tgt} ${LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX)

# add_executable(bzip2_excute bzip2.c)
# target_link_libraries (bzip2_excute ${LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX cuda_bzip2)
# set_target_properties(${tgt} PROPERTIES SEPARABLE_COMPILATION ON)

# set(tgt bzip2s)
# add_library(${tgt} SHARED ${SRCS} )
# target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

#add_executable(bzip2 bzip2.c ${SRCS})

#add_executable(bzip2recover bzip2recover.c)
#target_link_libraries(bzip2recover bz2)
