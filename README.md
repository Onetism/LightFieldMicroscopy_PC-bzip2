# Light field File Format (.lfm) C++11 source code and API  #

The LFM is a file format developed for efficiently storing light-field microscopy images using lossless compression. The purpose of this format is to developed a 4D phase-space continuity enhanced bzip2 lossless compression method to realize high-speed and efficient LFM data compression. By adding a suitable predictor determined by two-dimensional image entropy criterion before bzip2 or KLB com-pression, it can achieve almost 10% improvement in compression ratio with a little increase in time.

All code has been developed using standard C++11 which makes it easy to compile LFM code across platforms. Moreover, a simple API allows wrapping the open-source C++ code with other languages such as Java, Fiji or Matlab. The LFM format also allows future extensions, such as inclusion of new compression formats. 

Note：Please make sure your computer is equipped with an NVIDIA graphics card and has CUDA installed.

## LFM installation and compilation ##

This software package contains the C++11 source code for the LFM file format implementation as well as wrappers for Matlab and Java. The code has been tested on various Windows systems. However, Linux and Mac OS users need to compile both the source code and the Matlab wrappers to obtain libraries and executables. For the first part, a CMake file is available in the folder *src*. For the second part, the folder *matlabWrapper* contains the script compileMex.m for generating MEX files. The C++ libraries need to be compiled in release mode before compiling the MEX files. It's important to note that, before compiling, CUDA-related libraries, such as cudart.lib in Windows,  need to be copied to the the folder *matlabWrapper* .


## LFM install to ImageJ  ##

The LFM API is exposed on the Java side through a JNI wrapper, included in the *javaWrapper* subfolder. It can be built with Maven, includes compiled native libraries for Windows and Linux (both 64-bit) and will eventually be available as an artifact on a Maven repository. The C++ libraries need to be compiled in release and installed before builting with Maven.

1) Install Maven

2) Navigate to the *javaWrapper* subfolder

3) Run "mvn clean package"

4) JAR file will be built at "javaWrapper/target/lfm-[version].jar"

Please note that it's possible that the required packages may not be found in the default Maven repository. You may need to add the Maven repository configuration for http://maven.imagej.net/content/repositories/public.

## LFM header format ##

The LFM header contains the following items stored in binary format:

- uint8 headerVersion: LFM header information, The highest bit represents whether it's an image stack or a video stack,a nd the remaining bits represent different prediction modes. (uint8)

- uint8 Nnum: number of virtual pixels (either in x or y direction) under each microlens (uint8)

- uint32 xyzct[5]: image dimensions (x, y, z, channels, time points)

- float32 pixelSize[5]: sampling of each dimension (in units of µm, index count, seconds)

- uint8 dataType: look-up-table for data type (uint8, uint16, etc.)

- uint8 compressionType: look-up-table for compression type (none, pbzip2, etc.)

- char metadata[256]: character block providing space for user-defined metadata

- uint32 blockSize[5]: block size used to partition the data in each dimension

- uint64 blockOffset[Nb]: offset (in bytes) of each block in the file

