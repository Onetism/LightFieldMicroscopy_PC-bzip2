
%debugging
% mex -g -output writelfmstack  writelfmstack.cpp -I../src ../build/Debug/lfm_static.lib ../build/external/bzip2-1.0.6/Debug/bzip2.lib ../build/external/zlib-1.2.8/Debug/zlibstaticd.lib -DWINDOWS

mex -O -output writeLFMstack  writeLFMstack.cpp -I../src ../build/src/Release/lfm_static.lib ../build/src/external/bzip2-1.0.6/Release/bzip2.lib ../build/src/external/zlib-1.2.8/Release/zlibstatic.lib  ./cudart_static.lib -DWINDOWS -D_FILE_OFFSET_BITS=64

mex -O -output readLFMheader  readLFMheader.cpp -I../src ../build/src/Release/lfm_static.lib ../build/src/external/bzip2-1.0.6/Release/bzip2.lib ../build/src/external/zlib-1.2.8/Release/zlibstatic.lib ./cudart_static.lib -DWINDOWS -D_FILE_OFFSET_BITS=64

mex -O -output readLFMstack  readLFMstack.cpp -I../src ../build/src/Release/lfm_static.lib ../build/src/external/bzip2-1.0.6/Release/bzip2.lib ../build/src/external/zlib-1.2.8/Release/zlibstatic.lib ./cudart_static.lib -DWINDOWS -D_FILE_OFFSET_BITS=64
 
