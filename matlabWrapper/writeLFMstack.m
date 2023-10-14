% writeLFMstack save Matlab array (up to 5-dimensional) into  Light field File Format (.lfm). 
% Usage:   writeLFMstack(im, filename, numThreads, pixelSize, blockSize, compressionType, metadata) 
% 
% INPUT (all the optional parameters can be set to [] to use its default value):
% 
% 
% im:     Matlab array to be saved. It can be up to 5-dimensional of any class and any size.
% filename: string with the path to the lfm file to be written
% 
% 
% 
% numThreads (optional): integer scalar parameter to decide how many cores to use. Speed scales linearly with the number of threads. If numThreads < -1 (default) then all the available cores are used
% 
% pixelSize (optional):   float precision array of length 5 to store pixel size (in um,au,secs) along each dimension. Default value is -1.0 to indicate N/A
% 
% blockSize (optional): integer array of length <= 5 to define the block size along each dimension to partition the data. Default value is 64x64x8x1x1. Blocks should be below 1MB since it is a good trade off between speed and compression ratio. You can use the script testOptimalBlockSize.m to find out the best parameter for your data
% 
% compressionType (optional): integer scalar to define compression used to store each block. Right now, 0 = none, 1 = bzip2 (default). You can add your own using the source code of the klb library;
% 
% metadata (optional): char[KLB_METADATA_SIZE] to store any extra information you want
% 
% predictor : If predictor is less than 7, the LFM algorithm automatically
% determines the optimal predictor; otherwise, the predictor is set to
% predictor - 7, with a maximum of 7 possible predictor. Therefore, the
% maximum value for predictor is 15. default < 7
% 
% Nnum : number of virtual pixels (either in x or y direction) under each microlens.  default is 13
% 
% imageOrVideo : 0:image stack, 1:video stack, default is 0
    
function  writeLFMstack(im, filename, numThreads, pixelSize, blockSize, compressionType, metadata, predictor, Nnum, imageOrVideo) 