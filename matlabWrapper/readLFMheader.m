% readLFMheader Read image header from Light field File Format (.lfm)
% Usage:    header = readLFMheader(filename)
% 
% INPUT:
% 
% filename: string with the path to the lfm file to be read.
% 
% OUTPUT:
% 
% header: struct containing all the information about the file. This struct has the following fields (KLB_DATA_DIMS = 5) :
% 
%       Nnum;     //number of virtual pixels (either in x or y direction) under each microlens.
%       xyzct[KLB_DATA_DIMS];     //image dimensions in pixels
%       pixelSize[KLB_DATA_DIMS];     //pixel size (in um,au,secs) for each dimension
%       dataType;     //look switch statement at readKLBstack.cpp for specific datails. Each number specifies a possible datat type (uint8, uint16, etc
%       compressionType; //lookup table for compression type (0=none; 1=bzip2);
%       blockSize[KLB_DATA_DIMS];     //block size along each dimension to partition the data for bzip. The total size of each block should be ~1MB
%       char metadata[KLB_METADATA_SIZE]; //wildcard to store any data you want
%       uint8 headerVersion; //highest bit: 0,image 1,video    0~6bit: predictor
    
function header = readLFMheader(filename)