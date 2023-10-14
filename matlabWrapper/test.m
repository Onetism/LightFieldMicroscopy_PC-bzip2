
clear;

numThreads = -1;
pixelSize = [1,1,1];
compressionType = 1;

filepath = './img.tif';
sourceImage = uint16(imread(filepath));     
blockSize = [size(sourceImage,1) size(sourceImage,2)];

outname = './imgLFM.lfm';
writeLFMstack(sourceImage, outname ,1, pixelSize, blockSize, compressionType, 'test', 7, 13, 0);

fileheader = readLFMheader(outname);
read_data = readLFMstack(outname, numThreads);
if(find(read_data ~= sourceImage))
    sprintf('read data is not equal source!');
end 




