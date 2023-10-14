/*
* Copyright (C) 2014  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  mainTest_klnIO.cxx
*
*  Created on: October 1st, 2014
*      Author: Fernando Amat
*
* \brief Test the main klb I/O library
*
*
*/


#include <string>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <random>
#include "tiffio.h"

#include "klb_imageIO.h"
#include "bzlib.h"


using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int main(int argc, const char** argv)
{
	int numThreads = -1;//<= 0 indicates use as many as possible
	KLB_COMPRESSION_TYPE compressionType = KLB_COMPRESSION_TYPE::BZIP2;//1->bzip2; 0->none
	
	std::string filenameOut = string("D:/Thinkbook14 plus/datazebra_blood_flow_17zebra");

	int width, height, frame;
	char *filename = "D:/Thinkbook14 plus/Data/PC-Bzip2/datacompress_test/data/twoColorBloodFlow/zebra_blood_flow_17zebra.tif";
	TIFF *tiff_file = TIFFOpen(filename,"rb");
	TIFFGetField(tiff_file, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tiff_file, TIFFTAG_IMAGELENGTH, &height);
	frame = TIFFNumberOfDirectories(tiff_file);
	frame = 20;

	uint16_t *img = new uint16_t[width*height*frame];	
	int N_size = 0;
	for(int i = 0; i< frame; i++)
	{
		for (int row = 0; row < height; row++)
		{
			TIFFReadScanline(tiff_file, (img + N_size + row*width), row);
		}
		N_size += width*height;
		TIFFReadDirectory(tiff_file);	
	}

	std::uint32_t	xyzct[KLB_DATA_DIMS] = { width, height, frame, 1, 1 };
	std::uint32_t	blockSize[KLB_DATA_DIMS] = { width, height, frame , 1, 1 };

	int sizeX = xyzct[0], sizeY = xyzct[1], sizeZ = xyzct[2];

	filenameOut += ".lfm";

	//================================================
	//common definitions
	int err;
	auto t1 = Clock::now();
	auto t2 = Clock::now();
	uint16_t* imgB;
	//========================================================================

	klb_imageIO imgIO( filenameOut );

	//setup header
	float32_t pixelSize_[KLB_DATA_DIMS];
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		pixelSize_[ii] = 1.2f*(ii + 1);

	char metadata_[KLB_METADATA_SIZE];
	sprintf(metadata_, "Testing metadata");

	imgIO.header.setHeader(xyzct, KLB_DATA_TYPE::UINT16_TYPE, pixelSize_, blockSize, compressionType, metadata_);
	memcpy(imgIO.header.xyzct, xyzct, sizeof(uint32_t)* KLB_DATA_DIMS);
	memcpy(imgIO.header.blockSize, blockSize, sizeof(uint32_t)* KLB_DATA_DIMS);
	imgIO.header.dataType = KLB_DATA_TYPE::UINT16_TYPE;//uint16
	imgIO.header.compressionType = compressionType;
	imgIO.header.headerVersion = imgIO.header.headerVersion | (1<<7);  // 1: video stack  0:image stack
	
	
	cout << "Compressing file to " << filenameOut << endl;

	for (int aa = 0; aa < 1; aa++)
	{

		t1 = Clock::now();
		err = imgIO.writeImage((char*)img, numThreads);//all the threads available
		if (err > 0)
			return 2;

		t2 = Clock::now();
		std::cout << "Written test file at " << filenameOut << " compress + write file =" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms using " << numThreads << " threads" << std::endl;

	}

	//===========================================================================================
	
	cout << endl << endl << "Reading entire image back" << endl;
	
	
	t1 = Clock::now();
	klb_imageIO imgFull(filenameOut);

	err = imgFull.readHeader();
	if (err > 0)
		return err;
	uint64_t N = imgFull.header.getImageSizePixels();
	uint16_t* imgA = new uint16_t[N];

	err = imgFull.readImageFull((char*)imgA, numThreads);
	if (err > 0)
		return err;

	t2 = Clock::now();

	std::cout << "Read full test file at " << filenameOut << " in =" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms using " << numThreads << " threads" << std::endl;

	//compare elements
	bool isEqual = true;
	for (uint64_t ii = 0; ii < imgFull.header.getImageSizePixels(); ii++)
	{
		if (imgA[ii] != img[ii])
		{
			cout << "ii = " << ii << ";imgOrig = " << img[ii] << "; imgLFM = " << imgA[ii] << endl;
			isEqual = false;
			break;
		}
	}
	if (!isEqual)
		cout << "ERROR!!!: images are different" << endl;

	
	delete[] imgA;
	//release memory
	delete[] img;

	return 0;
}
