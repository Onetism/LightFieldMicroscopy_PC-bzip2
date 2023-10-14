/*
* Copyright (C) 2014 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat
*  klb_imageIO.cpp
*
*  Created on: October 2nd, 2014
*      Author: Fernando Amat
*
* \brief Main class to read/write klb format
*/

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <mutex>
#include <chrono>
#include <stdlib.h>     /* div, div_t */
#include <cstring>
#include <map>
#include "klb_imageIO.h"
#include "bzlib.h"
#include "zlib.h"
#include "lfm_Predictors.h"
#include "lfm_Predictors_space.h"
#include "lfm_Predictors_angle.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tiffio.h"


typedef std::chrono::high_resolution_clock Clock;
#if defined(_WIN32) || defined(_WIN64)
//in windows long int is 32 bit, so fseek cannot read large files
	#define fseek _fseeki64
#endif

//#define DEBUG_PRINT_THREADS
typedef std::chrono::high_resolution_clock Clock;
using namespace std;


#ifdef PROFILE_COMPRESSION
std::atomic <long long> klb_imageIO::g_countCompression;//in case we want to measure only compression timing
#endif

//Round a / b to nearest higher integer value (T should be an integer class)
template<class T>
inline T iDivUp(const T a, const T b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
inline int getNegativeSign(int val)
{
	return (val >> 31);
}

inline uint32_t symbolize_kernel(int value)
{
	// map >= 0 to even, < 0 to odd
	return 2 * abs(value) + getNegativeSign(value);
}

inline int unsymbolize_kernel(uint32_t symbol)
{
	int negative = symbol % 2;
	// map even to >= 0, odd to < 0
	return (1 - 2 * negative) * ((symbol + negative) / 2);
}
//========================================================
//======================================================
void klb_imageIO::blockCompressor(const char* buffer, int* g_blockSize, std::atomic<uint64_t> *blockId, int* g_blockThreadId, klb_circular_dequeue* cq, int threadId, int* errFlag)
{
	*errFlag = 0;
	int BWTblockSize = 9;//maximum compression
	std::uint64_t blockId_t;
	int gcount;//read bytes
	unsigned int sizeCompressed;//size of block in bytes after compression
	

	const size_t bytesPerPixel = header.getBytesPerPixel();
	uint32_t blockSizeBytes = bytesPerPixel;
	uint32_t maxBlockSizeBytesCompressed = maximumBlockSizeCompressedInBytes();
	uint64_t fLength = bytesPerPixel;
	uint64_t dimsBlock[KLB_DATA_DIMS];//number of blocks on each dimension
	uint64_t coordBlock[KLB_DATA_DIMS];//coordinates (in block space). blockId_t = coordBblock[0] + dimsBblock[0] * coordBlock[1] + dimsBblock[0] * dimsBblock[1] * coordBlock[2] + ...
	uint64_t offsetBuffer;//starting offset for each buffer
	uint32_t blockSizeAux[KLB_DATA_DIMS];//for border cases where the blocksize might be different
	uint64_t xyzctCum[KLB_DATA_DIMS];//to calculate offsets for each dimension

	xyzctCum[0] = bytesPerPixel;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
	{
		blockSizeBytes *= header.blockSize[ii];
		fLength *= header.xyzct[ii];
		dimsBlock[ii] = ceil((float)(header.xyzct[ii]) / (float)(header.blockSize[ii]));
		if (ii > 0)
			xyzctCum[ii] = xyzctCum[ii - 1] * header.xyzct[ii - 1];
	}
	char* bufferIn = new char[blockSizeBytes];
	
	BWTblockSize = std::min( BWTblockSize, iDivUp ((int)blockSizeBytes , (int)100000) );//packages of 100,000 bytes
	
	

	std::uint64_t numBlocks = header.getNumBlocks();


	//main loop to keep processing blocks while they are available
	while (1)
	{
		blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);

		//check if we can access data or we cannot read longer
		if (blockId_t >= numBlocks)
			break;


#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d reading block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//-------------------read block-----------------------------------

		//calculate coordinate (in block space)
		std::uint64_t blockIdx_aux = blockId_t;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			coordBlock[ii] = blockIdx_aux % dimsBlock[ii];
			blockIdx_aux -= coordBlock[ii];
			blockIdx_aux /= dimsBlock[ii];
			coordBlock[ii] *= header.blockSize[ii];//parsing coordinates to image space (not block anymore)
		}

		//make sure it is not a border block
		gcount = bytesPerPixel;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			blockSizeAux[ii] = std::min(header.blockSize[ii], (uint32_t)(header.xyzct[ii] - coordBlock[ii]));
			gcount *= blockSizeAux[ii]; 
		}

		//calculate starting offset in the buffer
		offsetBuffer = 0;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			offsetBuffer += coordBlock[ii] * xyzctCum[ii];
		}		

		//copy block into local buffer bufferIn		
		uint32_t bcount[KLB_DATA_DIMS];//to count elements in the block
		memset(bcount, 0, sizeof(uint32_t)* KLB_DATA_DIMS);
		const size_t  bufferCopySize = bytesPerPixel * blockSizeAux[0];
		char* bufferInAux = bufferIn;
		int auxDim = 1;
		while (auxDim < KLB_DATA_DIMS)
		{
			//copy fastest moving coordinate all at once for efficiency			
			memcpy(bufferInAux, &(buffer[offsetBuffer]), bufferCopySize);
			bufferInAux += bufferCopySize;
			

			//increment counter			
			bcount[1]++;
			offsetBuffer += xyzctCum[1];//update offset 
			auxDim = 1;
			while (bcount[auxDim] == blockSizeAux[auxDim])
			{
				offsetBuffer -= bcount[auxDim] * xyzctCum[auxDim];//update buffer
				bcount[auxDim++] = 0;
				if (auxDim == KLB_DATA_DIMS)
					break;
				bcount[auxDim]++;
				offsetBuffer += xyzctCum[auxDim]; //update buffer
			}
		}

		//-------------------end of read block-----------------------------------


#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d compressor block check point 1 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//decide address where we write the compressed block output						
		char* bufferOutPtr = cq->getWriteBlock(); //this operation is thread safe	



#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d compressor block check point 2 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

#ifdef PROFILE_COMPRESSION
		auto t1 = Clock::now();
#endif
		//apply compression to block
		switch (header.compressionType)
		{
		case KLB_COMPRESSION_TYPE::NONE://no compression
			sizeCompressed = gcount;
			memcpy(bufferOutPtr, bufferIn, sizeCompressed);//fid.gcount():Returns the number of characters extracted by the last unformatted input operation performed on the object
			break;
		case KLB_COMPRESSION_TYPE::BZIP2://bzip2
		{
											 sizeCompressed = maxBlockSizeBytesCompressed;
											 // compress the memory buffer (blocksize=9*100k, verbose=0, worklevel=30)				  
											 int ret = BZ2_bzBuffToBuffCompress(bufferOutPtr, &sizeCompressed, bufferIn, gcount, BWTblockSize, 0, 30);
											 if (ret != BZ_OK)
											 {
												 std::cout << "ERROR: workerfunc: compressing data at block " << blockId_t << " with bzip2. Error code " << ret << std::endl;
												 *errFlag = 2;
												 sizeCompressed = 0;
											 }
											 break;
		}
		case KLB_COMPRESSION_TYPE::ZLIB:
		{									
										   z_stream strm;
										   strm.zalloc = Z_NULL;
										   strm.zfree = Z_NULL;
										   strm.opaque = Z_NULL;
										   //which is an integer in the range of - 1 to 9. Lower compression levels result in faster execution, but less compression.Higher levels result in greater compression, but slower execution.The zlib constant Z_DEFAULT_COMPRESSION, equal to - 1, provides a good compromise between compression and speed and is equivalent to level 6. Level 0 actually does no compression at all
										   *errFlag = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
										   

										   strm.avail_in = gcount;
										   strm.next_in = (Bytef*)bufferIn;
										   strm.avail_out = maxBlockSizeBytesCompressed;
										   strm.next_out = (Bytef*)bufferOutPtr;
										   strm.data_type = Z_BINARY;//data type
										   
										   int ret = deflate(&strm, Z_FINISH);
										   sizeCompressed = maxBlockSizeBytesCompressed - strm.avail_out;
										   if (ret != Z_STREAM_END && ret != Z_OK)
										   {
											   std::cout << "ERROR: workerfunc: compressing data at block " << blockId_t << " with zlib. Error code " << ret << std::endl;
											   *errFlag = 3;
											   sizeCompressed = 0;
										   }
										   //release strm
										   (void)deflateEnd(&strm);
										   break;
		}
		default:
			std::cout << "ERROR: workerfunc: compression type not implemented" << std::endl;
			*errFlag = 5;
			sizeCompressed = 0;
		}

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d compressor block check point 3 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

#ifdef PROFILE_COMPRESSION
		auto t2 = Clock::now();
		long long auxChrono = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();//in ms
		atomic_fetch_add(&g_countCompression, auxChrono);
#endif

		cq->pushWriteBlock();//notify content is ready in the queue

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d compressor block check point 4 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//signal blockWriter that this block can be writen
		std::unique_lock<std::mutex> locker(g_lockqueue);//adquires the lock		
		g_blockSize[blockId_t] = sizeCompressed;//I don't really need the lock to modify this. I only need to singal the condition variable
		g_blockThreadId[blockId_t] = threadId;
		locker.unlock();

		g_queuecheck.notify_all();

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d finished compressing block %d into %d bytes\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int) sizeCompressed);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
	}
	
	//release memory
	delete[] bufferIn;

}
//======================================================
//this is special case: I know buffer is 3D data. buffer[ii] is the ii-th 2D slice
void klb_imageIO::blockCompressorStackSlices(const char** buffer, int* g_blockSize, std::atomic<uint64_t> *blockId, int* g_blockThreadId, klb_circular_dequeue* cq, int threadId, int* errFlag)
{

	const int dataDims = 3;//this is special case: I know buffer is 3D data. buffer[ii] is the ii-th 2D slice

	*errFlag = 0;
	int BWTblockSize = 9;//maximum compression
	std::uint64_t blockId_t;
	int gcount;//read bytes
	unsigned int sizeCompressed;//size of block in bytes after compression

	const size_t bytesPerPixel = header.getBytesPerPixel();
	uint32_t blockSizeBytes = bytesPerPixel;
	uint32_t maxBlockSizeBytesCompressed = maximumBlockSizeCompressedInBytes();
	uint64_t fLength = bytesPerPixel;
	uint64_t dimsBlock[dataDims];//number of blocks on each dimension
	uint64_t coordBlock[dataDims];//coordinates (in block space). blockId_t = coordBblock[0] + dimsBblock[0] * coordBlock[1] + dimsBblock[0] * dimsBblock[1] * coordBlock[2] + ...
	uint64_t offsetBuffer;//starting offset for each buffer
	uint64_t offsetZ, offsetSlice;//starting offset in terms of Z + slice
	uint32_t blockSizeAux[dataDims];//for border cases where the blocksize might be different
	uint64_t xyzctCum[dataDims];//to calculate offsets for each dimension

	xyzctCum[0] = bytesPerPixel;
	for (int ii = 0; ii < dataDims; ii++)
	{
		blockSizeBytes *= header.blockSize[ii];
		fLength *= header.xyzct[ii];
		dimsBlock[ii] = ceil((float)(header.xyzct[ii]) / (float)(header.blockSize[ii]));
		if (ii > 0)
			xyzctCum[ii] = xyzctCum[ii - 1] * header.xyzct[ii - 1];
	}
	char* bufferIn = new char[blockSizeBytes];

	BWTblockSize = std::min(BWTblockSize, iDivUp((int)blockSizeBytes, (int)100000));//packages of 100,000 bytes



	std::uint64_t numBlocks = header.getNumBlocks();
	const std::uint64_t sliceSizeBytes = header.xyzct[0] * header.xyzct[1] * bytesPerPixel;


	//main loop to keep processing blocks while they are available
	while (1)
	{
		blockId_t = atomic_fetch_add(blockId, (uint64_t)1);

		//check if we can access data or we cannot read longer
		if (blockId_t >= numBlocks)
			break;


#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d reading block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//-------------------read block-----------------------------------

		//calculate coordinate (in block space)
		std::uint64_t blockIdx_aux = blockId_t;
		for (int ii = 0; ii < dataDims; ii++)
		{
			coordBlock[ii] = blockIdx_aux % dimsBlock[ii];
			blockIdx_aux -= coordBlock[ii];
			blockIdx_aux /= dimsBlock[ii];
			coordBlock[ii] *= header.blockSize[ii];//parsing coordinates to image space (not block anymore)
		}

		//make sure it is not a border block
		gcount = bytesPerPixel;
		for (int ii = 0; ii < dataDims; ii++)
		{
			blockSizeAux[ii] = std::min(header.blockSize[ii], (uint32_t)(header.xyzct[ii] - coordBlock[ii]));
			gcount *= blockSizeAux[ii];
		}

		//calculate starting offset in the buffer
		offsetBuffer = 0;
		for (int ii = 0; ii < dataDims; ii++)
		{
			offsetBuffer += coordBlock[ii] * xyzctCum[ii];
		}

		//copy block into local buffer bufferIn		
		uint32_t bcount[dataDims];//to count elements in the block
		memset(bcount, 0, sizeof(uint32_t)* dataDims);
		const size_t  bufferCopySize = bytesPerPixel * blockSizeAux[0];
		char* bufferInAux = bufferIn;
		int auxDim = 1;
		while (auxDim < dataDims)
		{
			//copy fastest moving coordinate all at once for efficiency		
			offsetZ = offsetBuffer / sliceSizeBytes;
			offsetSlice = offsetBuffer - offsetZ * sliceSizeBytes;
			memcpy(bufferInAux, &(buffer[offsetZ][offsetSlice]), bufferCopySize);
			bufferInAux += bufferCopySize;


			//increment counter			
			bcount[1]++;
			offsetBuffer += xyzctCum[1];//update offset 
			auxDim = 1;
			while (bcount[auxDim] == blockSizeAux[auxDim])
			{
				offsetBuffer -= bcount[auxDim] * xyzctCum[auxDim];//update buffer
				bcount[auxDim++] = 0;
				if (auxDim == dataDims)
					break;
				bcount[auxDim]++;
				offsetBuffer += xyzctCum[auxDim]; //update buffer
			}
		}

		//-------------------end of read block-----------------------------------


#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d uncompressor block check point 1 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//decide address where we write the compressed block output						
		char* bufferOutPtr = cq->getWriteBlock(); //this operation is thread safe	



#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d uncompressor block check point 2 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

#ifdef PROFILE_COMPRESSION
		auto t1 = Clock::now();
#endif
		//apply compression to block
		switch (header.compressionType)
		{
		case KLB_COMPRESSION_TYPE::NONE://no compression
			sizeCompressed = gcount;
			memcpy(bufferOutPtr, bufferIn, sizeCompressed);//fid.gcount():Returns the number of characters extracted by the last unformatted input operation performed on the object
			break;
		case KLB_COMPRESSION_TYPE::BZIP2://bzip2
		{
											 sizeCompressed = maxBlockSizeBytesCompressed;
											 // compress the memory buffer (blocksize=9*100k, verbose=0, worklevel=30)				  
											 int ret = BZ2_bzBuffToBuffCompress(bufferOutPtr, &sizeCompressed, bufferIn, gcount, BWTblockSize, 0, 30);
											 if (ret != BZ_OK)
											 {
												 std::cout << "ERROR: workerfunc: compressing data at block " << blockId_t << " with bzip2. Error code " << ret << std::endl;
												 *errFlag = 2;
												 sizeCompressed = 0;
											 }
											 break;
		}
		case KLB_COMPRESSION_TYPE::ZLIB:
		{
										   z_stream strm;
										   strm.zalloc = Z_NULL;
										   strm.zfree = Z_NULL;
										   strm.opaque = Z_NULL;
										   //which is an integer in the range of - 1 to 9. Lower compression levels result in faster execution, but less compression.Higher levels result in greater compression, but slower execution.The zlib constant Z_DEFAULT_COMPRESSION, equal to - 1, provides a good compromise between compression and speed and is equivalent to level 6. Level 0 actually does no compression at all
										   *errFlag = deflateInit(&strm, Z_DEFAULT_COMPRESSION);


										   strm.avail_in = gcount;
										   strm.next_in = (Bytef*)bufferIn;
										   strm.avail_out = maxBlockSizeBytesCompressed;
										   strm.next_out = (Bytef*)bufferOutPtr;
										   strm.data_type = Z_BINARY;//data type

										   int ret = deflate(&strm, Z_FINISH);
										   sizeCompressed = maxBlockSizeBytesCompressed - strm.avail_out;
										   if (ret != Z_STREAM_END && ret != Z_OK)
										   {
											   std::cout << "ERROR: workerfunc: compressing data at block " << blockId_t << " with zlib. Error code " << ret << std::endl;
											   *errFlag = 3;
											   sizeCompressed = 0;
										   }
										   //release strm
										   (void)deflateEnd(&strm);
										   break;
		}
		default:
			std::cout << "ERROR: workerfunc: compression type not implemented" << std::endl;
			*errFlag = 5;
			sizeCompressed = 0;
		}

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d uncompressor block check point 3 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

#ifdef PROFILE_COMPRESSION
		auto t2 = Clock::now();
		long long auxChrono = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();//in ms
		atomic_fetch_add(&g_countCompression, auxChrono);
#endif

		cq->pushWriteBlock();//notify content is ready in the queue

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d uncompressor block check point 4 for block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//signal blockWriter that this block can be writen
		std::unique_lock<std::mutex> locker(g_lockqueue);//adquires the lock		
		g_blockSize[blockId_t] = sizeCompressed;//I don't really need the lock to modify this. I only need to singal the condition variable
		g_blockThreadId[blockId_t] = threadId;
		locker.unlock();

		g_queuecheck.notify_all();

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d finished compressing block %d into %d bytes\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)sizeCompressed);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
	}


	//release memory
	delete[] bufferIn;

}
//======================================================
void klb_imageIO::blockUncompressor(char* bufferOut, std::atomic<uint64_t> *blockId, const klb_ROI* ROI, int *errFlag)
{
	*errFlag = 0;
	//open file to read elements
	FILE* fid = fopen(filename.c_str(), "rb"); 
	if ( fid == NULL )
	{
		cout << "ERROR: blockUncompressor: thread opening file " << filename << endl;
		*errFlag = 3;
		return;
	}

	//define variables
	std::uint64_t blockId_t;//to know which block we are processing
	unsigned int sizeCompressed, gcount;
	std::uint64_t offset;//size of block in bytes after compression
	

	size_t bytesPerPixel = header.getBytesPerPixel();
	uint32_t blockSizeBytes = bytesPerPixel;
	uint64_t fLength = bytesPerPixel;
	uint64_t dimsBlock[KLB_DATA_DIMS];//number of blocks on each dimension
	uint64_t coordBlock[KLB_DATA_DIMS];//coordinates (in block space). blockId_t = coordBblock[0] + dimsBblock[0] * coordBlock[1] + dimsBblock[0] * dimsBblock[1] * coordBlock[2] + ...
	uint64_t offsetBuffer;//starting offset for each buffer within ROI
	uint32_t offsetBufferBlock;////starting offset for each buffer within decompressed block
	uint32_t blockSizeAux[KLB_DATA_DIMS];//for border cases where the blocksize might be different
	uint64_t xyzctCum[KLB_DATA_DIMS];//to calculate offsets for each dimension in THE ROI
	uint64_t offsetHeaderBytes = header.getSizeInBytes();

	xyzctCum[0] = bytesPerPixel;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
	{
		blockSizeBytes *= header.blockSize[ii];
		fLength *= header.xyzct[ii];
		dimsBlock[ii] = ceil((float)(header.xyzct[ii]) / (float)(header.blockSize[ii]));
		if (ii > 0)
			xyzctCum[ii] = xyzctCum[ii - 1] * ROI->getSizePixels(ii-1);
	}

	std::uint64_t numBlocks = header.getNumBlocks();
	char* bufferIn = new char[blockSizeBytes];//temporary storage for decompressed block
	char* bufferFile = new char[blockSizeBytes];//temporary storage for compressed block from file

	//main loop to keep processing blocks while they are available
	while (1)
	{
		//get the blockId resource
		blockId_t = atomic_fetch_add(blockId, (uint64_t)1);

		//check if we have more blocks
		if (blockId_t >= numBlocks)
			break;

		//calculate coordinate (in block space)
		std::uint64_t blockIdx_aux = blockId_t;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			coordBlock[ii] = blockIdx_aux % dimsBlock[ii];
			blockIdx_aux -= coordBlock[ii];
			blockIdx_aux /= dimsBlock[ii];
			coordBlock[ii] *= header.blockSize[ii];//parsing coordinates to image space (not block anymore)
		}

		//check if ROI and bloock coordinates intersect
		bool intersect = true;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			//for each coordinate, we have to check: RectA.X1 < RectB.X2 && RectA.X2 > RectB.X1, where X1 is minimum nad X2 is maximum coordinate
			//from http://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
			if ( !( (coordBlock[ii] <= ROI->xyzctUB[ii]) && (coordBlock[ii] + header.blockSize[ii] - 1 >= ROI->xyzctLB[ii] ) ))
			{
				intersect = false;
				break;
			}
		}

		if (intersect == false)
			continue;//process another block

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d reading block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		
		//uncompress block into temp bufferIn
		sizeCompressed = header.getBlockCompressedSizeBytes(blockId_t);
		offset = header.getBlockOffset(blockId_t);

		
		fseek(fid, offsetHeaderBytes + offset, SEEK_SET);
		fread(bufferFile, 1, sizeCompressed, fid);//read compressed block

		//apply decompression to block
		switch (header.compressionType)
		{
		case KLB_COMPRESSION_TYPE::NONE://no compression
			gcount = sizeCompressed;
			memcpy(bufferIn, bufferFile, gcount);
			break;
		case KLB_COMPRESSION_TYPE::BZIP2://bzip2
		{
				   gcount = blockSizeBytes;
				   int ret = BZ2_bzBuffToBuffDecompress(bufferIn, &gcount, bufferFile, sizeCompressed, 0, 0);				   
				   if (ret != BZ_OK)
				   {
					   std::cout << "ERROR: workerfunc: uncompressing data at block " << blockId_t << std::endl;
					   *errFlag = 2;
					   gcount = 0;
				   }
				   break;
		}
		case KLB_COMPRESSION_TYPE::ZLIB:
		{
										   z_stream strm;
										   strm.zalloc = Z_NULL;
										   strm.zfree = Z_NULL;
										   strm.opaque = Z_NULL;
										   strm.avail_in = 0;
										   strm.next_in = Z_NULL;
										   *errFlag = inflateInit(&strm);


										   strm.avail_out = blockSizeBytes;
										   strm.next_out = (Bytef*)bufferIn;
										   strm.avail_in = sizeCompressed;
										   strm.next_in = (Bytef*)bufferFile;
										   strm.data_type = Z_BINARY;//data type

										   int ret = inflate(&strm, Z_FINISH);
										   //gcount = sizeCompressed - strm.avail_out;
										   if (ret != Z_STREAM_END && ret != Z_OK)
										   {
											   std::cout << "ERROR: workerfunc: uncompressing data at block " << blockId_t << " with zlib. Error code " << ret << std::endl;
											   *errFlag = 3;
											   gcount = 0;
										   }
										   //release strm
										   (void)deflateEnd(&strm);
										   break;
		}
		default:
			std::cout << "ERROR: workerfunc: decompression type not implemented" << std::endl;
			*errFlag = 5;
			sizeCompressed = 0;
		}



		//-------------------parse bufferIn to bufferOut image buffer-----------------------------------
		//------------------intersection of two ROI (blopck and image ROI) is another ROI, so we just need to calculate the intersection and its offsets

		//calculate block size in case we had border block
		uint32_t blockSizeAuxCum[KLB_DATA_DIMS];
		blockSizeAux[0] = std::min(header.blockSize[0], (uint32_t)(header.xyzct[0] - coordBlock[0]));
		blockSizeAuxCum[0] = bytesPerPixel;
		for (int ii = 1; ii < KLB_DATA_DIMS; ii++)
		{
			blockSizeAux[ii] = std::min(header.blockSize[ii], (uint32_t)(header.xyzct[ii] - coordBlock[ii]));
			blockSizeAuxCum[ii] = blockSizeAuxCum[ii - 1] * blockSizeAux[ii - 1];
		}

		//calculate upper and lower coordinate of the intersection between block and ROI wrt to block dimensions, so it is bounded by [0, blockSizeAux[ii])
		uint32_t bLB[KLB_DATA_DIMS], bUB[KLB_DATA_DIMS], bcount[KLB_DATA_DIMS];
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			if (coordBlock[ii] >= ROI->xyzctLB[ii])
				bLB[ii] = 0;
			else
				bLB[ii] = ROI->xyzctLB[ii] - coordBlock[ii];

			if (coordBlock[ii] + blockSizeAux[ii] - 1 <= ROI->xyzctUB[ii])
				bUB[ii] = blockSizeAux[ii];
			else
				bUB[ii] = ROI->xyzctUB[ii] - coordBlock[ii] + 1;

			blockSizeAux[ii] = bUB[ii] - bLB[ii];//one we have the cum, we do not need the individual dimensions. What we need is the size of the intersection with the ROI
		}
		memcpy(bcount, bLB, sizeof(uint32_t)* KLB_DATA_DIMS);

		//calculate starting offset in the buffer in ROI space
		offsetBuffer = 0;
		offsetBufferBlock = 0;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			offsetBuffer += (coordBlock[ii] + bLB[ii] - ROI->xyzctLB[ii]) * xyzctCum[ii];
			offsetBufferBlock += bLB[ii] * blockSizeAuxCum[ii];
		}		
		

		//copy block into local buffer bufferIn
		int auxDim = 1;
		const size_t bufferCopySize = bytesPerPixel * blockSizeAux[0];
		const size_t bufferInOffset = blockSizeAuxCum[1];
		char* bufferInAux = &(bufferIn[offsetBufferBlock]);
		while (auxDim < KLB_DATA_DIMS)
		{
			//copy fastest moving coordinate all at once for efficiency
			//memcpy(&(bufferOut[offsetBuffer]), &(bufferIn[offsetBufferBlock]), bufferCopySize);
			memcpy(&(bufferOut[offsetBuffer]), bufferInAux, bufferCopySize);
			bufferInAux += bufferInOffset;

			//increment counter			
			bcount[1]++;
			offsetBuffer += xyzctCum[1];//update offset for output buffer		
			offsetBufferBlock += blockSizeAuxCum[1];
			auxDim = 1;

			while (bcount[auxDim] == bUB[auxDim])
			{
				offsetBuffer -= blockSizeAux[auxDim] * xyzctCum[auxDim];//update buffer				
				offsetBufferBlock -= blockSizeAux[auxDim] * blockSizeAuxCum[auxDim];
				bcount[auxDim] = bLB[auxDim];
				auxDim++;
				if (auxDim == KLB_DATA_DIMS)
					break;
				bcount[auxDim]++;
				offsetBuffer += xyzctCum[auxDim]; //update buffer
				offsetBufferBlock += blockSizeAuxCum[auxDim];

				bufferInAux = &(bufferIn[offsetBufferBlock]);//with ROI it is not a constant increment
			}
		}
		//-------------------end of parse bufferIn to bufferOut image buffer-----------------------------------

		

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d finished decompressing block %d into %d bytes\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)gcount);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
	}


	//release memory
	fclose(fid);
	delete[] bufferIn;
	delete[] bufferFile;

}

//======================================================
void klb_imageIO::blockUncompressorInMem(char* bufferOut, std::atomic<uint64_t>	*blockId, char* bufferImgFull, int *errFlag)
{
	*errFlag = 0;
	

	//define variables
	std::uint64_t blockId_t;//to know which block we are processing
	unsigned int sizeCompressed, gcount;
	std::uint64_t offset;//size of block in bytes after compression


	const size_t bytesPerPixel = header.getBytesPerPixel();
	uint32_t blockSizeBytes = bytesPerPixel;	
	uint64_t dimsBlock[KLB_DATA_DIMS];//number of blocks on each dimension
	uint64_t coordBlock[KLB_DATA_DIMS];//coordinates (in block space). blockId_t = coordBblock[0] + dimsBblock[0] * coordBlock[1] + dimsBblock[0] * dimsBblock[1] * coordBlock[2] + ...
	uint64_t offsetBuffer;//starting offset for each buffer within ROI
	uint32_t blockSizeAux[KLB_DATA_DIMS];//for border cases where the blocksize might be different
	uint64_t xyzctCum[KLB_DATA_DIMS];//to calculate offsets for each dimension in THE ROI
	uint64_t offsetHeaderBytes = header.getSizeInBytes();

	xyzctCum[0] = bytesPerPixel;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
	{
		blockSizeBytes *= header.blockSize[ii];
		dimsBlock[ii] = ceil((float)(header.xyzct[ii]) / (float)(header.blockSize[ii]));
		if (ii > 0)
			xyzctCum[ii] = xyzctCum[ii - 1] * header.xyzct[ii - 1];
	}

	std::uint64_t numBlocks = header.getNumBlocks();
	char* bufferIn = new char[blockSizeBytes];//temporary storage for decompressed block
	char* bufferPtr;//pointer to preloaded compressed file in memory

	//main loop to keep processing blocks while they are available
	while (1)
	{
		//get the blockId resource		
		blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);

		//check if we have more blocks
		if (blockId_t >= numBlocks)
			break;

		//calculate coordinate (in block space)
		std::uint64_t blockIdx_aux = blockId_t;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			coordBlock[ii] = blockIdx_aux % dimsBlock[ii];
			blockIdx_aux -= coordBlock[ii];
			blockIdx_aux /= dimsBlock[ii];
			coordBlock[ii] *= header.blockSize[ii];//parsing coordinates to image space (not block anymore)
		}		

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d reading block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif


		//uncompress block into temp bufferIn
		sizeCompressed = header.getBlockCompressedSizeBytes(blockId_t);
		offset = header.getBlockOffset(blockId_t);

		bufferPtr = &(bufferImgFull[offsetHeaderBytes + offset]);		

		//apply decompression to block
		switch (header.compressionType)
		{
		case KLB_COMPRESSION_TYPE::NONE://no compression
			gcount = sizeCompressed;
			memcpy(bufferIn, bufferPtr, gcount);
			break;
		case KLB_COMPRESSION_TYPE::BZIP2://bzip2
		{
				   gcount = blockSizeBytes;
				   int ret = BZ2_bzBuffToBuffDecompress(bufferIn, &gcount, bufferPtr, sizeCompressed, 0, 0);
				   if (ret != BZ_OK)
				   {
					   std::cout << "ERROR: workerfunc: decompressing data at block " << blockId_t << std::endl;
					   *errFlag = 2;
					   gcount = 0;
				   }
				   break;
		}
		case KLB_COMPRESSION_TYPE::ZLIB:
		{
										   z_stream strm;
										   strm.zalloc = Z_NULL;
										   strm.zfree = Z_NULL;
										   strm.opaque = Z_NULL;
										   strm.avail_in = 0;
										   strm.next_in = Z_NULL;
										   *errFlag = inflateInit(&strm);


										   strm.avail_out = blockSizeBytes;
										   strm.next_out = (Bytef*)bufferIn;
										   strm.avail_in = sizeCompressed;
										   strm.next_in = (Bytef*)bufferPtr;
										   strm.data_type = Z_BINARY;//data type

										   int ret = inflate(&strm, Z_FINISH);
										   //gcount = sizeCompressed - strm.avail_out;
										   if (ret != Z_STREAM_END && ret != Z_OK)
										   {
											   std::cout << "ERROR: workerfunc: uncompressing data at block " << blockId_t << " with zlib. Error code " << ret << std::endl;
											   *errFlag = 3;
											   gcount = 0;
										   }
										   //release strm
										   (void)deflateEnd(&strm);
										   break;
		}
		default:
			std::cout << "ERROR: workerfunc: decompression type not implemented" << std::endl;
			*errFlag = 5;
			sizeCompressed = 0;
		}



		//-------------------parse bufferIn to bufferOut image buffer-----------------------------------		

		//calculate block size in case we had border block				
		blockSizeAux[0] = std::min(header.blockSize[0], (uint32_t)(header.xyzct[0] - coordBlock[0]));
		for (int ii = 1; ii < KLB_DATA_DIMS; ii++)
		{
			blockSizeAux[ii] = std::min(header.blockSize[ii], (uint32_t)(header.xyzct[ii] - coordBlock[ii]));
		}
		
				

		//calculate starting offset in the buffer in image space
		offsetBuffer = 0;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			offsetBuffer += coordBlock[ii] * xyzctCum[ii];//xyzct already has bytesPerPixel
		}

		//copy block into local buffer bufferIn
		int auxDim = 1;
		uint32_t bcount[KLB_DATA_DIMS];
		memset(bcount, 0, sizeof(uint32_t)* KLB_DATA_DIMS);
		const size_t bufferCopySize = bytesPerPixel * blockSizeAux[0];
		char* bufferInPtr = bufferIn;
		while (auxDim < KLB_DATA_DIMS)
		{
			
			//copy fastest moving coordinate all at once for efficiency
			memcpy(&(bufferOut[offsetBuffer]), bufferInPtr, bufferCopySize);
			bufferInPtr += bufferCopySize;

			//increment counter			
			bcount[1]++;
			offsetBuffer += xyzctCum[1];//update offset for output buffer. xyzct already has bytesPerPixel		
			auxDim = 1;
			while (bcount[auxDim] == blockSizeAux[auxDim])
			{
				offsetBuffer -= blockSizeAux[auxDim] * xyzctCum[auxDim];//update buffer				
				bcount[auxDim] = 0;
				auxDim++;
				if (auxDim == KLB_DATA_DIMS)
					break;
				bcount[auxDim]++;
				offsetBuffer += xyzctCum[auxDim]; //update buffer.xyzct already has bytesPerPixel
			}
		}
		//-------------------end of parse bufferIn to bufferOut image buffer-----------------------------------



#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d finished decompressing block %d into %d bytes\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)gcount);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
	}


	//release memory
	delete[] bufferIn;

}

//======================================================
void klb_imageIO::blockUncompressorImageFull(char* bufferOut, std::atomic<uint64_t>	*blockId, int *errFlag)
{
	*errFlag = 0;

	//open file to read elements	
	FILE* fid = fopen(filename.c_str(), "rb");
	if ( fid == NULL )
	{
		cout << "ERROR: blockUncompressor: thread opening file " << filename << endl;
		*errFlag = 3;
		return;
	}

	//define variables
	std::uint64_t blockId_t;//to know which block we are processing
	unsigned int sizeCompressed, gcount;
	std::uint64_t offset;//size of block in bytes after compression


	const size_t bytesPerPixel = header.getBytesPerPixel();
	uint32_t blockSizeBytes = bytesPerPixel;
	uint64_t dimsBlock[KLB_DATA_DIMS];//number of blocks on each dimension
	uint64_t coordBlock[KLB_DATA_DIMS];//coordinates (in block space). blockId_t = coordBblock[0] + dimsBblock[0] * coordBlock[1] + dimsBblock[0] * dimsBblock[1] * coordBlock[2] + ...
	uint64_t offsetBuffer;//starting offset for each buffer within ROI
	uint32_t blockSizeAux[KLB_DATA_DIMS];//for border cases where the blocksize might be different
	uint64_t xyzctCum[KLB_DATA_DIMS];//to calculate offsets for each dimension in THE ROI
	uint64_t offsetHeaderBytes = header.getSizeInBytes();

	xyzctCum[0] = bytesPerPixel;
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
	{
		blockSizeBytes *= header.blockSize[ii];
		dimsBlock[ii] = ceil((float)(header.xyzct[ii]) / (float)(header.blockSize[ii]));
		if (ii > 0)
			xyzctCum[ii] = xyzctCum[ii - 1] * header.xyzct[ii - 1];
	}

	std::uint64_t numBlocks = header.getNumBlocks();
	char* bufferIn = new char[blockSizeBytes];//temporary storage for decompressed block
	char* bufferFile = new char[maximumBlockSizeCompressedInBytes()];//temporary storage for compressed block from file

	//main loop to keep processing blocks while they are available
	while (1)
	{
		//get the blockId resource		
		blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);

		//check if we have more blocks
		if (blockId_t >= numBlocks)
			break;

		//calculate coordinate (in block space)
		std::uint64_t blockIdx_aux = blockId_t;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			coordBlock[ii] = blockIdx_aux % dimsBlock[ii];
			blockIdx_aux -= coordBlock[ii];
			blockIdx_aux /= dimsBlock[ii];
			coordBlock[ii] *= header.blockSize[ii];//parsing coordinates to image space (not block anymore)
		}

#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d reading block %d out of %d total blocks\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif

		//uncompress block into temp bufferIn
		sizeCompressed = header.getBlockCompressedSizeBytes(blockId_t);
		offset = header.getBlockOffset(blockId_t);


		fseek(fid, offsetHeaderBytes + offset, SEEK_SET);
		fread(bufferFile, 1, sizeCompressed, fid);

		//apply decompression to block
		switch (header.compressionType)
		{
		case KLB_COMPRESSION_TYPE::NONE://no compression
			gcount = sizeCompressed;
			memcpy(bufferIn, bufferFile, gcount);
			break;
		case KLB_COMPRESSION_TYPE::BZIP2://bzip2
		{
				   gcount = blockSizeBytes;
				   int ret = BZ2_bzBuffToBuffDecompress(bufferIn, &gcount, bufferFile, sizeCompressed, 0, 0);
				   if (ret != BZ_OK)
				   {
					   std::cout << "ERROR: workerfunc: decompressing data at block " << blockId_t << " with offset " << offsetHeaderBytes + offset << std::endl;
					   *errFlag = 2;
					   gcount = 0;
				   }
				   break;
		}
		case KLB_COMPRESSION_TYPE::ZLIB:
		{
										   z_stream strm;
										   strm.zalloc = Z_NULL;
										   strm.zfree = Z_NULL;
										   strm.opaque = Z_NULL;
										   strm.avail_in = 0;
										   strm.next_in = Z_NULL;
										   *errFlag = inflateInit(&strm);


										   strm.avail_out = blockSizeBytes;
										   strm.next_out = (Bytef*)bufferIn;
										   strm.avail_in = sizeCompressed;
										   strm.next_in = (Bytef*)bufferFile;
										   strm.data_type = Z_BINARY;//data type

										   int ret = inflate(&strm, Z_FINISH);
										   //gcount = sizeCompressed - strm.avail_out;
										   if (ret != Z_STREAM_END && ret != Z_OK)
										   {
											   std::cout << "ERROR: workerfunc: uncompressing data at block " << blockId_t << " with zlib. Error code " << ret << std::endl;
											   *errFlag = 3;
											   gcount = 0;
										   }
										   //release strm
										   (void)deflateEnd(&strm);
										   break;
		}
		default:
			std::cout << "ERROR: workerfunc: decompression type not implemented" << std::endl;
			*errFlag = 5;
			sizeCompressed = 0;
		}



		//-------------------parse bufferIn to bufferOut image buffer-----------------------------------		

		//calculate block size in case we had border block				
		blockSizeAux[0] = std::min(header.blockSize[0], (uint32_t)(header.xyzct[0] - coordBlock[0]));
		for (int ii = 1; ii < KLB_DATA_DIMS; ii++)
		{
			blockSizeAux[ii] = std::min(header.blockSize[ii], (uint32_t)(header.xyzct[ii] - coordBlock[ii]));
		}



		//calculate starting offset in the buffer in image space
		offsetBuffer = 0;
		for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		{
			offsetBuffer += coordBlock[ii] * xyzctCum[ii];//xyzct already has bytesPerPixel
		}

		//copy block into local buffer bufferIn
		int auxDim = 1;
		uint32_t bcount[KLB_DATA_DIMS];
		memset(bcount, 0, sizeof(uint32_t)* KLB_DATA_DIMS);
		const size_t bufferCopySize = bytesPerPixel * blockSizeAux[0];
		char* bufferInPtr = bufferIn;
		while (auxDim < KLB_DATA_DIMS)
		{

			//copy fastest moving coordinate all at once for efficiency
			memcpy(&(bufferOut[offsetBuffer]), bufferInPtr, bufferCopySize);
			bufferInPtr += bufferCopySize;

			//increment counter			
			bcount[1]++;
			offsetBuffer += xyzctCum[1];//update offset for output buffer. xyzct already has bytesPerPixel		
			auxDim = 1;
			while (bcount[auxDim] == blockSizeAux[auxDim])
			{
				offsetBuffer -= blockSizeAux[auxDim] * xyzctCum[auxDim];//update buffer				
				bcount[auxDim] = 0;
				auxDim++;
				if (auxDim == KLB_DATA_DIMS)
					break;
				bcount[auxDim]++;
				offsetBuffer += xyzctCum[auxDim]; //update buffer.xyzct already has bytesPerPixel
			}
		}
		//-------------------end of parse bufferIn to bufferOut image buffer-----------------------------------



#ifdef DEBUG_PRINT_THREADS
		printf("Thread %d finished decompressing block %d into %d bytes\n", (int)(std::this_thread::get_id().hash()), (int)blockId_t, (int)gcount);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
	}


	//release memory
	fclose(fid);
	delete[] bufferIn;
	delete[] bufferFile;
}

//=========================================================================
//writes compressed blocks sequentially as they become available (in order) from the workers
void klb_imageIO::blockWriter(FILE* fout, int* g_blockSize, int* g_blockThreadId, klb_circular_dequeue** cq, int* errFlag)
{
	*errFlag = 0;
	std::int64_t nextBlockId = 0, offset = 0;
	std::uint64_t numBlocks = header.getNumBlocks();
	header.resizeBlockOffset(numBlocks);//just in case it has not been setup

#define USE_MEM_BUFFER_WRITE //uncomment this line to use a large memory buffer before writing to file. It is slower since C++ write already buffers ofstream before flushing. However using C interface (FILE*) it is faster)

#ifdef USE_MEM_BUFFER_WRITE
	//buffer to avoid writing to disk all the time
	int bufferMaxSize = std::min( header.getImageSizeBytes() / 10, (uint64_t) (500 * 1048576));//maximum is 500MB or 10th of the original image size
	bufferMaxSize = std::max(bufferMaxSize, (int) maximumBlockSizeCompressedInBytes());//we need ot be able to fit at least one block
	char* bufferMem = new char[bufferMaxSize];
	int bufferOffset = 0;
#endif		

	//write header
	header.writeHeader(fout);	

	// loop until end is signaled			
	std::int64_t blockSize;
	while (nextBlockId < numBlocks)
	{

#ifdef DEBUG_PRINT_THREADS
		printf("Writer trying to append block %d out of %d\n", (int)nextBlockId, (int)numBlocks);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
		std::unique_lock<std::mutex> locker(g_lockqueue);//acquires the lock but this is the only thread using it. We cannot have condition_variables without a mutex
		g_queuecheck.wait(locker, [&](){return (g_blockSize[nextBlockId] >= 0 && g_blockThreadId[nextBlockId] >= 0); });//releases the lock until notify. If condition is not satisfied, it waits again

		locker.unlock();

#ifdef DEBUG_PRINT_THREADS
		printf("Writer appending block %d out of %d with %d bytes\n", (int)nextBlockId, (int) numBlocks,g_blockSize[nextBlockId]);
		fflush(stdout); // Will now print everything in the stdout buffer
#endif
				
		//write block
		blockSize = g_blockSize[nextBlockId];

#ifdef USE_MEM_BUFFER_WRITE
		//use large memory buffer
		if (bufferOffset + blockSize > bufferMaxSize)//we need to flush the buffer
		{
			fwrite(bufferMem, 1, bufferOffset, fout);
			bufferOffset = 0;
		}
		
		//add block to the memory buufer
		memcpy(&(bufferMem[bufferOffset]), cq[g_blockThreadId[nextBlockId]]->getReadBlock(), blockSize);
		bufferOffset += blockSize;
#else
		fwrite(cq[g_blockThreadId[nextBlockId]]->getReadBlock(),1, blockSize, fout);
#endif
		//now we can release data
		cq[g_blockThreadId[nextBlockId]]->popReadBlock();
		offset += blockSize;

		//update header blockOffset
		header.blockOffset[nextBlockId] = offset;//at the end, so we can recover length for all blocks

		//update variables
		nextBlockId++;
	}
#ifdef USE_MEM_BUFFER_WRITE
	//flush the rest of the buffer
	fwrite(bufferMem, 1, bufferOffset, fout);
#endif
	//update header.blockOffset	
	fseek(fout, header.getSizeInBytesFixPortion(), SEEK_SET);
	fwrite((char*)(&(header.blockOffset[0])), 1, header.Nb * sizeof(std::uint64_t), fout);

	//close file	
	fclose(fout);

#ifdef USE_MEM_BUFFER_WRITE
	delete[] bufferMem;
#endif
}

int klb_imageIO::Predictor_both(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	uint16_t* dpImage = nullptr;
	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;

	cudaMalloc(&dpImage, (long)(sizeX * sizeY * 2 * sizeof(uint16_t)));
	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));
	std::vector<uint16_t> symbols(header.getImageSizePixels());

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{	
			if (z == 0)
			{
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)( sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}
			else
			{
				cudaMemcpy((dpImage + sizeX * sizeY), (in + (z - 1)*sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}
		
		}
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_tiles_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU(dpSymbols, dpBuffer, sizeX, sizeY, 1, 0, 0);	
			cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	cudaFree(dpImage);
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
}
int klb_imageIO::Predictor_space(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	uint16_t* dpImage = nullptr;
	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;

	cudaMalloc(&dpImage, (long)(sizeX * sizeY * 2 * sizeof(uint16_t)));
	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));
	std::vector<uint16_t> symbols(header.getImageSizePixels());

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			if (z == 0)
			{
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)( sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}
			else
			{
				cudaMemcpy((dpImage + sizeX * sizeY), (in + (z - 1)*sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}		
		}
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_space_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			// case (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2_Exten):
			// {
			// 	predictor7_space_exten_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, tileSize);
			// 	break;
			// }
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU(dpSymbols, dpBuffer, sizeX, sizeY, 1, 0, 0);	
			cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	cudaFree(dpImage);
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
}
int klb_imageIO::Predictor_angle(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	uint16_t* dpImage = nullptr;
	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;
	
	cudaMalloc(&dpImage, (long)(sizeX * sizeY * 2 * sizeof(uint16_t)));
	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));
	std::vector<uint16_t> symbols(header.getImageSizePixels());

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			if (z == 0)
			{
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)( sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}
			else
			{
				cudaMemcpy((dpImage + sizeX * sizeY), (in + (z - 1)*sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpImage, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
				cudaMemcpy(dpBuffer, dpImage, (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyDeviceToDevice);					
			}		
		}
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_angle_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			// case (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2_Exten):
			// {
			// 	predictor7_angle_exten_GPU(dpImage, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, tileSize);
			// 	break;
			// }
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU(dpSymbols, dpBuffer, sizeX, sizeY, 1, 0, 0);	
			cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	cudaFree(dpImage);
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
}

int klb_imageIO::Predictor_space_GPU(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	// std::vector<uint16_t> symbols(header.getImageSizePixels());
	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				cudaMemcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
				// memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_space_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			// case (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2_Exten):
			// {
			// 	predictor7_angle_exten_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, tileSize);
			// 	break;
			// }
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU((out + z *sizeX * sizeY), dpBuffer, sizeX, sizeY, 1, 0, 0);	
			// cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	// cudaFree(dpImage);
	cudaFree(dpBuffer);
	// cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
}

int klb_imageIO::Predictor_angle_GPU(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	// std::vector<uint16_t> symbols(header.getImageSizePixels());

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				cudaMemcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
				// memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_angle_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			// case (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2_Exten):
			// {
			// 	predictor7_angle_exten_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, tileSize);
			// 	break;
			// }
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU((out + z *sizeX * sizeY), dpBuffer, sizeX, sizeY, 1, 0, 0);	
			// cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	// cudaFree(dpImage);
	cudaFree(dpBuffer);
	// cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
}

int klb_imageIO::Predictor_both_GPU(const uint16_t* in, uint16_t* out, int tileSize, int predictors)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	// std::vector<uint16_t> symbols(header.getImageSizePixels());

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	for(int z = 0; z < sizeZ; z++)
	{
		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				cudaMemcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
				// memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				predictor1_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				predictor2_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				predictor3_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				predictor4_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				predictor5_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				predictor6_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}


			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				predictor7_tiles_GPU(in, dpBuffer, sizeX * sizeof(uint16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			default:
				std::cout << "ERROR: The predictors hava not selected!" << std::endl;
				return 1;
		}	
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			symbolize_GPU((out + z *sizeX * sizeY), dpBuffer, sizeX, sizeY, 1, 0, 0);	
			// cudaMemcpy((out + z *sizeX * sizeY), dpSymbols, (long)(sizeX * sizeY  * sizeof(int16_t)), cudaMemcpyDeviceToHost);
		}	

	}
	// cudaFree(dpImage);
	cudaFree(dpBuffer);
	// cudaFree(dpSymbols);
	return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
}

int klb_imageIO::unPredictor(const int16_t* in, uint16_t* out, int tileSize)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(int16_t)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	std::uint8_t predictors = header.headerVersion & 0x7F;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			cudaMemcpy(dpSymbols, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
			unsymbolize_GPU(dpBuffer, dpSymbols, sizeX, sizeY, 1, 0, 0);
			cudaMemcpy(((int16_t*)in  + z *sizeX * sizeY), dpBuffer, (long)sizeX * sizeY * 1 * sizeof(int16_t), cudaMemcpyDeviceToHost);
		}

		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				unPredictor1_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				unPredictor2_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				unPredictor3_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				unPredictor4_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				unPredictor5_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				unPredictor6_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				unPredictor7_tiles(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			default:
				std::cout << "ERROR: The unPredictors hava not selected!" << std::endl;
				return 1;
		}	
	}
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
}

int klb_imageIO::unPredictor_space(const int16_t* in, uint16_t* out, int tileSize)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(float)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));
	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	std::uint8_t predictors = header.headerVersion & 0x7F;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			cudaMemcpy(dpSymbols, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
			unsymbolize_GPU(dpBuffer, dpSymbols, sizeX, sizeY, 1, 0, 0);
			cudaMemcpy(((int16_t*)in  + z *sizeX * sizeY), dpBuffer, (long)sizeX * sizeY * 1 * sizeof(int16_t), cudaMemcpyDeviceToHost);
		}

		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				unPredictor1_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				unPredictor2_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				unPredictor3_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				unPredictor4_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				unPredictor5_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				unPredictor6_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				unPredictor7_space(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			default:
				std::cout << "ERROR: The unPredictors hava not selected!" << std::endl;
				return 1;
		}	
	}
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
}

int klb_imageIO::unPredictor_angle(const int16_t* in, uint16_t* out, int tileSize)
{

	uint64_t sizeX = header.xyzct[0];
	uint64_t sizeY = header.xyzct[1];
	uint64_t sizeZ = header.xyzct[2];

	int16_t* dpBuffer = nullptr;
	uint16_t* dpSymbols = nullptr;

	cudaMalloc(&dpBuffer, (long)(sizeX * sizeY  * sizeof(float)));
	cudaMalloc(&dpSymbols, (long)(sizeX * sizeY  * sizeof(uint16_t)));

	std::uint8_t i_or_v = (header.headerVersion & 0x80) >> 7;
	std::uint8_t predictors = header.headerVersion & 0x7F;
	for(int z = 0; z < sizeZ; z++)
	{
		if (predictors != LFM_PREDICTORS_TYPE::NO_PREIDICTORS)
		{
			cudaMemcpy(dpSymbols, (in + z *sizeX * sizeY), (long)(sizeX * sizeY  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
			unsymbolize_GPU(dpBuffer, dpSymbols, sizeX, sizeY, 1, 0, 0);
			cudaMemcpy(((int16_t*)in  + z *sizeX * sizeY), dpBuffer, (long)sizeX * sizeY * 1 * sizeof(int16_t), cudaMemcpyDeviceToHost);
		}

		switch (predictors)
		{
			case (LFM_PREDICTORS_TYPE::NO_PREIDICTORS):
			{
				memcpy((out + z *sizeX * sizeY), (in + z *sizeX * sizeY), sizeX * sizeY  * sizeof(uint16_t));
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A):
			{
				unPredictor1_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B):
			{
				unPredictor2_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_C):
			{
				unPredictor3_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_DC):
			{
				unPredictor4_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_A_BDC_Div2):
			{
				unPredictor5_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_B_ADC_Div2):
			{
				unPredictor6_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			case  (LFM_PREDICTORS_TYPE::PREIDCTORS_APB_Div2):
			{
				unPredictor7_angle(((int16_t*)in + z *sizeX * sizeY), (out + z *sizeX * sizeY), sizeX * sizeof(int16_t), sizeX, sizeY, i_or_v&z, tileSize);
				break;
			}
			default:
				std::cout << "ERROR: The unPredictors hava not selected!" << std::endl;
				return 1;
		}	
	}
	cudaFree(dpBuffer);
	cudaFree(dpSymbols);
}

int klb_imageIO::bwt_entropy_comparsion_2D(uint16_t* dpPredict, const uint16_t* dpSrc)
{
	uint64_t block_size = 450000;
	uint8_t *dpIn = nullptr;
	uint8_t *dpFirst = nullptr;
	uint8_t *dpLast = nullptr;
	unsigned long long int* dpStatic = nullptr;
	float* dpEntropy = nullptr;

	cudaMalloc(&dpIn, (long)(block_size * sizeof(uint16_t)));
	cudaMalloc(&dpFirst, (long)(block_size  * sizeof(float))+1);
	cudaMalloc(&dpLast, (long)(block_size  * sizeof(uint16_t))+1);
	cudaMalloc(&dpStatic, (long)(65535  * sizeof(unsigned long long int)));
	cudaMalloc(&dpEntropy, (long)(65535  * sizeof(float)));

	// std::vector<uint16_t> buffer_In(450000);
	// std::vector<uint8_t> buffer_first(450000 * sizeof(uint16_t)+1);
	// std::vector<uint8_t> buffer_last(450000 * sizeof(uint16_t)+1);
	// std::vector<uint64_t> buffer_static(65535 * sizeof(uint64_t));
	float entropy_A = 0;
	float entropy_B = 0;
	
	for (int z = 0 ; z < header.getImageSizePixels(); z+=block_size)
	{
		int64_t loop_i = block_size;
		if (header.getImageSizePixels() - z < block_size)
			loop_i = header.getImageSizePixels() - z;

		cudaMemsetAsync(dpStatic, 0, (long)(65535 * sizeof(unsigned long long int)));
		cudaMemsetAsync(dpEntropy, 0, (long)(65535 * sizeof(float)));
		// cudaMemsetAsync(dpIn, 0, (long)(450000 * sizeof(uint16_t)));
		cudaMemcpy(dpIn, dpPredict + z, (long)(loop_i * sizeof(uint16_t)), cudaMemcpyHostToDevice);
		bwt_GPU(dpIn, dpFirst, dpLast, loop_i * sizeof(uint16_t));
		static_bwt_GPU(dpLast, dpStatic, loop_i * sizeof(uint16_t));
		sum_bwt_GPU(dpStatic, dpEntropy, loop_i * sizeof(uint16_t), &entropy_A);

		cudaMemsetAsync(dpStatic, 0, (long)(65535 * sizeof(unsigned long long int)));
		cudaMemsetAsync(dpEntropy, 0, (long)(65535 * sizeof(float)));
		// cudaMemsetAsync(dpIn, 0, (long)(450000 * sizeof(uint16_t)));
		cudaMemcpy(dpIn, dpSrc + z, (long)(loop_i * sizeof(uint16_t)), cudaMemcpyHostToDevice);
			
		bwt_GPU(dpIn, dpFirst, dpLast, loop_i * sizeof(uint16_t));
		static_bwt_GPU(dpLast, dpStatic, loop_i * sizeof(uint16_t));
		sum_bwt_GPU(dpStatic, dpEntropy, loop_i * sizeof(uint16_t), &entropy_B);	
		
	}
	cudaFree(dpIn);
	cudaFree(dpFirst);
	cudaFree(dpLast);
	cudaFree(dpStatic);
	cudaFree(dpEntropy);

	if (entropy_A/entropy_B > 0.96)	
		return 1;

	return 0;
}

float klb_imageIO::bwt_entropy_2D(uint16_t* In, float *entropy, int is_src)
{
	uint64_t block_size = 450000;
	// uint8_t *dpIn = nullptr;
	uint8_t *dpFirst = nullptr;
	uint8_t *dpLast = nullptr;
	unsigned long long int* dpStatic = nullptr;
	float* dpEntropy = nullptr;

	// dpIn = (uint8_t*)In;
	// cudaMalloc(&dpIn, (long)(block_size * sizeof(uint16_t)));
	cudaMalloc(&dpFirst, (long)(block_size  * sizeof(uint16_t))+1);
	cudaMalloc(&dpLast, (long)(block_size  * sizeof(uint16_t))+1);
	cudaMalloc(&dpStatic, (long)(65535  * sizeof(unsigned long long int)));
	cudaMalloc(&dpEntropy, (long)(65535  * sizeof(float)));
	// std::vector<uint16_t> buffer_In(450000);
	// std::vector<uint8_t> buffer_first(450000 * sizeof(uint16_t)+1);
	// std::vector<uint8_t> buffer_last(450000 * sizeof(uint16_t)+1);
	// std::vector<uint64_t> buffer_static(65535 * sizeof(uint64_t));
	float entropy_A = 0;
	
	for (int z = 0 ; z < header.getImageSizePixels(); z += block_size)
	{
		int64_t loop_i = block_size;
		if (header.getImageSizePixels() - z < block_size)
			loop_i = header.getImageSizePixels() - z;

		cudaMemsetAsync(dpStatic, 0, (long)(65535 * sizeof(unsigned long long int)));
		cudaMemsetAsync(dpEntropy, 0, (long)(65535 * sizeof(float)));
		// cudaMemsetAsync(dpIn, 0, (long)(450000 * sizeof(uint16_t)));

		bwt_GPU( ((uint8_t*)In + z * sizeof(uint16_t)), dpFirst, dpLast, loop_i * sizeof(uint16_t) );
		// std::vector<uint8_t> buffer_last(2*loop_i +1);
		// cudaMemcpy(buffer_last.data(), dpLast, (long)(loop_i * sizeof(uint16_t))+1, cudaMemcpyDeviceToHost);
		// map<uint64_t, int> map_bwt_predict;
		// for (int i = 0; i < 2*loop_i-7 ; i++)
		// {
		// 	map_bwt_predict[(buffer_last[i]<<56) | (buffer_last[i+1] <<48) |(buffer_last[+2]<<40) | (buffer_last[i+4] <<32) |(buffer_last[i+4]<<24) | (buffer_last[i+5] <<16) | (buffer_last[i+6]<<8) | (buffer_last[i+7])]++;
		// }
		// std::vector<int> map_src_sort;
		// for (auto it = map_bwt_predict.begin(); it != map_bwt_predict.end(); it++)
		// 	map_src_sort.push_back(it->second);

		// sort(map_src_sort.begin(), map_src_sort.end(), greater<int>());
		// for (int i = 0 ; (i < map_src_sort.size()) && (map_src_sort.at(i)!=0); i++)
		// {
		// 	entropy_A = entropy_A - map_src_sort.at(i)/(double)(loop_i)*log(map_src_sort.at(i)/(double)(loop_i));
		// }		
		static_bwt_GPU( dpLast, dpStatic, loop_i * sizeof(uint16_t) );
		sum_bwt_GPU( dpStatic, dpEntropy, loop_i * sizeof(uint16_t), &entropy_A );
		
	}
	cudaFree(dpFirst);
	cudaFree(dpLast);
	cudaFree(dpStatic);
	cudaFree(dpEntropy);

	if (is_src != 0)
		*entropy = entropy_A;
	else
		*entropy = entropy_A*0.96;

	return entropy_A ;
}

int klb_imageIO::bwt_entropy_comparsion_1D(uint16_t* dpPredict, const uint16_t* dpSrc)
{
	uint64_t block_size = header.getImageSizePixels()-1;
	uint8_t *dpIn = nullptr;
	uint8_t *dpFirst = nullptr;
	uint8_t *dpLast = nullptr;
	unsigned long long int* dpStatic = nullptr;
	float* dpEntropy = nullptr;

	cudaMalloc(&dpIn, (long)(block_size * sizeof(uint16_t)));
	cudaMalloc(&dpFirst, (long)(block_size  * sizeof(float))+1);
	cudaMalloc(&dpLast, (long)(block_size  * sizeof(uint16_t))+1);
	cudaMalloc(&dpStatic, (long)(65535  * sizeof(unsigned long long int)));
	cudaMalloc(&dpEntropy, (long)(65535  * sizeof(float)));

	// std::vector<uint16_t> buffer_In(450000);
	// std::vector<uint8_t> buffer_first(450000 * sizeof(uint16_t)+1);
	// std::vector<uint8_t> buffer_last(450000 * sizeof(uint16_t)+1);
	// std::vector<uint64_t> buffer_static(65535 * sizeof(uint64_t));
	float entropy_A = 0;
	float entropy_B = 0;
	
	for (int z = 0 ; z < header.getImageSizePixels(); z+=block_size)
	{
		int64_t loop_i = block_size;
		if (header.getImageSizePixels() - z < block_size)
			loop_i = header.getImageSizePixels() - z;

		cudaMemsetAsync(dpStatic, 0, (long)(65535 * sizeof(unsigned long long int)));
		cudaMemsetAsync(dpEntropy, 0, (long)(65535 * sizeof(float)));
		// cudaMemsetAsync(dpIn, 0, (long)(450000 * sizeof(uint16_t)));
		cudaMemcpy(dpIn, dpPredict + z, (long)(loop_i * sizeof(uint16_t)), cudaMemcpyHostToDevice);
		bwt_GPU(dpIn, dpFirst, dpLast, loop_i * sizeof(uint16_t));
		static_GPU(dpLast, dpStatic, loop_i * sizeof(uint16_t));
		sum_bwt_GPU(dpStatic, dpEntropy, loop_i * sizeof(uint16_t), &entropy_A);

		cudaMemsetAsync(dpStatic, 0, (long)(65535 * sizeof(unsigned long long int)));
		cudaMemsetAsync(dpEntropy, 0, (long)(65535 * sizeof(float)));
		// cudaMemsetAsync(dpIn, 0, (long)(450000 * sizeof(uint16_t)));
		cudaMemcpy(dpIn, dpSrc + z, (long)(loop_i * sizeof(uint16_t)), cudaMemcpyHostToDevice);
			
		bwt_GPU(dpIn, dpFirst, dpLast, loop_i * sizeof(uint16_t));
		static_GPU(dpLast, dpStatic, loop_i * sizeof(uint16_t));
		sum_bwt_GPU(dpStatic, dpEntropy, loop_i * sizeof(uint16_t), &entropy_B);	
		
	}
	cudaFree(dpIn);
	cudaFree(dpFirst);
	cudaFree(dpLast);
	cudaFree(dpStatic);
	cudaFree(dpEntropy);

	if (entropy_A/entropy_B > 0.98)	
		return 1;

	return 0;
}

int klb_imageIO::huffman_bits_comparsion(uint16_t* dpPredict, const uint16_t* dpSrc)
{

	unsigned long long int* dpStatic = nullptr;
	uint8_t* dpSymbols = nullptr;
	cudaMalloc(&dpStatic, (long)(65535  * sizeof(uint64_t)));
	cudaMalloc(&dpSymbols, (long)(450000  * sizeof(uint16_t)));

	// std::vector<uint64_t> sort_predictA(65535);
	// std::vector<uint64_t> sort_predictB(65535);

	unsigned long long int predict_need_bits = 0;
	unsigned long long int src_need_bits = 0;
	for (int z = 0 ; z < header.getImageSizePixels(); z+=450000)
	{
		int64_t loop_i = 450000;
		if (header.getImageSizePixels() - z < 450000)
			loop_i = header.getImageSizePixels() - z;

		cudaMemsetAsync(dpStatic, 0, 65535  * sizeof(unsigned long long int));
		cudaMemcpy(dpSymbols, (dpPredict + z), (long)(loop_i  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
		static_GPU(dpSymbols, dpStatic, (loop_i  * sizeof(uint16_t)));
		// cudaMemcpy(sort_predictA.data(), dpStatic, (long)(65535  * sizeof(uint64_t)), cudaMemcpyDeviceToHost);
		sort_GPU(dpStatic, 65535);
		sum_GPU(dpStatic, 65535, &predict_need_bits);	
		
		cudaMemsetAsync(dpStatic, 0, 65535  * sizeof(unsigned long long int));
		cudaMemcpy(dpSymbols, (dpSrc + z), (long)(loop_i  * sizeof(uint16_t)), cudaMemcpyHostToDevice);
		static_GPU(dpSymbols, dpStatic, (loop_i  * sizeof(uint16_t)));
		sort_GPU(dpStatic,65535);
		// cudaMemcpy(sort_predictA.data(), dpStatic, (long)(65535  * sizeof(int64_t)), cudaMemcpyDeviceToHost);
		sum_GPU(dpStatic, 65535, &src_need_bits);
		// cudaMemcpy(sort_predictB.data(), dpStatic, (long)(65535  * sizeof(int64_t)), cudaMemcpyDeviceToHost);
	}

	cudaFree(dpSymbols);
	cudaFree(dpStatic);	

	if (predict_need_bits > src_need_bits)
		return 1;

	return 0;
}

int klb_imageIO::predict_and_2DEntropy(uint16_t *In, uint16_t **out, float *entropy, std::atomic<uint64_t> *blockId, int numPredictors)
{
	std::uint64_t blockId_t;
	
	while(true)
	{
		blockId_t = atomic_fetch_add(blockId, (uint64_t) 1);
		if(blockId_t >= numPredictors)
			break;
		switch (LFM_PREDICTOR_WAY)
		{
		case (LFM_PREDICTORS::ANGLE_AND_SPACE):
			Predictor_both_GPU(In, out[blockId_t], header.Nnum, blockId_t);
			break;
		case  (LFM_PREDICTORS::ANGLE):
			Predictor_angle_GPU(In, out[blockId_t], header.Nnum, blockId_t);
			break;
		case  (LFM_PREDICTORS::SPACE):
			Predictor_space_GPU(In, out[blockId_t], header.Nnum, blockId_t);
			break;			
		default:
			std::cout << "ERROR: The predictors(angle or space or both) hava not selected!" << std::endl;
			break;
		}
		
		bwt_entropy_2D(out[blockId_t], &entropy[blockId_t], blockId_t);
	}
	return 0;
}


int klb_imageIO::image_realign(uint16_t* In, uint16_t* Out)
{
	return 0;
}

//=======================================================

klb_imageIO::klb_imageIO()
{
	numThreads = std::thread::hardware_concurrency();
}

klb_imageIO::klb_imageIO(const std::string &filename_)
{
	filename = filename_;//it could be used as output or input file
	numThreads = std::thread::hardware_concurrency();
}

//=================================================

int klb_imageIO::writeImage(const char* img, int numThreads)
{
	// auto t1_All = Clock::now();
	uint16_t* img_to_compress = nullptr;
	//redirect standard out
#ifdef DEBUG_PRINT_THREADS
	//cout << "Redirecting stdout for klb_imageIO::writeImage" << endl;
	//freopen("E:/temp/cout_klb_imageIO.txt", "w", stdout);	
#endif

	if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();

	//open output file
	//std::ofstream fout(filenameOut.c_str(), std::ios::binary | std::ios::out);	
	//we do this before calling the thread in case we have problems
	FILE* fout = fopen(filename.c_str(), "wb");//for wahtever reason FILE* is 4X faster than std::ofstream over the network. C interface is much faster than C++ streams
	if (fout == NULL)
	{
		std::cout << "ERROR: file " << filename << " could not be opened" << std::endl;
		return 5;
	}
	// auto t1 = Clock::now();
	std::vector<uint16_t> symbols(header.getImageSizePixels());
	std::uint8_t predictor = header.headerVersion & 0x7F;
	if( predictor < NUM_PREDICTORS && header.xyzct[2] == 1)
	{
		header.headerVersion = header.headerVersion & 0x80;
		std::atomic<uint64_t> predictors_blockId;//counter shared all workers so each worker thread knows which block to readblockId = 0;
		atomic_store(&predictors_blockId, (uint64_t)0);

		uint16_t *srcImg = nullptr;
		cudaMalloc(&srcImg, header.getImageSizeBytes());
		cudaMemcpy(srcImg, img, header.getImageSizeBytes(), cudaMemcpyHostToDevice);

		uint16_t *buff_Predictor[NUM_PREDICTORS];		
		for(int i = 0; i < NUM_PREDICTORS; i++)
		{
			cudaMalloc(&buff_Predictor[i], header.getImageSizeBytes());
		}
		float *entropy = new float[NUM_PREDICTORS];

		std::vector<std::thread> predictors_threads;
		for (int i = 0; i < numThreads; i++)
		{
			predictors_threads.push_back(std::thread(&klb_imageIO::predict_and_2DEntropy, this, srcImg, buff_Predictor, entropy, &predictors_blockId, NUM_PREDICTORS));
		}
		for (auto& t : predictors_threads)
		{
			t.join();
		}

		map<float,int> sort_entropy;
		for(int i = 0; i < NUM_PREDICTORS ; i++)
		{
			sort_entropy[entropy[i]] = i;
		}
		header.headerVersion = header.headerVersion | sort_entropy.begin()->second ;
		cudaMemcpy(symbols.data(), buff_Predictor[sort_entropy.begin()->second], header.getImageSizeBytes(), cudaMemcpyDeviceToHost);
		// img_to_compress = buff_Predictor[sort_entropy.begin()->second];
		for(int i = 0; i < NUM_PREDICTORS; i++)
		{
			cudaFree(buff_Predictor[i]);
		}
		cudaFree((void*)buff_Predictor);
		cudaFree(srcImg);
		delete[] entropy;
	}
	else if (predictor < NUM_PREDICTORS && header.xyzct[2] != 1)
	{
		header.headerVersion = header.headerVersion & 0x80;
		uint32_t frame = header.xyzct[2];
		header.xyzct[2] = 1;
		std::atomic<uint64_t> predictors_blockId;//counter shared all workers so each worker thread knows which block to readblockId = 0;
		atomic_store(&predictors_blockId, (uint64_t)0);

		uint16_t *srcImg = nullptr;
		uint32_t frameSize = header.xyzct[0] * header.xyzct[1] * header.getBytesPerPixel();
		cudaMalloc(&srcImg, frameSize);
		cudaMemcpy(srcImg, img,frameSize, cudaMemcpyHostToDevice);

		uint16_t *buff_Predictor[NUM_PREDICTORS];		
		for(int i = 0; i < NUM_PREDICTORS; i++)
		{
			cudaMalloc(&buff_Predictor[i], frameSize);
		}
		float *entropy = new float[NUM_PREDICTORS];

		std::vector<std::thread> predictors_threads;
		for (int i = 0; i < numThreads; i++)
		{
			predictors_threads.push_back(std::thread(&klb_imageIO::predict_and_2DEntropy, this, srcImg, buff_Predictor, entropy, &predictors_blockId, NUM_PREDICTORS));
		}
		for (auto& t : predictors_threads)
		{
			t.join();
		}

		map<float,int> sort_entropy;
		for(int i = 0; i < NUM_PREDICTORS ; i++)
		{
			sort_entropy[entropy[i]] = i;
		}

		header.headerVersion = header.headerVersion | sort_entropy.begin()->second;
		header.xyzct[2] = frame;
		for(int i = 0; i < NUM_PREDICTORS; i++)
		{
			cudaFree(buff_Predictor[i]);
		}
		cudaFree((void*)buff_Predictor);
		cudaFree(srcImg);
		delete[] entropy;

		switch (LFM_PREDICTOR_WAY)
		{
		case (LFM_PREDICTORS::ANGLE_AND_SPACE):
			Predictor_both((uint16_t*)img, symbols.data(), header.Nnum, header.headerVersion & 0x7F);
			break;
		case  (LFM_PREDICTORS::ANGLE):
			Predictor_angle((uint16_t*)img, symbols.data(), header.Nnum, header.headerVersion & 0x7F);
			break;
		case  (LFM_PREDICTORS::SPACE):
			Predictor_space((uint16_t*)img, symbols.data(), header.Nnum, header.headerVersion & 0x7F);
			break;			
		default:
			std::cout << "ERROR: The predictors(angle or space or both) hava not selected!" << std::endl;
			break;
		}			
	}
	else
	{
		uint8_t cur_predictor = header.headerVersion & 0x7F - 8;
		header.headerVersion = (header.headerVersion & 0x80) | cur_predictor;
		switch (LFM_PREDICTOR_WAY)
		{
		case (LFM_PREDICTORS::ANGLE_AND_SPACE):
			Predictor_both((uint16_t*)img, symbols.data(), header.Nnum, cur_predictor);
			break;
		case  (LFM_PREDICTORS::ANGLE):
			Predictor_angle((uint16_t*)img, symbols.data(), header.Nnum, cur_predictor);
			break;
		case  (LFM_PREDICTORS::SPACE):
			Predictor_space((uint16_t*)img, symbols.data(), header.Nnum, cur_predictor);
			break;			
		default:
			std::cout << "ERROR: The predictors(angle or space or both) hava not selected!" << std::endl;
			break;
		}			

	}
	img_to_compress = symbols.data();

#ifdef PROFILE_COMPRESSION
	//reset counter
	atomic_store(&g_countCompression, 0);
#endif

	//safety checks to avoid blocksize too large
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		header.blockSize[ii] = std::min(header.blockSize[ii], header.xyzct[ii]);//block size cannot be larger than dimensions

	//set constants
	const uint32_t blockSizeBytes = header.getBlockSizeBytes();
	const uint64_t fLength = header.getImageSizeBytes();			
	const std::uint64_t numBlocks = header.calculateNumBlocks();
	
	header.resizeBlockOffset(numBlocks);


	//number of threads should not be highr than number of blocks (in case somebody set block size too large)
	numThreads = std::min((std::uint64_t) numThreads, numBlocks);

	std::atomic<uint64_t> blockId;//counter shared all workers so each worker thread knows which block to readblockId = 0;
	atomic_store(&blockId, (uint64_t)0);

	int* g_blockSize = new int[numBlocks];//number of bytes (after compression) to be written. If the block has not been compressed yet, it has a -1 value
	int* g_blockThreadId = new int[numBlocks];//indicates which thread wrote the nlock so the writer can find the appropoate circular queue
	for (std::uint64_t ii = 0; ii < numBlocks; ii++)
	{
		g_blockSize[ii] = -1;
		g_blockThreadId[ii] = -1;
	}

	//generate circular queues to exchange blocks between read write
	int numBlocskPerQueue = std::max(numThreads, 5);//total memory = numThreads * blockSizeBytes * numBlocksPerQueue so it should be low. Also, not many blocks should be queued in general
	numBlocskPerQueue = std::min(numBlocskPerQueue, 20);
	numBlocskPerQueue = std::min(numBlocskPerQueue, (int)iDivUp(numBlocks, (std::uint64_t)numThreads));

	//TODO: find the best method to adjust this number automatically
	const uint32_t maxBlockSizeBytesCompressed = maximumBlockSizeCompressedInBytes();
	klb_circular_dequeue** cq = new klb_circular_dequeue*[numThreads];
	for (int ii = 0; ii < numThreads; ii++)
		cq[ii] = new klb_circular_dequeue(maxBlockSizeBytesCompressed, numBlocskPerQueue);


	// start the thread to write
	int errFlagW = 0;
	std::thread writerthread(&klb_imageIO::blockWriter, this, fout, g_blockSize, g_blockThreadId, cq, &errFlagW);

	// start the working threads
	std::vector<std::thread> threads;
	std::vector<int> errFlagVec(numThreads, 0);
	for (int i = 0; i < numThreads; ++i)
	{
		threads.push_back(std::thread(&klb_imageIO::blockCompressor, this, (char*)(img_to_compress), g_blockSize, &blockId, g_blockThreadId, cq[i], i, &(errFlagVec[i])));
	}

	//wait for the workers to finish
	for (auto& t : threads)
	{
		t.join();
	}

	//wait for the writer
	writerthread.join();

	//release memory
	delete[] g_blockSize;
	delete[] g_blockThreadId;
	for (int ii = 0; ii < numThreads; ii++)
		delete cq[ii];
	delete[] cq;

	if (errFlagW != 0)
		return errFlagW;
	for (int ii = 0; ii < numThreads; ii++)
	{
		if (errFlagVec[ii] != 0)
			return errFlagVec[ii];
	}
	// for (int i = 0; i< NUM_PREDICTORS; i++)
	// {
	// 	delete buff_Predictor[i];
	// }
	// delete[] buff_Predictor;
	
#ifdef PROFILE_COMPRESSION
	long long auxChrono = atomic_load(&g_countCompression);
	cout << "Average time spent in compression per thread is =" << auxChrono / numThreads << " ms"<<endl;
#endif
	// auto t2_All = Clock::now();
	// std::cout << "total time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2_All - t1_All).count() << " ms "  << std::endl;
	return 0;//TODO: catch errors from threads (especially opening file)
}

//=================================================

int klb_imageIO::writeImageStackSlices(const char** img, int numThreads)
{

	//redirect standard out
#ifdef DEBUG_PRINT_THREADS
	//cout << "Redirecting stdout for klb_imageIO::writeImage" << endl;
	//freopen("E:/temp/cout_klb_imageIO.txt", "w", stdout);	
#endif

	if (header.xyzct[3] != 1 || header.xyzct[4] != 1)
	{
		printf("Error: number of channels or number of time points must be 1 for this API call\n");
		return 3;
	}

	if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();

	//open output file
	//std::ofstream fout(filenameOut.c_str(), std::ios::binary | std::ios::out);	
	//we do this before calling the thread in case we have problems
	FILE* fout = fopen(filename.c_str(), "wb");//for wahtever reason FILE* is 4X faster than std::ofstream over the network. C interface is much faster than C++ streams
	if (fout == NULL)
	{
		std::cout << "ERROR: file " << filename << " could not be opened" << std::endl;
		return 5;
	}

#ifdef PROFILE_COMPRESSION
	//reset counter
	atomic_store(&g_countCompression, 0);
#endif

	//safety checks to avoid blocksize too large
	for (int ii = 0; ii < KLB_DATA_DIMS; ii++)
		header.blockSize[ii] = std::min(header.blockSize[ii], header.xyzct[ii]);//block size cannot be larger than dimensions

	//set constants
	const uint32_t blockSizeBytes = header.getBlockSizeBytes();
	const uint64_t fLength = header.getImageSizeBytes();
	const std::uint64_t numBlocks = header.calculateNumBlocks();

	header.resizeBlockOffset(numBlocks);


	//number of threads should not be highr than number of blocks (in case somebody set block size too large)
	numThreads = std::min((std::uint64_t) numThreads, numBlocks);

	std::atomic<uint64_t> blockId;//counter shared all workers so each worker thread knows which block to readblockId = 0;
	atomic_store(&blockId, (uint64_t)0);

	int* g_blockSize = new int[numBlocks];//number of bytes (after compression) to be written. If the block has not been compressed yet, it has a -1 value
	int* g_blockThreadId = new int[numBlocks];//indicates which thread wrote the nlock so the writer can find the appropoate circular queue
	for (std::uint64_t ii = 0; ii < numBlocks; ii++)
	{
		g_blockSize[ii] = -1;
		g_blockThreadId[ii] = -1;
	}

	//generate circular queues to exchange blocks between read write
	int numBlocskPerQueue = std::max(numThreads, 5);//total memory = numThreads * blockSizeBytes * numBlocksPerQueue so it should be low. Also, not many blocks should be queued in general
	numBlocskPerQueue = std::min(numBlocskPerQueue, 20);
	numBlocskPerQueue = std::min(numBlocskPerQueue, (int)iDivUp(numBlocks, (std::uint64_t)numThreads));

	//TODO: find the best method to adjust this number automatically
	const uint32_t maxBlockSizeBytesCompressed = maximumBlockSizeCompressedInBytes();
	klb_circular_dequeue** cq = new klb_circular_dequeue*[numThreads];
	for (int ii = 0; ii < numThreads; ii++)
		cq[ii] = new klb_circular_dequeue(maxBlockSizeBytesCompressed, numBlocskPerQueue);


	// start the thread to write
	int errFlagW = 0;
	std::thread writerthread(&klb_imageIO::blockWriter, this, fout, g_blockSize, g_blockThreadId, cq, &errFlagW);

	// start the working threads
	std::vector<std::thread> threads;
	std::vector<int> errFlagVec(numThreads, 0);
	for (int i = 0; i < numThreads; ++i)
	{
		threads.push_back(std::thread(&klb_imageIO::blockCompressorStackSlices, this, img, g_blockSize, &blockId, g_blockThreadId, cq[i], i, &(errFlagVec[i])));
	}

	//wait for the workers to finish
	for (auto& t : threads)
	{
		t.join();
	}

	//wait for the writer
	writerthread.join();

	//release memory
	delete[] g_blockSize;
	delete[] g_blockThreadId;
	for (int ii = 0; ii < numThreads; ii++)
		delete cq[ii];
	delete[] cq;

	if (errFlagW != 0)
		return errFlagW;
	for (int ii = 0; ii < numThreads; ii++)
	{
		if (errFlagVec[ii] != 0)
			return errFlagVec[ii];
	}


#ifdef PROFILE_COMPRESSION
	long long auxChrono = atomic_load(&g_countCompression);
	cout << "Average time spent in compression per thread is =" << auxChrono / numThreads << " ms" << endl;
#endif

	return 0;//TODO: catch errors from threads (especially opening file)
}

//=================================================

int klb_imageIO::readImage(char* img, const klb_ROI* ROI, int numThreads)
{
	std::vector<int16_t> imgA(header.getImageSizePixels());
	if (filename.empty())
	{
		std::cerr << "ERROR: Filename has not been defined. We cannot read image" << std::endl;
		return 3;
	}

	if (header.Nb == 0)//try to read header
	{
		int err = readHeader();
		if (err > 0)
			return err;
		if (header.Nb == 0)//something is wring
		{
			std::cerr << "ERROR: Image to read has not blocks" << std::endl;
			return 2;
		}
	}
	

	if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();
	
	const std::uint64_t numBlocks = header.calculateNumBlocks();

	//number of threads should not be highr than number of blocks (in case somebody set block size too large)
	numThreads = std::min((std::uint64_t) numThreads, numBlocks);	
	
	std::atomic<uint64_t> blockId;
	atomic_store(&blockId, (uint64_t)0);

	// start the working threads
	std::vector<std::thread> threads;
	std::vector<int> errFlagVec(numThreads, 0);
	for (int i = 0; i < numThreads; ++i)
	{
		threads.push_back(std::thread(&klb_imageIO::blockUncompressor, this, (char*)imgA.data(), &blockId, ROI, &(errFlagVec[i])));
	}

	//wait for the workers to finish
	for (auto& t : threads)
		t.join();

	//release memory
	
	for (int ii = 0; ii < numThreads; ii++)
	{
		if (errFlagVec[ii] != 0)
			return errFlagVec[ii];
	}
	switch (LFM_PREDICTOR_WAY)
	{
	case (LFM_PREDICTORS::ANGLE_AND_SPACE):
		unPredictor(imgA.data(), (uint16_t*)img, header.Nnum);
		break;
	case  (LFM_PREDICTORS::ANGLE):
		unPredictor_angle(imgA.data(), (uint16_t*)img, header.Nnum);
		break;
	case  (LFM_PREDICTORS::SPACE):
		unPredictor_space(imgA.data(), (uint16_t*)img, header.Nnum);
		break;			
	default:
		std::cout << "ERROR: The predictors(angle or space or both) hava not selected!" << std::endl;
		break;
	}	
	return 0;//TODO: catch errors from threads (especially opening file)
}

//=================================================

int klb_imageIO::readImageFull(char* imgOut, int numThreads)
{	
	std::vector<int16_t> imgA(header.getImageSizePixels());
	if (filename.empty())
	{
		std::cerr << "ERROR: Filename has not been defined. We cannot read image" << std::endl;
		return 3;
	}

	if (header.Nb == 0)//try to read header
	{
		int err = readHeader();
		if (err > 0)
			return err;
		if (header.Nb == 0)//something is wring
		{
			std::cerr << "ERROR: Image to read has not blocks" << std::endl;
			return 2;
		}
	}

	if (numThreads <= 0)//use maximum available
		numThreads = std::thread::hardware_concurrency();

	const std::uint64_t numBlocks = header.calculateNumBlocks();

	//number of threads should not be highr than number of blocks (in case somebody set block size too large)
	numThreads = std::min((std::uint64_t) numThreads, numBlocks);

	std::atomic<uint64_t>	g_blockId;
	atomic_store(&g_blockId, (uint64_t)0);

	
//#define USE_MEM_BUFFER_READ //uncomment this line to read the compressed file in memory first to have a single read access to the disk

#ifdef USE_MEM_BUFFER_READ		
	//read compressed file from disk into memory
	//open file to read elements	
	FILE* fid = fopen(filename.c_str(), "rb");
	if (fid == NULL)
	{
		cout << "ERROR: readImageFull: reading file " << filename << endl;
		return 3;
	}
	//auto t1 = Clock::now();
	char* imgIn = new char[header.getCompressedFileSizeInBytes()];
	fread(imgIn, 1, header.getCompressedFileSizeInBytes(), fid);
	fclose(fid);
	//auto t2 = Clock::now();
	//std::cout << "=======DEBUGGING:took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms to read file from disk memory by a single thread" << std::endl;
#endif

	// start the working threads
	std::vector<std::thread> threads;
	std::vector<int> errFlagVec(numThreads, 0);
	for (int i = 0; i < numThreads; ++i)
	{
#ifdef USE_MEM_BUFFER_READ		
			threads.push_back(std::thread(&klb_imageIO::blockUncompressorInMem, this, imgOut, &g_blockId, imgIn, &(errFlagVec[i])));
#else
		threads.push_back(std::thread(&klb_imageIO::blockUncompressorImageFull, this, (char*)imgA.data(), &g_blockId, &(errFlagVec[i])));		
#endif
	}

	//wait for the workers to finish
	for (auto& t : threads)
		t.join();

	//release memory
#ifdef USE_MEM_BUFFER_READ		
	delete[] imgIn;
#endif
	for (int ii = 0; ii < numThreads; ii++)
	{
		if (errFlagVec[ii] != 0)
			return errFlagVec[ii];
	}

	switch (LFM_PREDICTOR_WAY)
	{
	case (LFM_PREDICTORS::ANGLE_AND_SPACE):
		unPredictor(imgA.data(), (uint16_t*)imgOut, header.Nnum);
		break;
	case  (LFM_PREDICTORS::ANGLE):
		unPredictor_angle(imgA.data(), (uint16_t*)imgOut, header.Nnum);
		break;
	case  (LFM_PREDICTORS::SPACE):
		unPredictor_space(imgA.data(), (uint16_t*)imgOut, header.Nnum);
		break;			
	default:
		std::cout << "ERROR: The predictors(angle or space or both) hava not selected!" << std::endl;
		break;
	}
	

	return 0;//TODO: catch errors from threads (especially opening file)
}


//======================================================
std::uint32_t klb_imageIO::maximumBlockSizeCompressedInBytes()
{
	uint32_t blockSizeBytes = header.getBlockSizeBytes();

	switch (header.compressionType)
	{
	case KLB_COMPRESSION_TYPE::NONE://no compression
		//nothing to do
		break;
	case KLB_COMPRESSION_TYPE::BZIP2:
	case KLB_COMPRESSION_TYPE::ZLIB:
		/*
			From bzip2 man page: Compression is  always  performed,  even	 if  the  compressed  file  is
			slightly	 larger	 than the original.Files of less than about one hun -
			dred bytes tend to get larger, since the compression  mechanism	has  a
			constant	 overhead  in  the region of 50 bytes.Random data(including
			the output of most file compressors) is coded at about  8.05  bits  per
			byte, giving an expansion of around 0.5%.

			(ngc) testing indicates the man page is wrong.  For encoding random floats,
			need a factor of at least 1.33.  Using 2 just to be safe.
		*/
		blockSizeBytes = ceil(((float)blockSizeBytes) * 2.0f + 50.0f );
		break;
	default:
		std::cout << "ERROR: maximumBlockSizeCompressedInBytes: compression type not implemented" << std::endl;
		blockSizeBytes = 0;
	}

	return blockSizeBytes;
}
