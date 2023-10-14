/*
 * @Author: Onetism_SU
 * @Date: 2021-07-21 20:57:32
 * @LastEditTime: 2021-11-22 23:39:43
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\src\lfm_Predictors.h
 */
#ifndef __LFM_PREDICTORS_H__
#define __LFM_PREDICTORS_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

void predictor1_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor2_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor3_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor4_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor5_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor6_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor7_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);


void unPredictor1_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor2_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor3_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor4_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor5_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor6_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor7_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);



void sum_bwt_GPU(unsigned long long int* dpIn, float* dpEntropy, unsigned long long int size, float* entropy);
void sum_GPU(unsigned long long int* dpIn, unsigned long long int size, unsigned long long int* bits);
void sort_GPU(unsigned long long int* dpIn, unsigned long long int size);
void static_bwt_GPU(uint8_t* dpSymbols, unsigned long long int* dpData, uint32_t size);
void static_GPU(uint8_t* dpSymbols, unsigned long long int* dpData, uint32_t size);
void symbolize_GPU(uint16_t* dpSymbols, const int16_t* dpData, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t rowPitchSrc, uint32_t slicePitchSrc);
void unsymbolize_GPU(int16_t* dpData, const uint16_t* dpSymbols, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t rowPitchDst, uint32_t slicePitchDst);

void bwt_GPU(uint8_t* dpIn, uint8_t* dpFirst, uint8_t* dpLast, uint32_t size);
#endif