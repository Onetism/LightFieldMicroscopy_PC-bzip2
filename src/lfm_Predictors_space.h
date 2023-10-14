/*
 * @Author: Onetism_SU
 * @Date: 2021-07-21 20:57:32
 * @LastEditTime: 2021-08-16 08:10:09
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\src\lfm_Predictors.h
 */
#ifndef __LFM_PREDICTORS_SPACE_H__
#define __LFM_PREDICTORS_SPACE_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

void predictor1_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor2_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor3_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor4_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor5_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor6_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor7_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);


void unPredictor1_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor2_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor3_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor4_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor5_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor6_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor7_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);

#endif