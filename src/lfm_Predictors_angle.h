/*
 * @Author: Onetism_SU
 * @Date: 2021-07-21 20:57:32
 * @LastEditTime: 2021-08-16 08:10:09
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\src\lfm_Predictors.h
 */
#ifndef __LFM_PREDICTORS_ANGLE_H__
#define __LFM_PREDICTORS_ANGLE_H__

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

void predictor1_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor2_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor3_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor4_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor5_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor6_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);
void predictor7_angle_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize);

void unPredictor1_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor2_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor3_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor4_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor5_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor6_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);
void unPredictor7_angle(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize);

#endif