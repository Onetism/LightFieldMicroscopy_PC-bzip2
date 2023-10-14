/*
 * @Author: Onetism_SU
 * @Date: 2021-06-25 09:34:25
 * @LastEditTime: 2021-11-23 16:16:59
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\test\Predictors.cu
 */
 #include "lfm_Predictors.h"
 #include "device_launch_parameters.h"
 #include <math.h>
 #include <stdio.h>
 #include <thrust/sort.h>
 #include <thrust/device_vector.h>
 
 __device__ inline int getNegativeSign(int val)
 {
	 return (val >> 31);
 }
 
 __device__ inline uint32_t symbolize(int value)
 {
	 // map >= 0 to even, < 0 to odd
	 return 2 * abs(value) + getNegativeSign(value);
	 // return 2 * abs(value);
 }
 
 __device__ inline int unsymbolize(uint32_t symbol)
 {
	 int negative = symbol % 2;
	 // map even to >= 0, odd to < 0
	 return (1 - 2 * negative) * ((symbol + negative) / 2);
 }
 
 __global__ void _predictor1_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] ;
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - 1) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1] ;
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y  + x - 1] + (int)in[p * y + x - tileSize]) >> 1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];;
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] -(((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x]  - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1) ;
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y  + x - 1] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1)  + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }
	
	 }
 }
 
 __global__ void _predictor2_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] ;
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - 1) + x];
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - 1) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - 1) + x];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }
	
	 }
 }
 
 __global__ void _predictor3_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] ;
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - 1) + x - 1];
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - tileSize) + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x -1] + (int)in[p * (y - tileSize) + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] +  (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x - tileSize])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - tileSize) + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x - 1] + (int)in[p * (y - tileSize) + x - tileSize])>>1);
				}
			}	
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] -  (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x - 1] + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x -1] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] +  (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x - 1] + (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
			}				 
		 }
		 
	 }
 }
 
 __global__ void _predictor4_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = in[p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]));
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]
														+ (int)in[p * (y - tileSize) + x]) >> 1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]
														+ (int)in[p * y + x - tileSize]) >> 1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * (y - 1) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * y + x - 1]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize]
													+ (int)in[p * (y - 1) + x] + (int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1]) >> 1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]
														+ (int)in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1]
														+ (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * (y - 1) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * y + x - 1]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize]
													+ (int)in[p * (y - 1) + x] + (int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }
	 }
 }
 
 __global__ void _predictor5_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = in[p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ((int)in[p * y + x - 1] + ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1 );
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * (y - tileSize) + x]) >> 1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * y + x - tileSize]) >> 1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * (y - 1) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1));
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] +(((int)in[p * y + x - tileSize]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * y + x - 1]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize])>>1)
													+ (int)in[p * (y - 1) + x] + (((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)) >> 1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * y + x - 1] + ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1 ) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * (y - 1) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1)) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] +(((int)in[p * y + x - tileSize]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * y + x - 1]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize])>>1)
													+ (int)in[p * (y - 1) + x] + (((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }
	 }
 }
 
 __global__ void _predictor6_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = in[p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - 1) + x]  + (((int)in[p * y + x - 1]- (int)in[p * (y - 1) + x - 1])>>1) );
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + ( ((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * (y - tileSize) + x]) >> 1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + ( ((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * y + x - tileSize]) >> 1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * (y - 1) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1));
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] +(((int)in[p * (y - tileSize) + x]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * y + x - 1]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x] - (int)in[p * (y - tileSize) + x - tileSize])>>1)
													+ (int)in[p * y + x - 1] + (((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)) >> 1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * (y - 1) + x]  + (((int)in[p * y + x - 1]- (int)in[p * (y - 1) + x - 1])>>1) ) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + ( ((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + ( ((int)in[p * y + x - 1] - (int)in[p * (y - 1) + x - 1])>>1)
														+ (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * (y - 1) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1)) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - tileSize] +(((int)in[p * (y - tileSize) + x]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1) + (int)in[p * y + x - 1]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x] - (int)in[p * (y - tileSize) + x - tileSize])>>1)
													+ (int)in[p * y + x - 1] + (((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1)) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }

	 }
 }
 
 __global__ void _predictor7_tiles(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int z,int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height)
	 {
		 if (z == 0)
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((int)in[p * y + x] - (int)in[p * (y - 1) + x]);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] ;
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - 1];
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] 
														+ (int)in[p * (y - tileSize) + x - 1] + (int)in[p * (y - tileSize - 1) + x]) >> 2);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x]
														+ (int)in[p * (y - 1) + x - tileSize] + (int)in[p * y + x - tileSize - 1]) >> 2);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * (y - 1) + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * y + x - 1]) >> 1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize -1) + x] + (int)in[p * y + x - tileSize -1]
													+ (int)in[p * (y - 1) + x] + (int)in[p * y + x - 1] ) >> 2);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x])>>1) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * (y - tileSize) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((int)((in[p * y + x - 1] + in[p * (y - tileSize) + x]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] 
														+ (int)in[p * (y - tileSize) + x - 1] + (int)in[p * (y - tileSize - 1) + x]) >> 2) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - 1) + x] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (((int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * y + x - tileSize])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x]
														+ (int)in[p * (y - 1) + x - tileSize] + (int)in[p * y + x - tileSize - 1]) >> 2) + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * (y - 1) + x])>>1) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] + (int)in[p * y + x - 1]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (((((int)in[p * (y - tileSize -1) + x] + (int)in[p * y + x - tileSize -1]
													+ (int)in[p * (y - 1) + x] + (int)in[p * y + x - 1] ) >> 2) + (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }
	 }
 }
 
 
 
 // __global__ void _unPredictor7_tiles(const int16_t*  in,
 // 	volatile uint16_t*  out,
 // 	int p, int width, int height, int tileSize, int it)
 // {
	 
 // 	int th = blockIdx.x * blockDim.x + threadIdx.x;
 // 	int u = it - th;
 // 	int v = th;
 // 	int tx = blockIdx.y * blockDim.y + threadIdx.y;
 // 	int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
 // 	int x = u + tx * tileSize;
 // 	int y = v + ty * tileSize;
 
 // 	if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
 // 		if (tx == 0 && ty == 0)
 // 		{
 // 			if (u == 0) {
 // 				if (v > 0) {	
 // 					out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
 // 				}
 // 				else {
 // 					out[p * y + x] = in[p * y + x];
 // 				}
 // 			}
 // 			else if (v == 0) {
 // 				out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
 // 			}
 // 			else {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x]>>1));
 // 			}
 // 		}
 // 		else if (tx == 0 && ty != 0)
 // 		{
 // 			if (u == 0) {
 // 				if (v > 0) {
 // 					out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x]) >> 1);
 // 				}
 // 				else {
 // 					out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
 // 				}
 // 			}
 // 			else if (v == 0) {
 // 				out[p * y + x] = (int)in[p * y + x] + (int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1);
 // 			}
 // 			else {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] 
 // 									+ (int)in[p * (y - tileSize) + x - 1] + (int)in[p * (y - tileSize - 1) + x]) >> 2);
 // 			}
 // 		}
 // 		else if (tx != 0 && ty == 0)
 // 		{
 // 			if (u == 0) {
 // 				if (v > 0) {
 // 					out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1);
 // 				}
 // 				else {
 // 					out[p * y + x] = in[p * y + x] + out[p * y + x - tileSize];								
 // 				}
 // 			}
 // 			else if (v == 0) {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize]) >> 1);
 // 			}
 // 			else {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x]
 // 								+ (int)in[p * (y - 1) + x - tileSize] + (int)in[p * y + x - tileSize - 1]) >> 2);
 // 			}
 // 		}
 // 		else
 // 		{
 // 			if (u == 0) {
 // 				if (v > 0) {
 // 					out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
 // 						- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * (y - 1) + x]) >> 1);
 // 				}
 // 				else {
 // 					out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
 // 						- (int)out[p * (y - tileSize) + x - tileSize]);
 // 				}
 // 			}
 // 			else if (v == 0) {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
 // 					- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * y + x - 1]) >> 1);
 // 			}
 // 			else {
 // 				out[p * y + x] = (int)in[p * y + x] + (((int)in[p * (y - tileSize -1) + x] + (int)in[p * y + x - tileSize -1]
 // 				+ (int)in[p * (y - 1) + x] + (int)in[p * y + x - 1] ) >> 2);
 // 			}
 // 		}
 // 	}
 // }
 
 void predictor1_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor1_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor2_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor2_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor3_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor3_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor4_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor4_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor5_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor5_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor6_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor6_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor7_tiles_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor7_tiles << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 
 void unPredictor1_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int ty = 0; ty < tilesY; ty++)
	 {
		 for(int tx = 0; tx < tilesX; tx++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + tx * tileSize;
					 int y = v + ty * tileSize;
					 if (x >= width || y >= height)
						 continue;
					if (z == 0)
					{
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - 1) + x];
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1] ;
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y  + x - 1] + (int)out[p * y + x - tileSize]) >> 1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];;
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
						}						
					}
					else
					{
						uint16_t *pre_out = out - width * height;
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y  + x - 1] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}						
					}
				 }					
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor2_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int ty = 0; ty < tilesY; ty++)
	 {
		 for(int tx = 0; tx < tilesX; tx++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + tx * tileSize;
					 int y = v + ty * tileSize;
					 if (x >= width || y >= height)
						 continue;
					if (z == 0)
					{
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - 1) + x];
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - 1) + x];
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - 1) + x];
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
							}
						}	
					}
					else
					{
						uint16_t *pre_out = out - width * height;
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}							
					}
				
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor3_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int ty = 0; ty < tilesY; ty++)
	 {
		 for(int tx = 0; tx < tilesX; tx++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + tx * tileSize;
					 int y = v + ty * tileSize;
					 if (x >= width || y >= height)
						 continue;
					if (z == 0)
					{
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - 1) + x - 1];
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - tileSize) + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x -1] + (int)out[p * (y - tileSize) + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x - tileSize])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - tileSize) + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x - 1] + (int)out[p * (y - tileSize) + x - tileSize])>>1);
							}
						}	
					} 
					else
					{
						uint16_t *pre_out = out - width * height;
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {				
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x - 1] + (int)pre_out[p * y + x])>>1);
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x -1] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x - 1] + (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
						}							
					}
				
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor4_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int i = 0; i < tilesY; i++)
	 {
		 for(int j = 0; j < tilesX; j++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + j * tileSize;
					 int y = v + i * tileSize;
					 if (x >= width || y >= height)
						 continue;
					if (z == 0)
					{
						if (j == 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {	
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = in[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1]));
	
							}
						}
						else if (j == 0 && i != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1]
									+ (int)out[p * (y - tileSize) + x]) >> 1);
							}
						}
						else if (j != 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1);
								}
								else {
									out[p * y + x] = in[p * y + x] + out[p * y + x - tileSize];								
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1]
									+ (int)out[p * y + x - tileSize]) >> 1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * (y - 1) + x]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize]);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * y + x - 1]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize] - (int)out[p * (y - tileSize) + x - tileSize]
									+ (int)out[p * (y - 1) + x] + (int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1]) >> 1);
							}
						}
					}
					else
					{
						uint16_t *pre_out = out - width * height;
						if (j == 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {	
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x]  + (int)pre_out[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])) + (int)pre_out[p * y + x])>>1);
	
							}
						}
						else if (j == 0 && i != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1]
									+ (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (j != 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);								
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1]
									+ (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * (y - 1) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize]) + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * y + x - 1]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize] - (int)out[p * (y - tileSize) + x - tileSize]
									+ (int)out[p * (y - 1) + x] + (int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}						
					}
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor5_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int ty = 0; ty < tilesY; ty++)
	 {
		 for(int tx = 0; tx < tilesX; tx++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + tx * tileSize;
					 int y = v + ty * tileSize;
					 if (x >= width || y >= height)
						 continue;
					 if (z == 0)
					 {
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = in[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + ((int)out[p * y + x - 1] + ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1 );
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + ( ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * (y - tileSize) + x]) >> 1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + ( ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * y + x - tileSize]) >> 1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
																		- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * (y - 1) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
																						- (int)out[p * (y - tileSize) + x - tileSize])>>1));
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] +(((int)out[p * y + x - tileSize]
																- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * y + x - 1]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]- (int)out[p * (y - tileSize) + x - tileSize])>>1)
																+ (int)out[p * (y - 1) + x] + (((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)) >> 1);
							}
						}
					 }
					 else
					 {
						uint16_t *pre_out = out - width * height;
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * y + x - 1] + ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1 ) + (int)pre_out[p * y + x])>>1);
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + ( ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + ( ((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
																		- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * (y - 1) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
																						- (int)out[p * (y - tileSize) + x - tileSize])>>1)) + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] +(((int)out[p * y + x - tileSize]
																- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * y + x - 1]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]- (int)out[p * (y - tileSize) + x - tileSize])>>1)
																+ (int)out[p * (y - 1) + x] + (((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}						 
					 }
				
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor6_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 // dim3 dimGrid(((tileSize - 1) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(int16_t);
 
	 for (int ty = 0; ty < tilesY; ty++)
	 {
		 for(int tx = 0; tx < tilesX; tx++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + tx * tileSize;
					 int y = v + ty * tileSize;
					 if (x >= width || y >= height)
						 continue;
					 if (z == 0)
					 {
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = (int)in[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - 1) + x]  + (((int)out[p * y + x - 1]- (int)out[p * (y - 1) + x - 1])>>1) );
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)((out[p * y + x - 1] + (int)out[p * (y - tileSize) + x]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + ( ((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * (y - tileSize) + x]) >> 1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + ( ((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * y + x - tileSize]) >> 1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
																		- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * (y - 1) + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
																						- (int)out[p * (y - tileSize) + x - tileSize])>>1));
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] +(((int)out[p * (y - tileSize) + x]
																- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * y + x - 1]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x] - (int)out[p * (y - tileSize) + x - tileSize])>>1)
																+ (int)out[p * y + x - 1] + (((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)) >> 1);
							}
						}	
					 }
					 else
					 {
						uint16_t *pre_out = out - width * height;
						if (tx == 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									int temp = (z - 1) * width * height;
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * (y - 1) + x]  + (((int)out[p * y + x - 1]- (int)out[p * (y - 1) + x - 1])>>1) ) + (int)pre_out[p * y + x])>>1);
				
							}
						}
						else if (tx == 0 && ty != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)((out[p * y + x - 1] + (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + ( ((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (tx != 0 && ty == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize])>>1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + ( ((int)out[p * y + x - 1] - (int)out[p * (y - 1) + x - 1])>>1)
																	+ (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
																		- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * (y - 1) + x])>>1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
																						- (int)out[p * (y - tileSize) + x - tileSize])>>1)) + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - tileSize] +(((int)out[p * (y - tileSize) + x]
																- (int)out[p * (y - tileSize) + x - tileSize])>>1) + (int)out[p * y + x - 1]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x] - (int)out[p * (y - tileSize) + x - tileSize])>>1)
																+ (int)out[p * y + x - 1] + (((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1)) >> 1) + (int)pre_out[p * y + x])>>1);
							}
						}							 
					 }
			
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor7_tiles(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 int p = pitch / sizeof(int16_t);
 
 
	 for (int i = 0; i < tilesY; i++)
	 {
		 for(int j = 0; j < tilesX; j++)
		 {
			 for(int v = 0; v < tileSize; v++)
			 {
				 for(int u = 0; u < tileSize; u++)
				 {
					 int x = u + j * tileSize;
					 int y = v + i * tileSize;
					 if (x >= width || y >= height)
						 continue;
					 if (z == 0)
					 {
						if (j == 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {	
									out[p * y + x] = ((int)in[p * y + x] + (int)out[p * (y - 1) + x]);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - 1];
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] )>>1);
	
							}
						}
						else if (j == 0 && i != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] 
													+ (int)out[p * (y - tileSize) + x - 1] + (int)out[p * (y - tileSize - 1) + x]) >> 2);
							}
						}
						else if (j != 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1);
								}
								else {
									out[p * y + x] = in[p * y + x] + out[p * y + x - tileSize];								
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x]
								+ (int)out[p * (y - 1) + x - tileSize] + (int)out[p * y + x - tileSize - 1]) >> 2);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * (y - 1) + x]) >> 1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize]);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * y + x - 1]) >> 1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize -1) + x] + (int)out[p * y + x - tileSize -1]
								+ (int)out[p * (y - 1) + x] + (int)out[p * y + x - 1] ) >> 2);
							}
						}	
					 }
					 else
					 {
						uint16_t *pre_out = out - width * height;
						if (j == 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {	
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - 1) + x] + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (int)pre_out[p * y + x] ;
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - 1] + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] )>>1) + (int)pre_out[p * y + x])>>1);
	
							}
						}
						else if (j == 0 && i != 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * (y - tileSize) + x] + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((int)((out[p * y + x - 1] + out[p * (y - tileSize) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x] 
													+ (int)out[p * (y - tileSize) + x - 1] + (int)out[p * (y - tileSize - 1) + x]) >> 2) + (int)pre_out[p * y + x])>>1);
							}
						}
						else if (j != 0 && i == 0)
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - 1) + x] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + (((int)out[p * y + x - tileSize] + (int)pre_out[p * y + x])>>1);								
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * y + x - tileSize]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * y + x - 1] + (int)out[p * (y - 1) + x]
								+ (int)out[p * (y - 1) + x - tileSize] + (int)out[p * y + x - tileSize - 1]) >> 2) + (int)pre_out[p * y + x])>>1);
							}
						}
						else
						{
							if (u == 0) {
								if (v > 0) {
									out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * (y - 1) + x]) >> 1) + (int)pre_out[p * y + x])>>1);
								}
								else {
									out[p * y + x] = (int)in[p * y + x] + ((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
										- (int)out[p * (y - tileSize) + x - tileSize]) + (int)pre_out[p * y + x])>>1);
								}
							}
							else if (v == 0) {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									- (int)out[p * (y - tileSize) + x - tileSize] + (int)out[p * y + x - 1]) >> 1) + (int)pre_out[p * y + x])>>1);
							}
							else {
								out[p * y + x] = (int)in[p * y + x] + (((((int)out[p * (y - tileSize -1) + x] + (int)out[p * y + x - tileSize -1]
								+ (int)out[p * (y - 1) + x] + (int)out[p * y + x - 1] ) >> 2) + (int)pre_out[p * y + x])>>1);
							}
						}							 
					 }
		
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 
 
 __global__ void _realignKernel(const uint16_t* __restrict__ in,
	 int16_t* __restrict__ out,
	 int p, int width, int height, int tileSize)
 {
	 int i = blockIdx.x * blockDim.x + threadIdx.x;
	 int u = (i % tileSize);						// +1 to omit first column
	 int v = (i / tileSize);						// +1 to omit first row
 
	 int tx = blockIdx.y * blockDim.y + threadIdx.y;
	 int ty = blockIdx.z * blockDim.z + threadIdx.z;
 
	 //int offset = ty*p*height + tx*width;
 
	 int offset_width = (int)ceilf((float)width / (float)tileSize);
	 int offset_height = (int)ceilf((float)height / (float)tileSize);
 
	 int x = u + tx * tileSize;
	 int y = v + ty * tileSize;
 
	 if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) 
	 { 
		 out[v * offset_width * offset_height + u * offset_width + tx] = in[x * p + y];
	 }
 }
 
 
 __global__ void symbolizeKernel(
	 uint16_t* __restrict__ pSymbols, const int16_t* __restrict__ pData,
	 uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ,
	 uint32_t rowPitchSrc, uint32_t slicePitchSrc)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	 uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
 
	 if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
 
	 uint32_t indexSrc = x + y * rowPitchSrc + z * slicePitchSrc;
	 uint32_t indexDst = x + y * sizeX + z * sizeX * sizeY;
 
	 pSymbols[indexDst] = symbolize(pData[indexSrc]);
 }
 
 __global__ void unsymbolizeKernel(
	 int16_t* __restrict__ pData, const uint16_t* __restrict__ pSymbols,
	 uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ,
	 uint32_t rowPitchDst, uint32_t slicePitchDst)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	 uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
 
	 if (x >= sizeX || y >= sizeY || z >= sizeZ) return;
 
	 uint32_t indexSrc = x + y * sizeX + z * sizeX * sizeY;
	 uint32_t indexDst = x + y * rowPitchDst + z * slicePitchDst;
 
	 pData[indexDst] = unsymbolize(pSymbols[indexSrc]);
	//  if(indexDst == 0)
	//  	printf("%d",  pData[indexDst]);
 }
 
 void symbolize_GPU(uint16_t* dpSymbols, const int16_t* dpData, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t rowPitchSrc, uint32_t slicePitchSrc)
 {
	 if (rowPitchSrc == 0) rowPitchSrc = sizeX;
	 if (slicePitchSrc == 0) slicePitchSrc = rowPitchSrc * sizeY;
 
	 dim3 blockSize(64, 4, 1);
	 uint32_t blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
	 uint32_t blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
 
	 symbolizeKernel << <blockCount, blockSize >> > (dpSymbols, dpData, sizeX, sizeY, sizeZ, rowPitchSrc, slicePitchSrc);
 
 }
 
 void unsymbolize_GPU(int16_t* dpData, const uint16_t* dpSymbols, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, uint32_t rowPitchDst, uint32_t slicePitchDst)
 {
	 if (rowPitchDst == 0) rowPitchDst = sizeX;
 
	 dim3 blockSize(64, 4, 1);
	 uint32_t blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
	 uint32_t blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
 
	 unsymbolizeKernel << <blockCount, blockSize >> > (dpData, dpSymbols, sizeX, sizeY, sizeZ, rowPitchDst, slicePitchDst);
 }
 
 __global__ void static_bwt_Kernel(
	 uint8_t* __restrict__ pSymbols, unsigned long long int* __restrict__ pData,
	 uint32_t size)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 // uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	 // uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
 
	 if (x >= size) return;
 
	 uint32_t indexDst = x ;
	 atomicAdd(&pData[(pSymbols[indexDst]<<8)|(pSymbols[indexDst+1])], 1);
	 // pSymbols[indexDst] = symbolize(pData[indexSrc]);
 }
 void static_bwt_GPU(uint8_t* dpSymbols, unsigned long long int* dpData, uint32_t size)
 {
	 dim3 blockSize(1024, 1, 1);
	 uint32_t blockCountX = (size + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = 1;
	 uint32_t blockCountZ = 1;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
 
	 static_bwt_Kernel << <blockCount, blockSize >> > (dpSymbols, dpData, size);
 }
 
 __global__ void staticKernel(
	 uint8_t* __restrict__ pSymbols, unsigned long long int* __restrict__ pData,
	 uint32_t size)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 // uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	 // uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
 
	 if (x >= size) return;
 
	 uint32_t indexDst = x ;
	 atomicAdd(&pData[pSymbols[indexDst]], 1);
	 // pSymbols[indexDst] = symbolize(pData[indexSrc]);
 }
 void static_GPU(uint8_t* dpSymbols, unsigned long long int* dpData, uint32_t size)
 {
	 dim3 blockSize(1024, 1, 1);
	 uint32_t blockCountX = (size + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = 1;
	 uint32_t blockCountZ = 1;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
 
	 staticKernel << <blockCount, blockSize >> > (dpSymbols, dpData, size);
 }
 
 __global__ void bwtKernel(
	 uint8_t* __restrict__ pIn, uint8_t* __restrict__ pFirst,uint8_t* __restrict__ pLast,
	 uint32_t size)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
 
	 if (x >= size) return;
 
	 uint32_t indexDst = x ;
	 if (indexDst == 0)
	 {
		 pFirst[size] = 0;
		 pLast[0] = 0;
	 }
	 pFirst[indexDst] = pIn[indexDst];
	 pLast[indexDst+1] = pIn[indexDst];
 }
 
 void bwt_GPU(uint8_t* dpIn, uint8_t* dpFirst, uint8_t* dpLast, uint32_t size)
 {
	 dim3 blockSize(1024, 1, 1);
	 uint32_t blockCountX = (size + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = 1;
	 uint32_t blockCountZ = 1;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
 
	 bwtKernel << <blockCount, blockSize >> > (dpIn, dpFirst, dpLast, size);
 
	 thrust::device_ptr<unsigned char> dev_data_ptr(dpLast);
	 thrust::device_ptr<unsigned char> dev_keys_ptr(dpFirst);			
	 thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + size + 1, dev_data_ptr);
 }
 
 void sort_GPU(unsigned long long int* dpIn, unsigned long long int size)
 {
	 thrust::device_ptr<unsigned long long int> dev_ptr(dpIn);
	 thrust::stable_sort(dev_ptr, dev_ptr + size, thrust::greater<unsigned long long int>());
 
 }
 
 __global__ void sum_bwt_Kernel(unsigned long long int* __restrict__ pData, float* __restrict__ pEntropy, uint32_t size)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (x >= 65535) return;
 
	 uint32_t indexDst = x ;
 
	 if (pData[indexDst] == 0)
	 {
		 return;
	 }
	 float P = (float)pData[indexDst]/(float)size;
	 pEntropy[indexDst] = -1 * P * log(P);
 }
 void sum_bwt_GPU(unsigned long long int* dpIn, float* dpEntropy, unsigned long long int size, float* entropy)
 {
	 dim3 blockSize(1024, 1, 1);
	 uint32_t blockCountX = (65535 + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = 1;
	 uint32_t blockCountZ = 1;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
	 sum_bwt_Kernel << <blockCount, blockSize >> > (dpIn, dpEntropy, size);
 
	 thrust::device_ptr<float> dev_sum(dpEntropy);
	 *entropy += thrust::reduce(dev_sum, dev_sum + 65535, (float) 0, thrust::plus<float>());
 }
 
 __global__ void sum_Kernel(unsigned long long int* __restrict__ pData, uint32_t size)
 {
	 uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	 
	 if (x >= 65535) return;
 
	 uint32_t indexDst = x ;
 
	 if (pData[indexDst] == 0)
	 {
		 return;
	 }
	 pData[indexDst] = pData[indexDst] - 1 + indexDst;
 }
 void sum_GPU(unsigned long long int* dpIn, unsigned long long int size, unsigned long long int* bits)
 {
	 dim3 blockSize(1024, 1, 1);
	 uint32_t blockCountX = (65535 + blockSize.x - 1) / blockSize.x;
	 uint32_t blockCountY = 1;
	 uint32_t blockCountZ = 1;
	 dim3 blockCount(blockCountX, blockCountY, blockCountZ);
	 sum_Kernel << <blockCount, blockSize >> > (dpIn, size);
 
	 thrust::device_ptr<unsigned long long int> dev_sum(dpIn);
	 *bits += thrust::reduce(dev_sum, dev_sum + 65535, (float) 0, thrust::plus<unsigned long long int>());
 }