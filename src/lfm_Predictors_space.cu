/*
 * @Author: Onetism_SU
 * @Date: 2021-06-25 09:34:25
 * @LastEditTime: 2021-12-22 23:17:57
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\test\Predictors.cu
 */
 #include "lfm_Predictors_space.h"
 #include "device_launch_parameters.h"
 #include <math.h>
 #include <stdio.h>
 #include <thrust/sort.h>
 #include <thrust/device_vector.h>
 
 
 __global__ void _predictor1_space(const uint16_t* __restrict__ in,
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
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x] ;
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
		 }

		 else 
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
		 }


	
	 }
 }
 
 __global__ void _predictor2_space(const uint16_t* __restrict__ in,
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
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
			}	
		 }

		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - 1) + x] + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}	
		 }

	 }
 }
 
 __global__ void _predictor3_space(const uint16_t* __restrict__ in,
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
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] -  (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] -  (int)in[p * (y - tileSize) + x - tileSize];
				}
			}			 
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);;
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);;
				}
				else {				
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - 1) + x - 1] + (int)in[width * height + p * y + x])>>1);;
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] -  (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] -  (int)in[p * (y - tileSize) + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
		 }
	
	 }
 }
 
 __global__ void _predictor4_space(const uint16_t* __restrict__ in,
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
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] );
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] );
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize] );
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])) 
									+ (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] ) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] )+ (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize]) 
															+ (int)in[width * height + p * y + x])>>1);
				}
			}
		 }

 
	 }
 }
 
 __global__ void _predictor5_space(const uint16_t* __restrict__ in,
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
					out[p * y + x] = (int)in[p * y + x] - ((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1 ) );
	
				}
			}
			else if (tx == 0 && ty != 0)
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] -  (int)in[p * y + x - tileSize];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (( (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] )>>1));
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + ( ((int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize] )>>1));
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] +( ((int)in[p * y + x - tileSize]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1 ));
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize])>>1));
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * y + x - 1] + ( ((int)in[p * (y - 1) + x] - (int)in[p * (y - 1) + x - 1])>>1 ) )
												+ (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] -  (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (( (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] )>>1)) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + ( ((int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize] )>>1)) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] +( ((int)in[p * y + x - tileSize]
													- (int)in[p * (y - tileSize) + x - tileSize])>>1)) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (((int)in[p * y + x - tileSize]- (int)in[p * (y - tileSize) + x - tileSize])>>1))
													+ (int)in[width * height + p * y + x])>>1);
				}
			}
		 }

	 }
 }
 
 __global__ void _predictor6_space(const uint16_t* __restrict__ in,
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
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1));
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1) );
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (( (int)in[p * (y - tileSize) + x]
													- (int)in[p * (y - tileSize) + x - tileSize] )>>1 ));
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x] - (int)in[p * (y - tileSize) + x - tileSize] )>>1));
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - 1) + x]  + (((int)in[p * y + x - 1]- (int)in[p * (y - 1) + x - 1])>>1)) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
															- (int)in[p * (y - tileSize) + x - tileSize])>>1)) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x]
																			- (int)in[p * (y - tileSize) + x - tileSize])>>1) ) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (( (int)in[p * (y - tileSize) + x]
													- (int)in[p * (y - tileSize) + x - tileSize] )>>1 )) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * y + x - tileSize] + (((int)in[p * (y - tileSize) + x] - (int)in[p * (y - tileSize) + x - tileSize] )>>1))
													+ (int)in[width * height + p * y + x])>>1);
				}
			}			 
		 }

	 }
 }
 
 __global__ void _predictor7_space(const uint16_t* __restrict__ in,
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
					out[p * y + x] = (int)in[p * y + x] - ( ((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x])>>1 );
	
				}
			}
			else if (tx == 0 && ty != 0)
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * (y - tileSize) + x];
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
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - (int)in[p * y + x - tileSize];
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] );
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] );
				}
				else {
					out[p * y + x] = (int)in[p * y + x] - ( ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]) >> 1);
				}
			}
		 }
		 else
		 {
			if (tx == 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = ((((int)in[p * y + x] - (int)in[p * (y - 1) + x]) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (int)in[p * y + x] - (int)in[width * height + p * y + x];
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - 1] + (int)in[width * height + p * y + x])>>1);
				}
				else {				
					out[p * y + x] = (((int)in[p * y + x] - ( ((int)in[p * y + x - 1] + (int)in[p * (y - 1) + x])>>1 ) + (int)in[width * height + p * y + x])>>1);
	
				}
			}
			else if (tx == 0 && ty != 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * (y - tileSize) + x] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else if (tx != 0 && ty == 0)
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - (int)in[p * y + x - tileSize] + (int)in[width * height + p * y + x])>>1);
				}
			}
			else
			{
				if (u == 0) {
					if (v > 0) {
						out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] ) + (int)in[width * height + p * y + x])>>1);
					}
					else {
						out[p * y + x] = (((int)in[p * y + x] - ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
																			- (int)in[p * (y - tileSize) + x - tileSize]) + (int)in[width * height + p * y + x])>>1);
					}
				}
				else if (v == 0) {
					out[p * y + x] = (((int)in[p * y + x] - ( (int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]
															- (int)in[p * (y - tileSize) + x - tileSize] ) + (int)in[width * height + p * y + x])>>1);
				}
				else {
					out[p * y + x] = (((int)in[p * y + x] - ( ((int)in[p * (y - tileSize) + x] + (int)in[p * y + x - tileSize]) >> 1) + (int)in[width * height + p * y + x])>>1);
				}
			}
		 }

	 }
 }
 
 

 void predictor1_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor1_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor2_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor2_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor3_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor3_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor4_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor4_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor5_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor5_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor6_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor6_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }
 
 void predictor7_space_GPU(const uint16_t* in, int16_t* out, int pitch, int width, int height, int z, int tileSize)
 {
	 //const int num_tiles = 4;
	 dim3 dimBlock(1024, 1, 1);
 
	 //int tile_width = (width + num_tiles - 1) / num_tiles; // divide by num_tiles, round up
	 //int tile_height = (height + num_tiles - 1) / num_tiles;
 
	 int tilesX = (width + tileSize - 1) / tileSize;
	 int tilesY = (height + tileSize - 1) / tileSize;
 
	 dim3 dimGrid(((tileSize * tileSize) + dimBlock.x - 1) / dimBlock.x, tilesX, tilesY);
	 int p = pitch / sizeof(uint16_t);
 
	 _predictor7_space << <dimGrid, dimBlock >> > (in, out, p, width, height, z, tileSize);
 
	 return;
 }

 
 void unPredictor1_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
				 }					
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor2_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
					 }					
				 }
			 }
		 }
	 }
	 return;
 }
 
 void unPredictor3_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
					 else
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x - tileSize];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x - tileSize];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x - tileSize];
						 }
					 }					
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor4_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
					 }
					 else if (j != 0 && i == 0)
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
							 }
							 else {
								 out[p * y + x] = in[p * y + x] + (int)out[p * y + x - tileSize];								
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
					 else
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
								 	- (int)out[p * (y - tileSize) + x - tileSize]);
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									 - (int)out[p * (y - tileSize) + x - tileSize]);
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
							 		- (int)out[p * (y - tileSize) + x - tileSize]);
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									 - (int)out[p * (y - tileSize) + x - tileSize]);
						 }
					 }			
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor5_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * y + x - 1] + (((int)out[p * (y - 1) + x] - (int)out[p * (y - 1) + x - 1])>>1) );
			 
						 }
					 }
					 else if (tx == 0 && ty != 0)
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
					 else
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
													 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
																					 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
													 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (((int)out[p * y + x - tileSize]
													 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
						 }
					 }				
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor6_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
							 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - 1) + x]  +( ((int)out[p * y + x - 1]- (int)out[p * (y - 1) + x - 1])>>1 ));
			 
						 }
					 }
					 else if (tx == 0 && ty != 0)
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
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
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
					 else
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
								 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
																					 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] +((int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
							 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + ( (int)out[p * y + x - tileSize] + (((int)out[p * (y - tileSize) + x]
							 - (int)out[p * (y - tileSize) + x - tileSize])>>1));
						 }
					 }				
				 }
			 }
		 }
	 }
 
	 return;
 }
 
 void unPredictor7_space(const int16_t* in, uint16_t* out, int pitch, int width, int height, int z, int tileSize)
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
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * (y - tileSize) + x];
						 }
					 }
					 else if (j != 0 && i == 0)
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
							 }
							 else {
								 out[p * y + x] = in[p * y + x] + (int)out[p * y + x - tileSize];								
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + (int)out[p * y + x - tileSize];
						 }
					 }
					 else
					 {
						 if (u == 0) {
							 if (v > 0) {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
																 - (int)out[p * (y - tileSize) + x - tileSize] );
							 }
							 else {
								 out[p * y + x] = (int)in[p * y + x] + ((int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
									 							- (int)out[p * (y - tileSize) + x - tileSize]);
							 }
						 }
						 else if (v == 0) {
							 out[p * y + x] = (int)in[p * y + x] + ( (int)out[p * (y - tileSize) + x] + (int)out[p * y + x - tileSize]
														 - (int)out[p * (y - tileSize) + x - tileSize] );
						 }
						 else {
							 out[p * y + x] = (int)in[p * y + x] + ( ((int)out[p * (y - tileSize ) + x] + (int)out[p * y + x - tileSize ]) >> 1 );
						 }
					 }			
				 }
			 }
		 }
	 }
 
	 return;
 }

 
 
