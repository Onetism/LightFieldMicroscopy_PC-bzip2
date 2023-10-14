/*
 * @Author: your name
 * @Date: 2021-07-21 09:57:30
 * @LastEditTime: 2021-12-22 20:14:40
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \keller-lab-block-filetype\src\common.h
 */
#ifndef __KLB_IMAGE_COMMON_H__
#define __KLB_IMAGE_COMMON_H__

typedef float  float32_t;
typedef double float64_t;

#define KLB_DATA_DIMS (5) //our images at the most have 5 dimensions: x,y,z, c, t
#define KLB_METADATA_SIZE (256) //number of bytes in metadata
#define KLB_DEFAULT_HEADER_VERSION (0) //def
#define NUM_PREDICTORS (8)
#define LFM_PREDICTOR_WAY (0)

// Following mylib conventions here are the data types
enum KLB_DATA_TYPE
{
	UINT8_TYPE = 0,
	UINT16_TYPE = 1,
	UINT32_TYPE = 2,
	UINT64_TYPE = 3,
	INT8_TYPE = 4,
	INT16_TYPE = 5,
	INT32_TYPE = 6,
	INT64_TYPE = 7,
	FLOAT32_TYPE = 8,
	FLOAT64_TYPE = 9
};

//Compression type look up table (add to the list if you use a different one)
//To add more compression types just add it here and look for 
enum KLB_COMPRESSION_TYPE
{
	NONE = 0,
	BZIP2 = 1,
	ZLIB = 2
};

enum LFM_PREDICTORS
{
	ANGLE_AND_SPACE = 0,
	ANGLE = 1,
	SPACE = 2
};

enum LFM_PREDICTORS_TYPE
{
	NO_PREIDICTORS = 0,
	PREIDCTORS_A = 1,
	PREIDCTORS_B = 2,
	PREIDCTORS_C = 3,
	PREIDCTORS_APB_DC = 4,
	PREIDCTORS_A_BDC_Div2 = 5,
	PREIDCTORS_B_ADC_Div2 = 6,
	PREIDCTORS_APB_Div2 = 7,
	PREIDCTORS_APB_Div2_Exten = 8
};

#endif