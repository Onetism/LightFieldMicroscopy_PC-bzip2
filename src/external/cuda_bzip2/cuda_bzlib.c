
/*-------------------------------------------------------------*/
/*--- Library top-level functions.                          ---*/
/*---                                               bzlib.c ---*/
/*-------------------------------------------------------------*/

/* ------------------------------------------------------------------
   This file is part of bzip2/libbzip2, a program and library for
   lossless, block-sorting data compression.

   bzip2/libbzip2 version 1.0.6 of 6 September 2010
   Copyright (C) 1996-2010 Julian Seward <jseward@bzip.org>

   Please read the WARNING, DISCLAIMER and PATENTS sections in the 
   README file.

   This program is released under the terms of the license contained
   in the file LICENSE.
   ------------------------------------------------------------------ */

/* CHANGES
   0.9.0    -- original version.
   0.9.0a/b -- no changes in this file.
   0.9.0c   -- made zero-length BZ_FLUSH work correctly in bzCompress().
     fixed bzWrite/bzRead to ignore zero-length requests.
     fixed bzread to correctly handle read requests after EOF.
     wrong parameter order in call to bzDecompressInit in
     bzBuffToBuffDecompress.  Fixed.
*/

#include "cuda_bzlib_private.h"

/*---------------------------------------------------*/
/*--- Compression stuff                           ---*/
/*---------------------------------------------------*/


/*---------------------------------------------------*/
#ifndef BZ_NO_STDIO
void cuda_BZ2_bz__AssertH__fail ( int errcode )
{
   fprintf(stderr, 
      "\n\nbzip2/libbzip2: internal error number %d.\n"
      "This is a bug in bzip2/libbzip2, %s.\n"
      "Please report it to me at: jseward@bzip.org.  If this happened\n"
      "when you were using some program which uses libbzip2 as a\n"
      "component, you should also report this bug to the author(s)\n"
      "of that program.  Please make an effort to report this bug;\n"
      "timely and accurate bug reports eventually lead to higher\n"
      "quality software.  Thanks.  Julian Seward, 10 December 2007.\n\n",
      errcode,
      cuda_BZ2_bzlibVersion()
   );

   if (errcode == 1007) {
   fprintf(stderr,
      "\n*** A special note about internal error number 1007 ***\n"
      "\n"
      "Experience suggests that a common cause of i.e. 1007\n"
      "is unreliable memory or other hardware.  The 1007 assertion\n"
      "just happens to cross-check the results of huge numbers of\n"
      "memory reads/writes, and so acts (unintendedly) as a stress\n"
      "test of your memory system.\n"
      "\n"
      "I suggest the following: try compressing the file again,\n"
      "possibly monitoring progress in detail with the -vv flag.\n"
      "\n"
      "* If the error cannot be reproduced, and/or happens at different\n"
      "  points in compression, you may have a flaky memory system.\n"
      "  Try a memory-test program.  I have used Memtest86\n"
      "  (www.memtest86.com).  At the time of writing it is free (GPLd).\n"
      "  Memtest86 tests memory much more thorougly than your BIOSs\n"
      "  power-on test, and may find failures that the BIOS doesn't.\n"
      "\n"
      "* If the error can be repeatably reproduced, this is a bug in\n"
      "  bzip2, and I would very much like to hear about it.  Please\n"
      "  let me know, and, ideally, save a copy of the file causing the\n"
      "  problem -- without which I will be unable to investigate it.\n"
      "\n"
   );
   }

   exit(3);
}
#endif


/*---------------------------------------------------*/
static
int bz_config_ok ( void )
{
   if (sizeof(int)   != 4) return 0;
   if (sizeof(short) != 2) return 0;
   if (sizeof(char)  != 1) return 0;
   return 1;
}


/*---------------------------------------------------*/
static
void* default_bzalloc ( void* opaque, Int32 items, Int32 size )
{
   void* v = malloc ( items * size );
   return v;
}

static
void default_bzfree ( void* opaque, void* addr )
{
   if (addr != NULL) free ( addr );
}


/*---------------------------------------------------*/
static
void prepare_new_block ( EState* s )
{
   Int32 i;
   s->nblock = 0;
   s->numZ = 0;
   s->state_out_pos = 0;
   BZ_INITIALISE_CRC ( s->blockCRC );
   for (i = 0; i < 256; i++) s->inUse[i] = False;
   s->blockNo++;
}

/*---------------------------------------------------*/
static
void init_RL ( EState* s )
{
   s->state_in_ch  = 256;
   s->state_in_len = 0;
}


static
Bool isempty_RL ( EState* s )
{
   if (s->state_in_ch < 256 && s->state_in_len > 0)
      return False; else
      return True;
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzCompressInit) 
                    (cuda_bz_stream* strm, 
                     int        blockSize100k,
                     int        verbosity,
                     int        workFactor,
		     int        numThreads)
{
   Int32   n;
   EState* s;

   if (!bz_config_ok()) return BZ_CONFIG_ERROR;

   if (strm == NULL || 
       blockSize100k < 1 || blockSize100k > 150 ||
       workFactor < 0 || workFactor > 250)
     return BZ_PARAM_ERROR;

   if (workFactor == 0) workFactor = 30;
   if (strm->bzalloc == NULL) strm->bzalloc = default_bzalloc;
   if (strm->bzfree == NULL) strm->bzfree = default_bzfree;
   
   // number of filled EState's is initialized to 0 at the start of compression
   strm->state_fill_count = 0;

   s = (EState *) BZALLOC( sizeof(EState) );
   if (s == NULL) return BZ_MEM_ERROR;
   s->strm = strm;

   s->arr1 = NULL;
   s->arr2 = NULL;
   s->ftab = NULL;

   n       = 100000 * blockSize100k;
   s->arr1 = (UInt32 *) BZALLOC( n                  * sizeof(UInt32) );
   s->arr2 = (UInt32 *) BZALLOC( (n+BZ_N_OVERSHOOT) * sizeof(UInt32) );
   
   /* added for storing two arrays to merge */
   s->arr1_first_sort  = (UInt32*) BZALLOC(n * sizeof(UInt32));
   s->arr1_second_sort = (UInt32*) BZALLOC(n * sizeof(UInt32));
   s->arr1_first_sort_rank = (UInt32*) BZALLOC(n * sizeof(UInt32));
   
   s->ftab = (UInt32 *) BZALLOC( 65537              * sizeof(UInt32) );

   if (s->arr1 == NULL || s->arr2 == NULL || s->ftab == NULL) {
      if (s->arr1 != NULL) BZFREE(s->arr1);
      if (s->arr2 != NULL) BZFREE(s->arr2);
      if (s->ftab != NULL) BZFREE(s->ftab);
      if (s       != NULL) BZFREE(s);
      return BZ_MEM_ERROR;
   }

   s->blockNo           = 0;
   s->state             = BZ_S_INPUT;
   s->mode              = BZ_M_RUNNING;
   s->combinedCRC       = 0;
   s->blockSize100k     = blockSize100k;
   s->nblockMAX         = 100000 * blockSize100k - 19; //changed @aditya
   s->verbosity         = verbosity;
   s->workFactor        = workFactor;
   s->numThreads        = numThreads;
   s->block             = (UChar*)s->arr2;
   s->mtfv              = (UInt16*)s->arr1;
   s->zbits             = NULL;
   s->ptr               = (UInt32*)s->arr1;

   // allocate current state to the first empty location
   strm->state[strm->state_fill_count]          = s;
   strm->total_in_lo32  = 0;
   strm->total_in_hi32  = 0;
   strm->total_out_lo32 = 0;
   strm->total_out_hi32 = 0;
   init_RL ( s );
   prepare_new_block ( s );
   return BZ_OK;
}

void transfer_state_information (cuda_bz_stream *strm ) { 
  
   EState *prev = (EState*) strm->state[strm->state_fill_count]; 
   EState *s = (EState*) strm->state[strm->state_fill_count + 1];

   // s->mode              = prev->mode;
   // s->state             = prev->state;
   // s->avail_in_expect = prev->avail_in_expect;
   
   // s->origPtr = prev->origPtr;
   
   // s->zbits             = prev->zbits;
   
   // s->workFactor        = prev->workFactor;

   // s->state_in_ch       = prev->state_in_ch;
   // s->state_in_len       = prev->state_in_len;
   
   // s->nblockMAX         = INCREASE_BLOCK_SIZE * 100000 * prev->blockSize100k - 19;
 
   s->bsBuff = prev->bsBuff; 
   s->bsLive = prev->bsLive; 

   //s->blockCRC       = prev->blockCRC;
   s->combinedCRC       = prev->combinedCRC;
 
   BZ_INITIALISE_CRC ( s->blockCRC );
   // UInt32 i;
   // for (i = 0; i < 256; i++) s->inUse[i] = False;
  
   // s->verbosity         = prev->verbosity;
   // s->blockNo           = prev->blockNo;
   // s->blockSize100k     = prev->blockSize100k; 
   // prepare_new_block ( s ); 

   strm->state_fill_count++;

}

int allocate_new_block_in_stream ( cuda_bz_stream *strm )
{
   EState *s;

   // retrieve parameters of the EState associated with previous block
   EState *prev = (EState*) strm->state[strm->state_fill_count];

   s = (EState *) BZALLOC( sizeof(EState) );
   if (s == NULL) return BZ_MEM_ERROR;

   s->strm = strm;

   s->arr1 = NULL;
   s->arr2 = NULL;
   s->ftab = NULL;

   UInt32 n;

   n       = 100000 * prev->blockSize100k;
   s->arr1 = (UInt32 *) BZALLOC( n                  * sizeof(UInt32) );
   s->arr2 = (UInt32 *) BZALLOC( (n+BZ_N_OVERSHOOT) * sizeof(UInt32) );
   s->ftab = (UInt32 *) BZALLOC( 65537              * sizeof(UInt32) );

   /* added for storing two arrays to merge */
   s->arr1_first_sort  = (UInt32*) BZALLOC(n * sizeof(UInt32));
   s->arr1_second_sort = (UInt32*) BZALLOC(n * sizeof(UInt32));
   s->arr1_first_sort_rank = (UInt32*) BZALLOC(n * sizeof(UInt32));

   if (s->arr1 == NULL || s->arr2 == NULL || s->ftab == NULL) {
      if (s->arr1 != NULL) BZFREE(s->arr1);
      if (s->arr2 != NULL) BZFREE(s->arr2);
      if (s->ftab != NULL) BZFREE(s->ftab);
      if (s       != NULL) BZFREE(s);
      return BZ_MEM_ERROR;
   }

   s->ptr               = (UInt32*)s->arr1;
   s->block             = (UChar*)s->arr2;
   s->mtfv              = (UInt16*)s->arr1;
//
   s->mode              = prev->mode;
   s->state             = prev->state;
   s->avail_in_expect = prev->avail_in_expect;
   
   s->origPtr = prev->origPtr;
   
   s->zbits             = prev->zbits;
   
   s->workFactor        = prev->workFactor;

   s->state_in_ch       = prev->state_in_ch;
   s->state_in_len       = prev->state_in_len;
   
   s->nblockMAX         = 100000 * prev->blockSize100k - 19;
 
   s->bsBuff = prev->bsBuff; 
   s->bsLive = prev->bsLive; 

   s->blockCRC       = prev->blockCRC;
   s->combinedCRC       = prev->combinedCRC;
   
   s->verbosity         = prev->verbosity;
   s->blockNo           = prev->blockNo;
   s->blockSize100k     = prev->blockSize100k; 
   prepare_new_block ( s ); 
//
   strm->state[strm->state_fill_count + 1]          = s; 

   return BZ_OK;
}


/*---------------------------------------------------*/
static
void add_pair_to_block ( EState* s )
{
   Int32 i;
   UChar ch = (UChar)(s->state_in_ch);
   for (i = 0; i < s->state_in_len; i++) {
      BZ_UPDATE_CRC( s->blockCRC, ch );
   }
   s->inUse[s->state_in_ch] = True;
   switch (s->state_in_len) {
      case 1:
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         break;
      case 2:
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         break;
      case 3:
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         break;
      default:
         s->inUse[s->state_in_len-4] = True;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = (UChar)ch; s->nblock++;
         s->block[s->nblock] = ((UChar)(s->state_in_len-4));
         s->nblock++;
         break;
   }
}


/*---------------------------------------------------*/
static
void flush_RL ( EState* s )
{
   if (s->state_in_ch < 256) add_pair_to_block ( s );
   init_RL ( s );
}


/*---------------------------------------------------*/
#define ADD_CHAR_TO_BLOCK(zs,zchh0)               \
{                                                 \
   UInt32 zchh = (UInt32)(zchh0);                 \
   /*-- fast track the common case --*/           \
   if (zchh != zs->state_in_ch &&                 \
       zs->state_in_len == 1) {                   \
      UChar ch = (UChar)(zs->state_in_ch);        \
      BZ_UPDATE_CRC( zs->blockCRC, ch );          \
      zs->inUse[zs->state_in_ch] = True;          \
      zs->block[zs->nblock] = (UChar)ch;          \
      zs->nblock++;                               \
      zs->state_in_ch = zchh;                     \
   }                                              \
   else                                           \
   /*-- general, uncommon cases --*/              \
   if (zchh != zs->state_in_ch ||                 \
      zs->state_in_len == 255) {                  \
      if (zs->state_in_ch < 256)                  \
         add_pair_to_block ( zs );                \
      zs->state_in_ch = zchh;                     \
      zs->state_in_len = 1;                       \
   } else {                                       \
      zs->state_in_len++;                         \
   }                                              \
}


/*---------------------------------------------------*/
static
Bool copy_input_until_stop ( EState* s )
{
   Bool progress_in = False;

   if (s->mode == BZ_M_RUNNING) {

      /*-- fast track the common case --*/
      while (True) {
         /*-- block full? --*/
         if (s->nblock >= s->nblockMAX) break;
         /*-- no input? --*/
         if (s->strm->avail_in == 0) break;
         progress_in = True;
         ADD_CHAR_TO_BLOCK ( s, (UInt32)(*((UChar*)(s->strm->next_in))) ); 
         s->strm->next_in++;
         s->strm->avail_in--;
         s->strm->total_in_lo32++;
         if (s->strm->total_in_lo32 == 0) s->strm->total_in_hi32++;
      }

   } else {

      /*-- general, uncommon case --*/
      while (True) {
         /*-- block full? --*/
         if (s->nblock >= s->nblockMAX) break;
         /*-- no input? --*/
         if (s->strm->avail_in == 0) break;
         /*-- flush/finish end? --*/
         if (s->avail_in_expect == 0) break;
         progress_in = True;
         ADD_CHAR_TO_BLOCK ( s, (UInt32)(*((UChar*)(s->strm->next_in))) ); 
         s->strm->next_in++;
         s->strm->avail_in--;
         s->strm->total_in_lo32++;
         if (s->strm->total_in_lo32 == 0) s->strm->total_in_hi32++;
         s->avail_in_expect--;
      }
   }
   return progress_in;
}


/*---------------------------------------------------*/
static
Bool copy_output_until_stop ( EState* s )
{
   Bool progress_out = False;

   while (True) {

      /*-- no output space? --*/
      if (s->strm->avail_out == 0) break;

      /*-- block done? --*/
      if (s->state_out_pos >= s->numZ) break;

      progress_out = True;
      *(s->strm->next_out) = s->zbits[s->state_out_pos];
      s->state_out_pos++;
      s->strm->avail_out--;
      s->strm->next_out++;
      s->strm->total_out_lo32++;
      if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
   }

   return progress_out;
}


/*---------------------------------------------------*/
static
Bool copy_onechar_to_output (EState* s, char putchar )
{
   Bool progress_out = False;

   /*-- no output space? --*/
   if (s->strm->avail_out == 0) return False;

   /*-- block done? --*/
   if (s->state_out_pos >= s->numZ) return False;

   progress_out = True;
   *(s->strm->next_out) = putchar;
   s->state_out_pos++;
   s->strm->avail_out--;
   s->strm->next_out++;
   s->strm->total_out_lo32++;
   if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
 

   return progress_out;
}


/*---------------------------------------------------*/

Bool compress_all_blocks_in_stream (cuda_bz_stream* strm)
{
	// UInt32 i;
	// Int32 count;
	// EState* s;
	/* for(i=0; i <= strm->state_fill_count; i++) { 
		s = (EState *)strm->state[i];
		BZ_INITIALISE_CRC ( s->blockCRC );
		cuda_BZ2_compressBlock(s, (Bool) i == strm->state_fill_count);
                #ifdef PRINT_DEBUG
		printf("blockno %d, blocksize %d, numZ %d\n",s->blockNo, s->nblock, s->numZ);
		printf("bsLive %u, bsBuff %d\n",s->bsLive, s->bsBuff);
                #endif
	} */

	//cuda_BZ2_compressBlocks_pthreads( strm );  
	bool ret = cuda_BZ2_compressBlocks( strm ); 
   if(ret == false)
      return false; 
	/* We have all the bit streams, so write the compressed file */
	
	// Int32 startLiveValue = 0; 
	// UInt32 startBuffValue = 0;
	// Int32 endLiveValue = 0;
	// UInt32 endBuffValue = 0;
	// UChar printChar;
   // // FILE *outStr;

   // // FILE *outStr = fopen( "/home/scq/put.tiff.bz2", "wb" );
   // // strm->handle = outStr;

	// for(i = 0; i <= strm->state_fill_count; i++) { 
	// 	s = (EState *)strm->state[i];
	// 	startLiveValue = s->bsLive; 
	// 	startBuffValue = s->bsBuff;
   //    //           #ifdef PRINT_DEBUG
	// 	// printf("Before s->numZ %d, startLiveValue %d, startBuffValue %u\n",s->numZ, startLiveValue, startBuffValue);
	// 	// #endif	
	// 	while(startLiveValue >= 8) { 
	// 		s->zbits[s->numZ] = (UChar)(startBuffValue >> 24);
	// 		s->numZ++;
	// 		startBuffValue <<=8;
	// 		startLiveValue -=8;
	// 	}
	// 	#ifdef PRINT_DEBUG
	// 	printf("After s->numZ %d, startLiveValue %d, startBuffValue %u\n",s->numZ, startLiveValue, startBuffValue);
   //              #endif

	// 	if(endLiveValue < 0 || endLiveValue > 7) { 
	// 		printf("Error Concatenating Bit Streams\n");
	// 	}
      
	// 	for(count = 0; count < s->numZ; count++) {
	// 		printChar = (UChar)(endBuffValue >> 24) | ((UChar)s->zbits[count] >> endLiveValue);
	// 		endBuffValue = (UInt32)((UChar)s->zbits[count] << (8-endLiveValue)) << 24;
   //       copy_onechar_to_output(s,printChar);
   //       // *(s->strm->next_out) = printChar;
   //       /*-- no output space? --*/
   //       // if (s->strm->avail_out == 0) break;

   //       // /*-- block done? --*/
   //       // if (s->state_out_pos >= s->numZ) break;

   //       // *(s->strm->next_out) = printChar;
   //       // s->state_out_pos++;
   //       // s->strm->avail_out--;
   //       // s->strm->next_out++;
   //       // s->strm->total_out_lo32++;
   //       // if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
   //       // *(strm->compressdata++) = (char)printChar;
   //       // strcat(strm->compressdata,(char*)printChar);
	// 		// fprintf(strm->handle,"%c",(char)printChar);
	// 	}
	// 	endBuffValue |= ((UInt32)startBuffValue >> endLiveValue);
	// 	endLiveValue += startLiveValue;

	// 	// handle overflow in the stored buff
	// 	if(endLiveValue >= 8) { 
	// 		printChar = (UChar)(endBuffValue >> 24);
   //       copy_onechar_to_output(s,printChar);
   //       // *(strm->compressdata++) = (char)printChar;
	// 		// fprintf(strm->handle,"%c",(char)printChar);
	// 		endLiveValue -=8;
	// 		endBuffValue <<=8;
	// 	}
	// }

	// // terminating character printed
	// if(endLiveValue!=0) { 
	// 	printChar = (UChar)(endBuffValue >> 24) & ((UChar)(0xFF << endLiveValue));
   //    copy_onechar_to_output(s,printChar);
   //    // *(strm->compressdata++) = (char)printChar;
	// 	// fprintf(strm->handle,"%c",(char)printChar);
	// }
	return True; 
}



/*---------------------------------------------------*/
// static
// Bool cpu_handle_compress ( cuda_bz_stream* strm )
// {
//    Bool progress_in  = False;
//    Bool progress_out = False;
//    EState* s = (EState*)strm->state[0];
   
//    while (True) {

//       if (s->state == BZ_S_OUTPUT) {
//          progress_out |= copy_output_until_stop ( s );
//          if (s->state_out_pos < s->numZ) break;
//          if (s->mode == BZ_M_FINISHING && 
//              s->avail_in_expect == 0 &&
//              isempty_RL(s)) break;
//          prepare_new_block ( s );
//          s->state = BZ_S_INPUT;
//          if (s->mode == BZ_M_FLUSHING && 
//              s->avail_in_expect == 0 &&
//              isempty_RL(s)) break;
//       }

//       if (s->state == BZ_S_INPUT) {
//          progress_in |= copy_input_until_stop ( s );
//          if (s->mode != BZ_M_RUNNING && s->avail_in_expect == 0) {
//             flush_RL ( s );
//             cpu_cuda_BZ2_compressBlock ( s, (Bool)(s->mode == BZ_M_FINISHING) );
//             s->state = BZ_S_OUTPUT;
//          }
//          else
//          if (s->nblock >= s->nblockMAX) {
//             cpu_cuda_BZ2_compressBlock ( s, False );
//             s->state = BZ_S_OUTPUT;
//          }
//          else
//          if (s->strm->avail_in == 0) {
//             break;
//          }
//       }

//    }

//    return progress_in || progress_out;
// }
static
Bool handle_compress ( cuda_bz_stream* strm )
{
   Bool progress_in  = False;
   Bool progress_out = False;
   Bool is_last_block = False;
   // Retrieve the EState corresponding to the current block
   EState* s = (EState *) strm->state[strm->state_fill_count];
   
   while (True) {
/*
      if (s->state == BZ_S_OUTPUT) {
	 if(s->state_out_pos == 0) { 
	 	cuda_BZ2_compressBlock(s, is_last_block);
	 }
         progress_out |= copy_output_until_stop ( s );
         if (s->state_out_pos < s->numZ) break;
         if (s->mode == BZ_M_FINISHING && 
             s->avail_in_expect == 0 &&
             isempty_RL(s)) break;
	 // transfer state information to queue once current block has been compressed
	 transfer_state_information( strm );
	 s = (EState *) strm->state[strm->state_fill_count];
         s->state = BZ_S_INPUT;
         if (s->mode == BZ_M_FLUSHING && 
             s->avail_in_expect == 0 &&
             isempty_RL(s)) break;
      }
*/
      if (s->state == BZ_S_INPUT) {
         progress_in |= copy_input_until_stop ( s );
   	 
	 if (s->mode != BZ_M_RUNNING && s->avail_in_expect == 0) {
	    is_last_block = (Bool)(s->mode == BZ_M_FINISHING); 
	    if(is_last_block != True) { 
		    // allocate space for the new block 
		    // TODO : force load of the new block
		    allocate_new_block_in_stream(strm); 
		    #ifdef PRINT_DEBUG
	 	    printf("Called add_new_block1, s->blockNo %d\n",s->blockNo);
		    #endif
	    	    strm->state_fill_count++;
	            s = (EState *)strm->state[strm->state_fill_count];
		    s->state = BZ_S_INPUT;
	    }
       flush_RL ( s );
	    if(compress_all_blocks_in_stream( strm )) { 
		// #ifdef PRINT_DEBUG
	    	// printf("Compression Successfull\n");
               //  #endif
	    }
       else{
          printf("Compression failed\n");
          return false;
       }
            #ifdef PRINT_DEBUG
	    printf("Exiting...\n");
            #endif
            break;
	   //  exit(1);
            // cuda_BZ2_compressBlock ( s, is_last_block );
            //s->state = BZ_S_OUTPUT;
         }
         else
         if (s->nblock >= s->nblockMAX) {
            is_last_block = False;
            // allocate space for the new block 
            // TODO : force load of the new block
            allocate_new_block_in_stream(strm); 
                  #ifdef PRINT_DEBUG
            printf("Called add_new_block2, s->blockNo %d\n",s->blockNo);
                  #endif
            strm->state_fill_count++;
            s = (EState *)strm->state[strm->state_fill_count];
            s->state = BZ_S_INPUT;
            // cuda_BZ2_compressBlock ( s, is_last_block );
            // s->state = BZ_S_OUTPUT;
         }
         else
         if (s->strm->avail_in == 0) {
            break;
         }
      }

   }

   return progress_in || progress_out;
}
// /*---------------------------------------------------*/
// int BZ_API(cpu_cuda_BZ2_bzCompress) ( cuda_bz_stream *strm, int action )
// {
//    Bool progress;
//    EState* s;
//    if (strm == NULL) return BZ_PARAM_ERROR;
//    s = (EState *) strm->state[0];
//    // current EState is retrieved from state array of cuda_bz_stream structure

//    if (s == NULL) return BZ_PARAM_ERROR;
//    if (s->strm != strm) return BZ_PARAM_ERROR;

//    preswitch:
//    switch (s->mode) {

//       case BZ_M_IDLE:
//          return BZ_SEQUENCE_ERROR;

//       case BZ_M_RUNNING:
//          if (action == BZ_RUN) {
//             progress = cpu_handle_compress ( strm );
// 	    // reset memory location of the current EState in case new block is added
// 	   //  s = (EState*) strm->state[strm->state_fill_count];            
// 	    return progress ? BZ_RUN_OK : BZ_PARAM_ERROR;
//          } 
//          else
// 	 if (action == BZ_FLUSH) {
//             s->avail_in_expect = strm->avail_in;
//             s->mode = BZ_M_FLUSHING;
//             goto preswitch;
//          }
//          else
//          if (action == BZ_FINISH) {
//             s->avail_in_expect = strm->avail_in;
//             s->mode = BZ_M_FINISHING;
//             goto preswitch;
//          }
//          else 
//             return BZ_PARAM_ERROR;

//       case BZ_M_FLUSHING:
//          if (action != BZ_FLUSH) return BZ_SEQUENCE_ERROR;
//          if (s->avail_in_expect != s->strm->avail_in) 
//             return BZ_SEQUENCE_ERROR;
//          progress = cpu_handle_compress ( strm );
// 	 // reset memory location of the current EState in case new block is added
// 	//  s = (EState*) strm->state[strm->state_fill_count];
//          if (s->avail_in_expect > 0 || !isempty_RL(s) ||
//              s->state_out_pos < s->numZ) return BZ_FLUSH_OK;
//          s->mode = BZ_M_RUNNING;
//          return BZ_RUN_OK;

//       case BZ_M_FINISHING:
//          if (action != BZ_FINISH) return BZ_SEQUENCE_ERROR;
//          if (s->avail_in_expect != s->strm->avail_in) 
//             return BZ_SEQUENCE_ERROR;
//          progress = cpu_handle_compress ( strm );
// 	 // reset memory location of the current EState in case new block is added
// 	//  s = (EState*) strm->state[strm->state_fill_count];
//          if (!progress) return BZ_SEQUENCE_ERROR;
//          if (s->avail_in_expect > 0 || !isempty_RL(s) ||
//              s->state_out_pos < s->numZ) return BZ_FINISH_OK;
//          s->mode = BZ_M_IDLE;
//          return BZ_STREAM_END;
//    }
//    return BZ_OK; /*--not reached--*/
// }

/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzCompress) ( cuda_bz_stream *strm, int action )
{
   Bool progress;
   EState* s;
   if (strm == NULL) return BZ_PARAM_ERROR;
   s = (EState *) strm->state[strm->state_fill_count];
   // current EState is retrieved from state array of cuda_bz_stream structure

   if (s == NULL) return BZ_PARAM_ERROR;
   if (s->strm != strm) return BZ_PARAM_ERROR;

   preswitch:
   switch (s->mode) {

      case BZ_M_IDLE:
         return BZ_SEQUENCE_ERROR;

      case BZ_M_RUNNING:
         if (action == BZ_RUN) {
            progress = handle_compress ( strm );
            // reset memory location of the current EState in case new block is added
            s = (EState*) strm->state[strm->state_fill_count];            
            return progress ? BZ_RUN_OK : BZ_PARAM_ERROR;
         } 
         else
	      if (action == BZ_FLUSH) {
            s->avail_in_expect = strm->avail_in;
            s->mode = BZ_M_FLUSHING;
            goto preswitch;
         }
         else
         if (action == BZ_FINISH) {
            s->avail_in_expect = strm->avail_in;
            s->mode = BZ_M_FINISHING;
            goto preswitch;
         }
         else 
            return BZ_PARAM_ERROR;

      case BZ_M_FLUSHING:
         if (action != BZ_FLUSH) return BZ_SEQUENCE_ERROR;
         if (s->avail_in_expect != s->strm->avail_in) 
            return BZ_SEQUENCE_ERROR;
         progress = handle_compress ( strm );
         // reset memory location of the current EState in case new block is added
         s = (EState*) strm->state[strm->state_fill_count];
         if (s->avail_in_expect > 0 || !isempty_RL(s) ||
             s->state_out_pos < s->numZ) return BZ_FLUSH_OK;
         s->mode = BZ_M_RUNNING;
         return BZ_RUN_OK;

      case BZ_M_FINISHING:
         if (action != BZ_FINISH) return BZ_SEQUENCE_ERROR;
         if (s->avail_in_expect != s->strm->avail_in) 
            return BZ_SEQUENCE_ERROR;
         progress = handle_compress ( strm );
         if(progress == false)
            return BZ_ERRO;
         // reset memory location of the current EState in case new block is added
         s = (EState*) strm->state[strm->state_fill_count];
         if (!progress) return BZ_SEQUENCE_ERROR;
         if (s->avail_in_expect > 0 || !isempty_RL(s) ||
             s->state_out_pos < s->numZ) return BZ_FINISH_OK;
         s->mode = BZ_M_IDLE;
         return BZ_STREAM_END;
   }
   return BZ_OK; /*--not reached--*/
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzCompressEnd)  ( cuda_bz_stream *strm )
{
   EState* s;
   UInt32 i;
   if (strm == NULL) return BZ_PARAM_ERROR;

   // free all EState's in a for loop
   for(i = 0; i < strm->state_fill_count; i++) { 
	   s = (EState *)strm->state[i];
	   if (s == NULL) return BZ_PARAM_ERROR;
	   if (s->strm != strm) return BZ_PARAM_ERROR;

	   if (s->arr1 != NULL) BZFREE(s->arr1);
	   if (s->arr2 != NULL) BZFREE(s->arr2);
	   if (s->ftab != NULL) BZFREE(s->ftab);
   	   BZFREE(strm->state[i]);
   	   strm->state[i] = NULL;   
   }
   return BZ_OK;
}


/*---------------------------------------------------*/
/*--- Decompression stuff                         ---*/
/*---------------------------------------------------*/

/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzDecompressInit) 
                     ( cuda_bz_stream* strm, 
                       int        verbosity,
                       int        small )
{
   DState* s;

   if (!bz_config_ok()) return BZ_CONFIG_ERROR;

   if (strm == NULL) return BZ_PARAM_ERROR;
   if (small != 0 && small != 1) return BZ_PARAM_ERROR;
   if (verbosity < 0 || verbosity > 4) return BZ_PARAM_ERROR;

   if (strm->bzalloc == NULL) strm->bzalloc = default_bzalloc;
   if (strm->bzfree == NULL) strm->bzfree = default_bzfree;

   s = (DState *) BZALLOC( sizeof(DState) );
   if (s == NULL) return BZ_MEM_ERROR;
   s->strm                  = strm;
   // use only state[0] for the entire decoding pipeline
   strm->state[0]              = s;
   s->state                 = BZ_X_MAGIC_1;
   s->bsLive                = 0;
   s->bsBuff                = 0;
   s->calculatedCombinedCRC = 0;
   strm->total_in_lo32      = 0;
   strm->total_in_hi32      = 0;
   strm->total_out_lo32     = 0;
   strm->total_out_hi32     = 0;
   s->smallDecompress       = (Bool)small;
   s->ll4                   = NULL;
   s->ll16                  = NULL;
   s->tt                    = NULL;
   s->currBlockNo           = 0;
   s->verbosity             = verbosity;

   return BZ_OK;
}


/*---------------------------------------------------*/
/* Return  True iff data corruption is discovered.
   Returns False if there is no problem.
*/
static
Bool unRLE_obuf_to_output_FAST ( DState* s )
{
   UChar k1;

   if (s->blockRandomised) {

      while (True) {
         /* try to finish existing run */
         while (True) {
            if (s->strm->avail_out == 0) return False;
            if (s->state_out_len == 0) break;
            *( (UChar*)(s->strm->next_out) ) = s->state_out_ch;
            BZ_UPDATE_CRC ( s->calculatedBlockCRC, s->state_out_ch );
            s->state_out_len--;
            s->strm->next_out++;
            s->strm->avail_out--;
            s->strm->total_out_lo32++;
            if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
         }

         /* can a new run be started? */
         if (s->nblock_used == s->save_nblock+1) return False;
               
         /* Only caused by corrupt data stream? */
         if (s->nblock_used > s->save_nblock+1)
            return True;
   
         s->state_out_len = 1;
         s->state_out_ch = s->k0;
         BZ_GET_FAST(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 2;
         BZ_GET_FAST(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 3;
         BZ_GET_FAST(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         BZ_GET_FAST(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         s->state_out_len = ((Int32)k1) + 4;
         BZ_GET_FAST(s->k0); BZ_RAND_UPD_MASK; 
         s->k0 ^= BZ_RAND_MASK; s->nblock_used++;
      }

   } else {

      /* restore */
      UInt32        c_calculatedBlockCRC = s->calculatedBlockCRC;
      UChar         c_state_out_ch       = s->state_out_ch;
      Int32         c_state_out_len      = s->state_out_len;
      Int32         c_nblock_used        = s->nblock_used;
      Int32         c_k0                 = s->k0;
      UInt32*       c_tt                 = s->tt;
      UInt32        c_tPos               = s->tPos;
      char*         cs_next_out          = s->strm->next_out;
      unsigned int  cs_avail_out         = s->strm->avail_out;
      Int32         ro_blockSize100k     = s->blockSize100k;
      /* end restore */

      UInt32       avail_out_INIT = cs_avail_out;
      Int32        s_save_nblockPP = s->save_nblock+1;
      unsigned int total_out_lo32_old;

      while (True) {

         /* try to finish existing run */
         if (c_state_out_len > 0) {
            while (True) {
               if (cs_avail_out == 0) goto return_notr;
               if (c_state_out_len == 1) break;
               *( (UChar*)(cs_next_out) ) = c_state_out_ch;
               BZ_UPDATE_CRC ( c_calculatedBlockCRC, c_state_out_ch );
               c_state_out_len--;
               cs_next_out++;
               cs_avail_out--;
            }
            s_state_out_len_eq_one:
            {
               if (cs_avail_out == 0) { 
                  c_state_out_len = 1; goto return_notr;
               };
               *( (UChar*)(cs_next_out) ) = c_state_out_ch;
               BZ_UPDATE_CRC ( c_calculatedBlockCRC, c_state_out_ch );
               cs_next_out++;
               cs_avail_out--;
            }
         }   
         /* Only caused by corrupt data stream? */
         if (c_nblock_used > s_save_nblockPP)
            return True;

         /* can a new run be started? */
         if (c_nblock_used == s_save_nblockPP) {
            c_state_out_len = 0; goto return_notr;
         };   
         c_state_out_ch = c_k0;
         BZ_GET_FAST_C(k1); c_nblock_used++;
         if (k1 != c_k0) { 
            c_k0 = k1; goto s_state_out_len_eq_one; 
         };
         if (c_nblock_used == s_save_nblockPP) 
            goto s_state_out_len_eq_one;
   
         c_state_out_len = 2;
         BZ_GET_FAST_C(k1); c_nblock_used++;
         if (c_nblock_used == s_save_nblockPP) continue;
         if (k1 != c_k0) { c_k0 = k1; continue; };
   
         c_state_out_len = 3;
         BZ_GET_FAST_C(k1); c_nblock_used++;
         if (c_nblock_used == s_save_nblockPP) continue;
         if (k1 != c_k0) { c_k0 = k1; continue; };
   
         BZ_GET_FAST_C(k1); c_nblock_used++;
         c_state_out_len = ((Int32)k1) + 4;
         BZ_GET_FAST_C(c_k0); c_nblock_used++;
      }

      return_notr:
      total_out_lo32_old = s->strm->total_out_lo32;
      s->strm->total_out_lo32 += (avail_out_INIT - cs_avail_out);
      if (s->strm->total_out_lo32 < total_out_lo32_old)
         s->strm->total_out_hi32++;

      /* save */
      s->calculatedBlockCRC = c_calculatedBlockCRC;
      s->state_out_ch       = c_state_out_ch;
      s->state_out_len      = c_state_out_len;
      s->nblock_used        = c_nblock_used;
      s->k0                 = c_k0;
      s->tt                 = c_tt;
      s->tPos               = c_tPos;
      s->strm->next_out     = cs_next_out;
      s->strm->avail_out    = cs_avail_out;
      /* end save */
   }
   return False;
}



/*---------------------------------------------------*/
//removed inline
Int32 cuda_BZ2_indexIntoF ( Int32 indx, Int32 *cftab )
{
   Int32 nb, na, mid;
   nb = 0;
   na = 256;
   do {
      mid = (nb + na) >> 1;
      if (indx >= cftab[mid]) nb = mid; else na = mid;
   }
   while (na - nb != 1);
   return nb;
}


/*---------------------------------------------------*/
/* Return  True iff data corruption is discovered.
   Returns False if there is no problem.
*/
static
Bool unRLE_obuf_to_output_SMALL ( DState* s )
{
   UChar k1;

   if (s->blockRandomised) {

      while (True) {
         /* try to finish existing run */
         while (True) {
            if (s->strm->avail_out == 0) return False;
            if (s->state_out_len == 0) break;
            *( (UChar*)(s->strm->next_out) ) = s->state_out_ch;
            BZ_UPDATE_CRC ( s->calculatedBlockCRC, s->state_out_ch );
            s->state_out_len--;
            s->strm->next_out++;
            s->strm->avail_out--;
            s->strm->total_out_lo32++;
            if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
         }
   
         /* can a new run be started? */
         if (s->nblock_used == s->save_nblock+1) return False;

         /* Only caused by corrupt data stream? */
         if (s->nblock_used > s->save_nblock+1)
            return True;
   
         s->state_out_len = 1;
         s->state_out_ch = s->k0;
         BZ_GET_SMALL(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 2;
         BZ_GET_SMALL(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 3;
         BZ_GET_SMALL(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         BZ_GET_SMALL(k1); BZ_RAND_UPD_MASK; 
         k1 ^= BZ_RAND_MASK; s->nblock_used++;
         s->state_out_len = ((Int32)k1) + 4;
         BZ_GET_SMALL(s->k0); BZ_RAND_UPD_MASK; 
         s->k0 ^= BZ_RAND_MASK; s->nblock_used++;
      }

   } else {

      while (True) {
         /* try to finish existing run */
         while (True) {
            if (s->strm->avail_out == 0) return False;
            if (s->state_out_len == 0) break;
            *( (UChar*)(s->strm->next_out) ) = s->state_out_ch;
            BZ_UPDATE_CRC ( s->calculatedBlockCRC, s->state_out_ch );
            s->state_out_len--;
            s->strm->next_out++;
            s->strm->avail_out--;
            s->strm->total_out_lo32++;
            if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
         }
   
         /* can a new run be started? */
         if (s->nblock_used == s->save_nblock+1) return False;

         /* Only caused by corrupt data stream? */
         if (s->nblock_used > s->save_nblock+1)
            return True;
   
         s->state_out_len = 1;
         s->state_out_ch = s->k0;
         BZ_GET_SMALL(k1); s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 2;
         BZ_GET_SMALL(k1); s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         s->state_out_len = 3;
         BZ_GET_SMALL(k1); s->nblock_used++;
         if (s->nblock_used == s->save_nblock+1) continue;
         if (k1 != s->k0) { s->k0 = k1; continue; };
   
         BZ_GET_SMALL(k1); s->nblock_used++;
         s->state_out_len = ((Int32)k1) + 4;
         BZ_GET_SMALL(s->k0); s->nblock_used++;
      }

   }
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzDecompress) ( cuda_bz_stream *strm )
{
   Bool    corrupt;
   DState* s;
   if (strm == NULL) return BZ_PARAM_ERROR;
   // state[0] is used for the entire decoding pipeline
   s = (DState *)strm->state[0];
   if (s == NULL) return BZ_PARAM_ERROR;
   if (s->strm != strm) return BZ_PARAM_ERROR;

   while (True) {
      if (s->state == BZ_X_IDLE) return BZ_SEQUENCE_ERROR;
      if (s->state == BZ_X_OUTPUT) {
         if (s->smallDecompress)
            corrupt = unRLE_obuf_to_output_SMALL ( s ); else
            corrupt = unRLE_obuf_to_output_FAST  ( s );
         if (corrupt) return BZ_DATA_ERROR;
         if (s->nblock_used == s->save_nblock+1 && s->state_out_len == 0) {
            BZ_FINALISE_CRC ( s->calculatedBlockCRC );
            if (s->verbosity >= 3) 
               VPrintf2 ( " {0x%08x, 0x%08x}", s->storedBlockCRC, 
                          s->calculatedBlockCRC );
            if (s->verbosity >= 2) VPrintf0 ( "]" );
	    /* Commented CRC check @aditya
            if (s->calculatedBlockCRC != s->storedBlockCRC) { 
             	 printf("CRC Error1\n");
		 //return BZ_DATA_ERROR;
	    } */
            s->calculatedCombinedCRC 
               = (s->calculatedCombinedCRC << 1) | 
                    (s->calculatedCombinedCRC >> 31);
            s->calculatedCombinedCRC ^= s->calculatedBlockCRC;
            s->state = BZ_X_BLKHDR_1;
         } else {
            return BZ_OK;
         }
      }
      if (s->state >= BZ_X_MAGIC_1) {
         Int32 r = cuda_BZ2_decompress ( s );
         if (r == BZ_STREAM_END) {
            if (s->verbosity >= 3)
               VPrintf2 ( "\n    combined CRCs: stored = 0x%08x, computed = 0x%08x", 
                          s->storedCombinedCRC, s->calculatedCombinedCRC );
	    /* Commented CRC check @aditya
            if (s->calculatedCombinedCRC != s->storedCombinedCRC) {
             	 printf("CRC Error2\n");
		 //return BZ_DATA_ERROR;
	    } */
            return r;
         }
         if (s->state != BZ_X_OUTPUT) return r;
      }
   }

   AssertH ( 0, 6001 );

   return 0;  /*NOTREACHED*/
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzDecompressEnd)  ( cuda_bz_stream *strm )
{
   DState* s;
   if (strm == NULL) return BZ_PARAM_ERROR;
   // state[0] is used for the entire decoding pipeline
   s = (DState *)strm->state[0];
   if (s == NULL) return BZ_PARAM_ERROR;
   if (s->strm != strm) return BZ_PARAM_ERROR;

   if (s->tt   != NULL) BZFREE(s->tt);
   if (s->ll16 != NULL) BZFREE(s->ll16);
   if (s->ll4  != NULL) BZFREE(s->ll4);

   BZFREE(strm->state[0]);
   strm->state[0] = NULL;

   return BZ_OK;
}


#ifndef BZ_NO_STDIO
/*---------------------------------------------------*/
/*--- File I/O stuff                              ---*/
/*---------------------------------------------------*/

#define BZ_SETERR(eee)                    \
{                                         \
   if (bzerror != NULL) *bzerror = eee;   \
   if (bzf != NULL) bzf->lastErr = eee;   \
}

typedef 
   struct {
      FILE*     handle;
      Char      buf[BZ_MAX_UNUSED];
      Int32     bufN;
      Bool      writing;
      cuda_bz_stream strm;
      Int32     lastErr;
      Bool      initialisedOk;
   }
   bzFile;


/*---------------------------------------------*/
static Bool myfeof ( FILE* f )
{
   Int32 c = fgetc ( f );
   if (c == EOF) return True;
   ungetc ( c, f );
   return False;
}


/*---------------------------------------------------*/
BZFILE* BZ_API(cuda_BZ2_bzWriteOpen) 
                    ( int*  bzerror,      
                      FILE* f, 
                      int   blockSize100k, 
                      int   verbosity,
                      int   workFactor,
		      int   numThreads)
{
   Int32   ret;
   bzFile* bzf = NULL;

   BZ_SETERR(BZ_OK);

   if (f == NULL ||
       (blockSize100k < 1 || blockSize100k > 150) ||
       (workFactor < 0 || workFactor > 250) ||
       (verbosity < 0 || verbosity > 4))
      { BZ_SETERR(BZ_PARAM_ERROR); return NULL; };

   if (ferror(f))
      { BZ_SETERR(BZ_IO_ERROR); return NULL; };

   bzf = (bzFile *)malloc ( sizeof(bzFile) );
   if (bzf == NULL)
      { BZ_SETERR(BZ_MEM_ERROR); return NULL; };

   BZ_SETERR(BZ_OK);
   bzf->initialisedOk = False;
   bzf->bufN          = 0;
   bzf->handle        = f;
   bzf->writing       = True;
   bzf->strm.bzalloc  = NULL;
   bzf->strm.bzfree   = NULL;
   bzf->strm.opaque   = NULL;

   if (workFactor == 0) workFactor = 30;
   ret = cuda_BZ2_bzCompressInit ( &(bzf->strm), blockSize100k, 
                              verbosity, workFactor, numThreads );
   if (ret != BZ_OK)
      { BZ_SETERR(ret); free(bzf); return NULL; };

   bzf->strm.avail_in = 0;
   bzf->initialisedOk = True;
   return bzf;   
}



/*---------------------------------------------------*/
void BZ_API(cuda_BZ2_bzWrite)
             ( int*    bzerror, 
               BZFILE* b, 
               void*   buf, 
               int     len )
{
   Int32 n, n2, ret;
   bzFile* bzf = (bzFile*)b;

   BZ_SETERR(BZ_OK);
   if (bzf == NULL || buf == NULL || len < 0)
      { BZ_SETERR(BZ_PARAM_ERROR); return; };
   if (!(bzf->writing))
      { BZ_SETERR(BZ_SEQUENCE_ERROR); return; };
   if (ferror(bzf->handle))
      { BZ_SETERR(BZ_IO_ERROR); return; };

   if (len == 0)
      { BZ_SETERR(BZ_OK); return; };

   bzf->strm.avail_in = len;
   bzf->strm.next_in  = (char *) buf;

   while (True) {
      bzf->strm.avail_out =  BZ_MAX_UNUSED; 
      bzf->strm.next_out = bzf->buf;
      bzf->strm.handle = bzf->handle;
      ret = cuda_BZ2_bzCompress ( &(bzf->strm), BZ_RUN );
      if (ret != BZ_RUN_OK)
         { BZ_SETERR(ret); return; };

      if (bzf->strm.avail_out < BZ_MAX_UNUSED) {
         n = BZ_MAX_UNUSED - bzf->strm.avail_out;
         n2 = fwrite ( (void*)(bzf->buf), sizeof(UChar), 
                       n, bzf->handle );
         if (n != n2 || ferror(bzf->handle))
            { BZ_SETERR(BZ_IO_ERROR); return; };
      }

      if (bzf->strm.avail_in == 0)
         { BZ_SETERR(BZ_OK); return; };
   }
}


/*---------------------------------------------------*/
void BZ_API(cuda_BZ2_bzWriteClose)
                  ( int*          bzerror, 
                    BZFILE*       b, 
                    int           abandon,
                    unsigned int* nbytes_in,
                    unsigned int* nbytes_out )
{
   cuda_BZ2_bzWriteClose64 ( bzerror, b, abandon, 
                        nbytes_in, NULL, nbytes_out, NULL );
}


void BZ_API(cuda_BZ2_bzWriteClose64)
                  ( int*          bzerror, 
                    BZFILE*       b, 
                    int           abandon,
                    unsigned int* nbytes_in_lo32,
                    unsigned int* nbytes_in_hi32,
                    unsigned int* nbytes_out_lo32,
                    unsigned int* nbytes_out_hi32 )
{
   Int32   n, n2, ret;
   bzFile* bzf = (bzFile*)b;

   if (bzf == NULL)
      { BZ_SETERR(BZ_OK); return; };
   if (!(bzf->writing))
      { BZ_SETERR(BZ_SEQUENCE_ERROR); return; };
   if (ferror(bzf->handle))
      { BZ_SETERR(BZ_IO_ERROR); return; };

   if (nbytes_in_lo32 != NULL) *nbytes_in_lo32 = 0;
   if (nbytes_in_hi32 != NULL) *nbytes_in_hi32 = 0;
   if (nbytes_out_lo32 != NULL) *nbytes_out_lo32 = 0;
   if (nbytes_out_hi32 != NULL) *nbytes_out_hi32 = 0;

   if ((!abandon) && bzf->lastErr == BZ_OK) {
      while (True) {
         bzf->strm.avail_out = BZ_MAX_UNUSED;
         bzf->strm.next_out = bzf->buf;
         ret = cuda_BZ2_bzCompress ( &(bzf->strm), BZ_FINISH );
         if (ret != BZ_FINISH_OK && ret != BZ_STREAM_END)
            { BZ_SETERR(ret); return; };

         if (bzf->strm.avail_out < BZ_MAX_UNUSED) {
            n = BZ_MAX_UNUSED - bzf->strm.avail_out;
            n2 = fwrite ( (void*)(bzf->buf), sizeof(UChar), 
                          n, bzf->handle );
            if (n != n2 || ferror(bzf->handle))
               { BZ_SETERR(BZ_IO_ERROR); return; };
         }

         if (ret == BZ_STREAM_END) break;
      }
   }

   if ( !abandon && !ferror ( bzf->handle ) ) {
      fflush ( bzf->handle );
      if (ferror(bzf->handle))
         { BZ_SETERR(BZ_IO_ERROR); return; };
   }

   if (nbytes_in_lo32 != NULL)
      *nbytes_in_lo32 = bzf->strm.total_in_lo32;
   if (nbytes_in_hi32 != NULL)
      *nbytes_in_hi32 = bzf->strm.total_in_hi32;
   if (nbytes_out_lo32 != NULL)
      *nbytes_out_lo32 = bzf->strm.total_out_lo32;
   if (nbytes_out_hi32 != NULL)
      *nbytes_out_hi32 = bzf->strm.total_out_hi32;

   BZ_SETERR(BZ_OK);
   cuda_BZ2_bzCompressEnd ( &(bzf->strm) );
   free ( bzf );
}


/*---------------------------------------------------*/
BZFILE* BZ_API(cuda_BZ2_bzReadOpen) 
                   ( int*  bzerror, 
                     FILE* f, 
                     int   verbosity,
                     int   small,
                     void* unused,
                     int   nUnused )
{
   bzFile* bzf = NULL;
   int     ret;

   BZ_SETERR(BZ_OK);

   if (f == NULL || 
       (small != 0 && small != 1) ||
       (verbosity < 0 || verbosity > 4) ||
       (unused == NULL && nUnused != 0) ||
       (unused != NULL && (nUnused < 0 || nUnused > BZ_MAX_UNUSED)))
      { BZ_SETERR(BZ_PARAM_ERROR); return NULL; };

   if (ferror(f))
      { BZ_SETERR(BZ_IO_ERROR); return NULL; };

   bzf = (bzFile *)malloc ( sizeof(bzFile) );
   if (bzf == NULL) 
      { BZ_SETERR(BZ_MEM_ERROR); return NULL; };

   BZ_SETERR(BZ_OK);

   bzf->initialisedOk = False;
   bzf->handle        = f;
   bzf->bufN          = 0;
   bzf->writing       = False;
   bzf->strm.bzalloc  = NULL;
   bzf->strm.bzfree   = NULL;
   bzf->strm.opaque   = NULL;
   
   while (nUnused > 0) {
      bzf->buf[bzf->bufN] = *((UChar*)(unused)); bzf->bufN++;
      unused = ((void*)( 1 + ((UChar*)(unused))  ));
      nUnused--;
   }

   ret = cuda_BZ2_bzDecompressInit ( &(bzf->strm), verbosity, small );
   if (ret != BZ_OK)
      { BZ_SETERR(ret); free(bzf); return NULL; };

   bzf->strm.avail_in = bzf->bufN;
   bzf->strm.next_in  = bzf->buf;

   bzf->initialisedOk = True;
   return bzf;   
}


/*---------------------------------------------------*/
void BZ_API(cuda_BZ2_bzReadClose) ( int *bzerror, BZFILE *b )
{
   bzFile* bzf = (bzFile*)b;

   BZ_SETERR(BZ_OK);
   if (bzf == NULL)
      { BZ_SETERR(BZ_OK); return; };

   if (bzf->writing)
      { BZ_SETERR(BZ_SEQUENCE_ERROR); return; };

   if (bzf->initialisedOk)
      (void)cuda_BZ2_bzDecompressEnd ( &(bzf->strm) );
   free ( bzf );
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzRead) 
           ( int*    bzerror, 
             BZFILE* b, 
             void*   buf, 
             int     len )
{
   Int32   n, ret;
   bzFile* bzf = (bzFile*)b;

   BZ_SETERR(BZ_OK);

   if (bzf == NULL || buf == NULL || len < 0)
      { BZ_SETERR(BZ_PARAM_ERROR); return 0; };

   if (bzf->writing)
      { BZ_SETERR(BZ_SEQUENCE_ERROR); return 0; };

   if (len == 0)
      { BZ_SETERR(BZ_OK); return 0; };

   bzf->strm.avail_out = len;
   bzf->strm.next_out = (char *)buf;

   while (True) {

      if (ferror(bzf->handle)) 
         { BZ_SETERR(BZ_IO_ERROR); return 0; };

      if (bzf->strm.avail_in == 0 && !myfeof(bzf->handle)) {
         n = fread ( bzf->buf, sizeof(UChar), 
                     BZ_MAX_UNUSED, bzf->handle );
         if (ferror(bzf->handle))
            { BZ_SETERR(BZ_IO_ERROR); return 0; };
         bzf->bufN = n;
         bzf->strm.avail_in = bzf->bufN;
         bzf->strm.next_in = bzf->buf;
      }

      ret = cuda_BZ2_bzDecompress ( &(bzf->strm) );

      if (ret != BZ_OK && ret != BZ_STREAM_END)
         { BZ_SETERR(ret); return 0; };

      if (ret == BZ_OK && myfeof(bzf->handle) && 
          bzf->strm.avail_in == 0 && bzf->strm.avail_out > 0)
         { BZ_SETERR(BZ_UNEXPECTED_EOF); return 0; };

      if (ret == BZ_STREAM_END)
         { BZ_SETERR(BZ_STREAM_END);
           return len - bzf->strm.avail_out; };
      if (bzf->strm.avail_out == 0)
         { BZ_SETERR(BZ_OK); return len; };
      
   }

   return 0; /*not reached*/
}


/*---------------------------------------------------*/
void BZ_API(cuda_BZ2_bzReadGetUnused) 
                     ( int*    bzerror, 
                       BZFILE* b, 
                       void**  unused, 
                       int*    nUnused )
{
   bzFile* bzf = (bzFile*)b;
   if (bzf == NULL)
      { BZ_SETERR(BZ_PARAM_ERROR); return; };
   if (bzf->lastErr != BZ_STREAM_END)
      { BZ_SETERR(BZ_SEQUENCE_ERROR); return; };
   if (unused == NULL || nUnused == NULL)
      { BZ_SETERR(BZ_PARAM_ERROR); return; };

   BZ_SETERR(BZ_OK);
   *nUnused = bzf->strm.avail_in;
   *unused = bzf->strm.next_in;
}
#endif

/*---------------------------------------------------*/
int BZ_API(cpu_cuda_BZ2_bzCompressInit) 
                    ( cuda_bz_stream* strm, 
                     int        blockSize100k,
                     int        verbosity,
                     int        workFactor )
{
   Int32   n;
   EState* s;

   if (!bz_config_ok()) return BZ_CONFIG_ERROR;

   if (strm == NULL || 
       blockSize100k < 1 || blockSize100k > 9 ||
       workFactor < 0 || workFactor > 250)
     return BZ_PARAM_ERROR;

   if (workFactor == 0) workFactor = 30;
   if (strm->bzalloc == NULL) strm->bzalloc = default_bzalloc;
   if (strm->bzfree == NULL) strm->bzfree = default_bzfree;

   s =(EState *) BZALLOC( sizeof(EState) );
   if (s == NULL) return BZ_MEM_ERROR;
   s->strm = strm;

   s->arr1 = NULL;
   s->arr2 = NULL;
   s->ftab = NULL;

   n       = 100000 * blockSize100k;
   s->arr1 = (UInt32 *) BZALLOC( n                  * sizeof(UInt32) );
   s->arr2 = (UInt32 *) BZALLOC( (n+BZ_N_OVERSHOOT) * sizeof(UInt32) );
   s->ftab = (UInt32 *) BZALLOC( 65537              * sizeof(UInt32) );

   if (s->arr1 == NULL || s->arr2 == NULL || s->ftab == NULL) {
      if (s->arr1 != NULL) BZFREE(s->arr1);
      if (s->arr2 != NULL) BZFREE(s->arr2);
      if (s->ftab != NULL) BZFREE(s->ftab);
      if (s       != NULL) BZFREE(s);
      return BZ_MEM_ERROR;
   }

   s->blockNo           = 0;
   s->state             = BZ_S_INPUT;
   s->mode              = BZ_M_RUNNING;
   s->combinedCRC       = 0;
   s->blockSize100k     = blockSize100k;
   s->nblockMAX         = 100000 * blockSize100k - 19;
   s->verbosity         = verbosity;
   s->workFactor        = workFactor;

   s->block             = (UChar*)s->arr2;
   s->mtfv              = (UInt16*)s->arr1;
   s->zbits             = NULL;
   s->ptr               = (UInt32*)s->arr1;

   strm->state[0]         = s;
   strm->total_in_lo32  = 0;
   strm->total_in_hi32  = 0;
   strm->total_out_lo32 = 0;
   strm->total_out_hi32 = 0;
   init_RL ( s );
   prepare_new_block ( s );
   return BZ_OK;
}
/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bwtTransfer) 
                         ( cuda_bz_stream *strm, 
                           char*         source, 
                           unsigned int  sourceLen,
                           int           blockSize100k, 
                           int           verbosity, 
                           int           workFactor,
                           int           numThreads )
{
   // cuda_bz_stream strm;
   int ret;

   if (source == NULL ||
       blockSize100k < 1 || blockSize100k > 150 ||
       verbosity < 0 || verbosity > 4 ||
       workFactor < 0 || workFactor > 250) 
      return BZ_PARAM_ERROR;

   if (workFactor == 0) workFactor = 30;
   strm->bzalloc = NULL;
   strm->bzfree = NULL;
   strm->opaque = NULL;
   strm->handle = NULL;
   ret = cuda_BZ2_bzCompressInit ( strm, blockSize100k, 
                              verbosity, workFactor,1 /* change */ );
   if (ret != BZ_OK) return ret;

   strm->next_in = source;
   strm->next_out = NULL;
   strm->avail_in = sourceLen;
   strm->avail_out = NULL;

   ret = cuda_BZ2_bzCompress ( strm, BZ_FINISH );
   // if (ret == BZ_FINISH_OK) goto output_overflow;
   // if (ret != BZ_STREAM_END) goto errhandler;

   // /* normal termination */
   // // *destLen -= strm.avail_out;   
   // cuda_BZ2_bzCompressEnd ( strm );
   // return BZ_OK;

   // output_overflow:
   // cuda_BZ2_bzCompressEnd ( strm );
   // return BZ_OUTBUFF_FULL;

   // errhandler:
   // cuda_BZ2_bzCompressEnd ( strm );
   return ret;
}
/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzBuffToBuffCompress) 
                         ( char*         dest, 
                           unsigned int* destLen,
                           char*         source, 
                           unsigned int  sourceLen,
                           int           blockSize100k, 
                           int           verbosity, 
                           int           workFactor,
                           int           numThreads )
{
   cuda_bz_stream strm;
   EState* s;
   int ret;

   if (dest == NULL || destLen == NULL || 
       source == NULL ||
       blockSize100k < 1 || blockSize100k > 150 ||
       verbosity < 0 || verbosity > 4 ||
       workFactor < 0 || workFactor > 250) 
      return BZ_PARAM_ERROR;

   if (workFactor == 0) workFactor = 30;
   strm.bzalloc = NULL;
   strm.bzfree = NULL;
   strm.opaque = NULL;
   strm.handle = (FILE*)dest;
   ret = cuda_BZ2_bzCompressInit ( &strm, blockSize100k, 
                              verbosity, workFactor,1 /* change */ );
   if (ret != BZ_OK) return ret;

   strm.next_in = source;
   strm.next_out = dest;
   strm.avail_in = sourceLen;
   strm.avail_out = *destLen;

   

   ret = cuda_BZ2_bzCompress ( &strm, BZ_FINISH );
   if (ret == BZ_FINISH_OK) goto output_overflow;
   if (ret != BZ_STREAM_END) goto errhandler;

   s = (EState *)strm.state[0];

   /* normal termination */
   *destLen -= strm.avail_out;   
   cuda_BZ2_bzCompressEnd ( &strm );
   return BZ_OK;

   output_overflow:
   cuda_BZ2_bzCompressEnd ( &strm );
   return BZ_OUTBUFF_FULL;

   errhandler:
   cuda_BZ2_bzCompressEnd ( &strm );
   return ret;
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzBuffToBuffDecompress) 
                           ( char*         dest, 
                             unsigned int* destLen,
                             char*         source, 
                             unsigned int  sourceLen,
                             int           small,
                             int           verbosity )
{
   cuda_bz_stream strm;
   int ret;

   if (dest == NULL || destLen == NULL || 
       source == NULL ||
       (small != 0 && small != 1) ||
       verbosity < 0 || verbosity > 4) 
          return BZ_PARAM_ERROR;

   strm.bzalloc = NULL;
   strm.bzfree = NULL;
   strm.opaque = NULL;
   ret = cuda_BZ2_bzDecompressInit ( &strm, verbosity, small );
   if (ret != BZ_OK) return ret;

   strm.next_in = source;
   strm.next_out = dest;
   strm.avail_in = sourceLen;
   strm.avail_out = *destLen;

   ret = cuda_BZ2_bzDecompress ( &strm );
   if (ret == BZ_OK) goto output_overflow_or_eof;
   if (ret != BZ_STREAM_END) goto errhandler;

   /* normal termination */
   *destLen -= strm.avail_out;
   cuda_BZ2_bzDecompressEnd ( &strm );
   return BZ_OK;

   output_overflow_or_eof:
   if (strm.avail_out > 0) {
      cuda_BZ2_bzDecompressEnd ( &strm );
      return BZ_UNEXPECTED_EOF;
   } else {
      cuda_BZ2_bzDecompressEnd ( &strm );
      return BZ_OUTBUFF_FULL;
   };      

   errhandler:
   cuda_BZ2_bzDecompressEnd ( &strm );
   return ret; 
}
/*---------------------------------------------------*/
int BZ_API(cpu_cuda_BZ2_bzDecompressInit) 
                     ( cuda_bz_stream* strm, 
                       int        verbosity,
                       int        small )
{
   DState* s;

   if (!bz_config_ok()) return BZ_CONFIG_ERROR;

   if (strm == NULL) return BZ_PARAM_ERROR;
   if (small != 0 && small != 1) return BZ_PARAM_ERROR;
   if (verbosity < 0 || verbosity > 4) return BZ_PARAM_ERROR;

   if (strm->bzalloc == NULL) strm->bzalloc = default_bzalloc;
   if (strm->bzfree == NULL) strm->bzfree = default_bzfree;

   s = (DState*)BZALLOC( sizeof(DState) );
   if (s == NULL) return BZ_MEM_ERROR;
   s->strm                  = strm;
   strm->state[0]              = s;
   s->state                 = BZ_X_MAGIC_1;
   s->bsLive                = 0;
   s->bsBuff                = 0;
   s->calculatedCombinedCRC = 0;
   strm->total_in_lo32      = 0;
   strm->total_in_hi32      = 0;
   strm->total_out_lo32     = 0;
   strm->total_out_hi32     = 0;
   s->smallDecompress       = (Bool)small;
   s->ll4                   = NULL;
   s->ll16                  = NULL;
   s->tt                    = NULL;
   s->currBlockNo           = 0;
   s->verbosity             = verbosity;

   return BZ_OK;
}

// /*---------------------------------------------------*/
// int BZ_API(cpu_cuda_BZ2_bzDecompress) ( cuda_bz_stream *strm )
// {
//    Bool    corrupt;
//    DState* s;
//    if (strm == NULL) return BZ_PARAM_ERROR;
//    s = ( DState*)strm->state[0];
//    if (s == NULL) return BZ_PARAM_ERROR;
//    if (s->strm != strm) return BZ_PARAM_ERROR;

//    while (True) {
//       if (s->state == BZ_X_IDLE) return BZ_SEQUENCE_ERROR;
//       if (s->state == BZ_X_OUTPUT) {
//          if (s->smallDecompress)
//             corrupt = unRLE_obuf_to_output_SMALL ( s ); else
//             corrupt = unRLE_obuf_to_output_FAST  ( s );
//          if (corrupt) return BZ_DATA_ERROR;
//          if (s->nblock_used == s->save_nblock+1 && s->state_out_len == 0) {
//             BZ_FINALISE_CRC ( s->calculatedBlockCRC );
//             if (s->verbosity >= 3) 
//                VPrintf2 ( " {0x%08x, 0x%08x}", s->storedBlockCRC, 
//                           s->calculatedBlockCRC );
//             if (s->verbosity >= 2) VPrintf0 ( "]" );
//             if (s->calculatedBlockCRC != s->storedBlockCRC)
//                return BZ_DATA_ERROR;
//             s->calculatedCombinedCRC 
//                = (s->calculatedCombinedCRC << 1) | 
//                     (s->calculatedCombinedCRC >> 31);
//             s->calculatedCombinedCRC ^= s->calculatedBlockCRC;
//             s->state = BZ_X_BLKHDR_1;
//          } else {
//             return BZ_OK;
//          }
//       }
//       if (s->state >= BZ_X_MAGIC_1) {
//          Int32 r = cpu_cuda_BZ2_decompress ( s );
//          if (r == BZ_STREAM_END) {
//             if (s->verbosity >= 3)
//                VPrintf2 ( "\n    combined CRCs: stored = 0x%08x, computed = 0x%08x", 
//                           s->storedCombinedCRC, s->calculatedCombinedCRC );
//             if (s->calculatedCombinedCRC != s->storedCombinedCRC)
//                return BZ_DATA_ERROR;
//             return r;
//          }
//          if (s->state != BZ_X_OUTPUT) return r;
//       }
//    }

//    AssertH ( 0, 6001 );

//    return 0;  /*NOTREACHED*/
// }

// /*---------------------------------------------------*/
// int BZ_API(cpu_cuda_BZ2_bzBuffToBuffDecompress) 
//                            ( char*         dest, 
//                              unsigned int* destLen,
//                              char*         source, 
//                              unsigned int  sourceLen,
//                              int           small,
//                              int           verbosity )
// {
//    cuda_bz_stream strm;
//    int ret;

//    if (dest == NULL || destLen == NULL || 
//        source == NULL ||
//        (small != 0 && small != 1) ||
//        verbosity < 0 || verbosity > 4) 
//           return BZ_PARAM_ERROR;

//    strm.bzalloc = NULL;
//    strm.bzfree = NULL;
//    strm.opaque = NULL;
//    ret = cpu_cuda_BZ2_bzDecompressInit ( &strm, verbosity, small );
//    if (ret != BZ_OK) return ret;

//    strm.next_in = source;
//    strm.next_out = dest;
//    strm.avail_in = sourceLen;
//    strm.avail_out = *destLen;

//    ret = cpu_cuda_BZ2_bzDecompress ( &strm );
//    if (ret == BZ_OK) goto output_overflow_or_eof;
//    if (ret != BZ_STREAM_END) goto errhandler;

//    /* normal termination */
//    *destLen -= strm.avail_out;
//    cuda_BZ2_bzDecompressEnd ( &strm );
//    return BZ_OK;

//    output_overflow_or_eof:
//    if (strm.avail_out > 0) {
//       cuda_BZ2_bzDecompressEnd ( &strm );
//       return BZ_UNEXPECTED_EOF;
//    } else {
//       cuda_BZ2_bzDecompressEnd ( &strm );
//       return BZ_OUTBUFF_FULL;
//    };      

//    errhandler:
//    cuda_BZ2_bzDecompressEnd ( &strm );
//    return ret; 
// }

/*---------------------------------------------------*/
/*--
   Code contributed by Yoshioka Tsuneo (tsuneo@rr.iij4u.or.jp)
   to support better zlib compatibility.
   This code is not _officially_ part of libbzip2 (yet);
   I haven't tested it, documented it, or considered the
   threading-safeness of it.
   If this code breaks, please contact both Yoshioka and me.
--*/
/*---------------------------------------------------*/

/*---------------------------------------------------*/
/*--
   return version like "0.9.5d, 4-Sept-1999".
--*/
const char * BZ_API(cuda_BZ2_bzlibVersion)(void)
{
   return BZ_VERSION;
}


#ifndef BZ_NO_STDIO
/*---------------------------------------------------*/

#if defined(_WIN32) || defined(OS2) || defined(MSDOS)
#   include <fcntl.h>
#   include <io.h>
#   define SET_BINARY_MODE(file) setmode(fileno(file),O_BINARY)
#else
#   define SET_BINARY_MODE(file)
#endif
static
BZFILE * bzopen_or_bzdopen
               ( const char *path,   /* no use when bzdopen */
                 int fd,             /* no use when bzdopen */
                 const char *mode,
                 int open_mode)      /* bzopen: 0, bzdopen:1 */
{
   int    bzerr;
   char   unused[BZ_MAX_UNUSED];
   int    blockSize100k = 9;
   int    writing       = 0;
   char   mode2[10]     = "";
   FILE   *fp           = NULL;
   BZFILE *bzfp         = NULL;
   int    verbosity     = 0;
   int    workFactor    = 30;
   int    smallMode     = 0;
   int    nUnused       = 0; 

   if (mode == NULL) return NULL;
   while (*mode) {
      switch (*mode) {
      case 'r':
         writing = 0; break;
      case 'w':
         writing = 1; break;
      case 's':
         smallMode = 1; break;
      default:
         if (isdigit((int)(*mode))) {
            blockSize100k = *mode-BZ_HDR_0;
         }
      }
      mode++;
   }
   strcat(mode2, writing ? "w" : "r" );
   strcat(mode2,"b");   /* binary mode */

   if (open_mode==0) {
      if (path==NULL || strcmp(path,"")==0) {
        fp = (writing ? stdout : stdin);
        SET_BINARY_MODE(fp);
      } else {
        fp = fopen(path,mode2);
      }
   } else {
#ifdef BZ_STRICT_ANSI
      fp = NULL;
#else
      fp = fdopen(fd,mode2);
#endif
   }
   if (fp == NULL) return NULL;

   if (writing) {
      /* Guard against total chaos and anarchy -- JRS */
      if (blockSize100k < 1) blockSize100k = 1;
      if (blockSize100k > 150) blockSize100k = 150; 
      bzfp = cuda_BZ2_bzWriteOpen(&bzerr,fp,blockSize100k,
                             verbosity, workFactor, 1 /* change */);
   } else {
      bzfp = cuda_BZ2_bzReadOpen(&bzerr,fp,verbosity,smallMode,
                            unused,nUnused);
   }
   if (bzfp == NULL) {
      if (fp != stdin && fp != stdout) fclose(fp);
      return NULL;
   }
   return bzfp;
}


/*---------------------------------------------------*/
/*--
   open file for read or write.
      ex) bzopen("file","w9")
      case path="" or NULL => use stdin or stdout.
--*/
BZFILE * BZ_API(cuda_BZ2_bzopen)
               ( const char *path,
                 const char *mode )
{
   return bzopen_or_bzdopen(path,-1,mode,/*bzopen*/0);
}


/*---------------------------------------------------*/
BZFILE * BZ_API(cuda_BZ2_bzdopen)
               ( int fd,
                 const char *mode )
{
   return bzopen_or_bzdopen(NULL,fd,mode,/*bzdopen*/1);
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzread) (BZFILE* b, void* buf, int len )
{
   int bzerr, nread;
   if (((bzFile*)b)->lastErr == BZ_STREAM_END) return 0;
   nread = cuda_BZ2_bzRead(&bzerr,b,buf,len);
   if (bzerr == BZ_OK || bzerr == BZ_STREAM_END) {
      return nread;
   } else {
      return -1;
   }
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzwrite) (BZFILE* b, void* buf, int len )
{
   int bzerr;

   cuda_BZ2_bzWrite(&bzerr,b,buf,len);
   if(bzerr == BZ_OK){
      return len;
   }else{
      return -1;
   }
}


/*---------------------------------------------------*/
int BZ_API(cuda_BZ2_bzflush) (BZFILE *b)
{
   /* do nothing now... */
   return 0;
}


/*---------------------------------------------------*/
void BZ_API(cuda_BZ2_bzclose) (BZFILE* b)
{
   int bzerr;
   FILE *fp;
   
   if (b==NULL) {return;}
   fp = ((bzFile *)b)->handle;
   if(((bzFile*)b)->writing){
      cuda_BZ2_bzWriteClose(&bzerr,b,0,NULL,NULL);
      if(bzerr != BZ_OK){
         cuda_BZ2_bzWriteClose(NULL,b,1,NULL,NULL);
      }
   }else{
      cuda_BZ2_bzReadClose(&bzerr,b);
   }
   if(fp!=stdin && fp!=stdout){
      fclose(fp);
   }
}


/*---------------------------------------------------*/
/*--
   return last error code 
--*/
static const char *bzerrorstrings[] = {
       "OK"
      ,"SEQUENCE_ERROR"
      ,"PARAM_ERROR"
      ,"MEM_ERROR"
      ,"DATA_ERROR"
      ,"DATA_ERROR_MAGIC"
      ,"IO_ERROR"
      ,"UNEXPECTED_EOF"
      ,"OUTBUFF_FULL"
      ,"CONFIG_ERROR"
      ,"???"   /* for future */
      ,"???"   /* for future */
      ,"???"   /* for future */
      ,"???"   /* for future */
      ,"???"   /* for future */
      ,"???"   /* for future */
};


const char * BZ_API(cuda_BZ2_bzerror) (BZFILE *b, int *errnum)
{
   int err = ((bzFile *)b)->lastErr;

   if(err>0) err = 0;
   *errnum = err;
   return bzerrorstrings[err*-1];
}
#endif


/*-------------------------------------------------------------*/
/*--- end                                           bzlib.c ---*/
/*-------------------------------------------------------------*/
