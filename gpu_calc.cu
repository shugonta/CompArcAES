#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

#define MUL3(x) (x & 0x80 ? ((x << 1 ^0x1b) & 0xff ^x) : ((x << 1) ^ x))
#define MUL2(x) (x & 0x80 ? (x << 1 ^0x1b) & 0xff  : (x << 1))

#define BLOCKSIZE (128)
#define GRIDSIZE ((FILESIZE/16)/BLOCKSIZE) //128*26*32
#define Stream (64)
#define NBb2 (NBb << 1)

texture<int, 1, cudaReadModeElementType> pt_texture;
__constant__ int rkey[44];
__shared__ unsigned char SboxCUDA[256];
__constant__ unsigned char SboxCUDAConst[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__global__ void device_aes_encrypt(unsigned char *pt, unsigned char *ct) {

  //This kernel executes AES encryption on a GPU.
  //Please modify this kernel!!
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  /* if (thread_id == 0)
     printf("size = %ld\n", size);
 //  printf("You can use printf function to eliminate bugs in your kernel.\n");
 */
  memcpy(&(SboxCUDA[threadIdx.x << 1]), &(SboxCUDAConst[threadIdx.x << 1]), 2);
  __syncthreads();

  unsigned char cb[NBb2];
  int *cw = (int *) cb;

  cw[0] = ((int *) pt)[thread_id << 2] ^ rkey[0];
  cw[1] = ((int *) pt)[thread_id << 2 | 1] ^ rkey[1];
  cw[2] = ((int *) pt)[thread_id << 2 | 2] ^ rkey[2];
  cw[3] = ((int *) pt)[thread_id << 2 | 3] ^ rkey[3];

/*  cw[0] = tex1Dfetch(pt_texture, thread_id << 2) ^ rkey[0];
  cw[1] = tex1Dfetch(pt_texture, thread_id << 2 | 1) ^ rkey[1];
  cw[2] = tex1Dfetch(pt_texture, thread_id << 2 | 2) ^ rkey[2];
  cw[3] = tex1Dfetch(pt_texture, thread_id << 2 | 3) ^ rkey[3];*/
//round 1
  cw[4] = (MUL2(SboxCUDA[((unsigned char *) cw)[0]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[5]]) ^
           SboxCUDA[((unsigned char *) cw)[10]] ^
           SboxCUDA[((unsigned char *) cw)[15]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[5]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[10]]) ^
            SboxCUDA[((unsigned char *) cw)[15]] ^
            SboxCUDA[((unsigned char *) cw)[0]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[10]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[15]]) ^
            SboxCUDA[((unsigned char *) cw)[0]] ^
            SboxCUDA[((unsigned char *) cw)[5]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[15]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[0]]) ^
            SboxCUDA[((unsigned char *) cw)[5]] ^
            SboxCUDA[((unsigned char *) cw)[10]]) << 24)
          ^ rkey[4];

  cw[5] = (MUL2(SboxCUDA[((unsigned char *) cw)[4]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[9]]) ^
           SboxCUDA[((unsigned char *) cw)[14]] ^
           SboxCUDA[((unsigned char *) cw)[3]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[9]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[14]]) ^
            SboxCUDA[((unsigned char *) cw)[3]] ^
            SboxCUDA[((unsigned char *) cw)[4]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[14]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[3]]) ^
            SboxCUDA[((unsigned char *) cw)[4]] ^
            SboxCUDA[((unsigned char *) cw)[9]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[3]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[4]]) ^
            SboxCUDA[((unsigned char *) cw)[9]] ^
            SboxCUDA[((unsigned char *) cw)[14]]) << 24)
          ^ rkey[5];

  cw[6] = (MUL2(SboxCUDA[((unsigned char *) cw)[8]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[13]]) ^
           SboxCUDA[((unsigned char *) cw)[2]] ^
           SboxCUDA[((unsigned char *) cw)[7]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[13]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[2]]) ^
            SboxCUDA[((unsigned char *) cw)[7]] ^
            SboxCUDA[((unsigned char *) cw)[8]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[2]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[7]]) ^
            SboxCUDA[((unsigned char *) cw)[8]] ^
            SboxCUDA[((unsigned char *) cw)[13]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[7]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[8]]) ^
            SboxCUDA[((unsigned char *) cw)[13]] ^
            SboxCUDA[((unsigned char *) cw)[2]]) << 24)
          ^ rkey[6];

  cw[7] = (MUL2(SboxCUDA[((unsigned char *) cw)[12]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[1]]) ^
           SboxCUDA[((unsigned char *) cw)[6]] ^
           SboxCUDA[((unsigned char *) cw)[11]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[1]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[6]]) ^
            SboxCUDA[((unsigned char *) cw)[11]] ^
            SboxCUDA[((unsigned char *) cw)[12]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[6]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[11]]) ^
            SboxCUDA[((unsigned char *) cw)[12]] ^
            SboxCUDA[((unsigned char *) cw)[1]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[11]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[12]]) ^
            SboxCUDA[((unsigned char *) cw)[1]] ^
            SboxCUDA[((unsigned char *) cw)[6]]) << 24)
          ^ rkey[7];
//round 2
  cw[0] = (MUL2(SboxCUDA[((unsigned char *) cw)[16]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[21]]) ^
           SboxCUDA[((unsigned char *) cw)[26]] ^
           SboxCUDA[((unsigned char *) cw)[31]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[21]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[26]]) ^
            SboxCUDA[((unsigned char *) cw)[31]] ^
            SboxCUDA[((unsigned char *) cw)[16]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[26]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[31]]) ^
            SboxCUDA[((unsigned char *) cw)[16]] ^
            SboxCUDA[((unsigned char *) cw)[21]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[31]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[16]]) ^
            SboxCUDA[((unsigned char *) cw)[21]] ^
            SboxCUDA[((unsigned char *) cw)[26]]) << 24)
          ^ rkey[8];

  cw[1] = (MUL2(SboxCUDA[((unsigned char *) cw)[20]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[25]]) ^
           SboxCUDA[((unsigned char *) cw)[30]] ^
           SboxCUDA[((unsigned char *) cw)[19]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[25]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[30]]) ^
            SboxCUDA[((unsigned char *) cw)[19]] ^
            SboxCUDA[((unsigned char *) cw)[20]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[30]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[19]]) ^
            SboxCUDA[((unsigned char *) cw)[20]] ^
            SboxCUDA[((unsigned char *) cw)[25]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[19]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[20]]) ^
            SboxCUDA[((unsigned char *) cw)[25]] ^
            SboxCUDA[((unsigned char *) cw)[30]]) << 24)
          ^ rkey[9];

  cw[2] = (MUL2(SboxCUDA[((unsigned char *) cw)[24]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[29]]) ^
           SboxCUDA[((unsigned char *) cw)[18]] ^
           SboxCUDA[((unsigned char *) cw)[23]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[29]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[18]]) ^
            SboxCUDA[((unsigned char *) cw)[23]] ^
            SboxCUDA[((unsigned char *) cw)[24]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[18]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[23]]) ^
            SboxCUDA[((unsigned char *) cw)[24]] ^
            SboxCUDA[((unsigned char *) cw)[29]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[23]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[24]]) ^
            SboxCUDA[((unsigned char *) cw)[29]] ^
            SboxCUDA[((unsigned char *) cw)[18]]) << 24)
          ^ rkey[10];

  cw[3] = (MUL2(SboxCUDA[((unsigned char *) cw)[28]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[17]]) ^
           SboxCUDA[((unsigned char *) cw)[22]] ^
           SboxCUDA[((unsigned char *) cw)[27]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[17]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[22]]) ^
            SboxCUDA[((unsigned char *) cw)[27]] ^
            SboxCUDA[((unsigned char *) cw)[28]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[22]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[27]]) ^
            SboxCUDA[((unsigned char *) cw)[28]] ^
            SboxCUDA[((unsigned char *) cw)[17]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[27]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[28]]) ^
            SboxCUDA[((unsigned char *) cw)[17]] ^
            SboxCUDA[((unsigned char *) cw)[22]]) << 24)
          ^ rkey[11];

  //round 3
  cw[4] = (MUL2(SboxCUDA[((unsigned char *) cw)[0]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[5]]) ^
           SboxCUDA[((unsigned char *) cw)[10]] ^
           SboxCUDA[((unsigned char *) cw)[15]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[5]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[10]]) ^
            SboxCUDA[((unsigned char *) cw)[15]] ^
            SboxCUDA[((unsigned char *) cw)[0]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[10]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[15]]) ^
            SboxCUDA[((unsigned char *) cw)[0]] ^
            SboxCUDA[((unsigned char *) cw)[5]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[15]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[0]]) ^
            SboxCUDA[((unsigned char *) cw)[5]] ^
            SboxCUDA[((unsigned char *) cw)[10]]) << 24)
          ^ rkey[12];

  cw[5] = (MUL2(SboxCUDA[((unsigned char *) cw)[4]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[9]]) ^
           SboxCUDA[((unsigned char *) cw)[14]] ^
           SboxCUDA[((unsigned char *) cw)[3]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[9]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[14]]) ^
            SboxCUDA[((unsigned char *) cw)[3]] ^
            SboxCUDA[((unsigned char *) cw)[4]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[14]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[3]]) ^
            SboxCUDA[((unsigned char *) cw)[4]] ^
            SboxCUDA[((unsigned char *) cw)[9]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[3]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[4]]) ^
            SboxCUDA[((unsigned char *) cw)[9]] ^
            SboxCUDA[((unsigned char *) cw)[14]]) << 24)
          ^ rkey[13];

  cw[6] = (MUL2(SboxCUDA[((unsigned char *) cw)[8]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[13]]) ^
           SboxCUDA[((unsigned char *) cw)[2]] ^
           SboxCUDA[((unsigned char *) cw)[7]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[13]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[2]]) ^
            SboxCUDA[((unsigned char *) cw)[7]] ^
            SboxCUDA[((unsigned char *) cw)[8]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[2]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[7]]) ^
            SboxCUDA[((unsigned char *) cw)[8]] ^
            SboxCUDA[((unsigned char *) cw)[13]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[7]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[8]]) ^
            SboxCUDA[((unsigned char *) cw)[13]] ^
            SboxCUDA[((unsigned char *) cw)[2]]) << 24)
          ^ rkey[14];

  cw[7] = (MUL2(SboxCUDA[((unsigned char *) cw)[12]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[1]]) ^
           SboxCUDA[((unsigned char *) cw)[6]] ^
           SboxCUDA[((unsigned char *) cw)[11]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[1]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[6]]) ^
            SboxCUDA[((unsigned char *) cw)[11]] ^
            SboxCUDA[((unsigned char *) cw)[12]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[6]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[11]]) ^
            SboxCUDA[((unsigned char *) cw)[12]] ^
            SboxCUDA[((unsigned char *) cw)[1]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[11]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[12]]) ^
            SboxCUDA[((unsigned char *) cw)[1]] ^
            SboxCUDA[((unsigned char *) cw)[6]]) << 24)
          ^ rkey[15];
//round 4
  cw[0] = (MUL2(SboxCUDA[((unsigned char *) cw)[16]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[21]]) ^
           SboxCUDA[((unsigned char *) cw)[26]] ^
           SboxCUDA[((unsigned char *) cw)[31]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[21]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[26]]) ^
            SboxCUDA[((unsigned char *) cw)[31]] ^
            SboxCUDA[((unsigned char *) cw)[16]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[26]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[31]]) ^
            SboxCUDA[((unsigned char *) cw)[16]] ^
            SboxCUDA[((unsigned char *) cw)[21]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[31]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[16]]) ^
            SboxCUDA[((unsigned char *) cw)[21]] ^
            SboxCUDA[((unsigned char *) cw)[26]]) << 24)
          ^ rkey[16];

  cw[1] = (MUL2(SboxCUDA[((unsigned char *) cw)[20]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[25]]) ^
           SboxCUDA[((unsigned char *) cw)[30]] ^
           SboxCUDA[((unsigned char *) cw)[19]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[25]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[30]]) ^
            SboxCUDA[((unsigned char *) cw)[19]] ^
            SboxCUDA[((unsigned char *) cw)[20]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[30]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[19]]) ^
            SboxCUDA[((unsigned char *) cw)[20]] ^
            SboxCUDA[((unsigned char *) cw)[25]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[19]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[20]]) ^
            SboxCUDA[((unsigned char *) cw)[25]] ^
            SboxCUDA[((unsigned char *) cw)[30]]) << 24)
          ^ rkey[17];

  cw[2] = (MUL2(SboxCUDA[((unsigned char *) cw)[24]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[29]]) ^
           SboxCUDA[((unsigned char *) cw)[18]] ^
           SboxCUDA[((unsigned char *) cw)[23]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[29]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[18]]) ^
            SboxCUDA[((unsigned char *) cw)[23]] ^
            SboxCUDA[((unsigned char *) cw)[24]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[18]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[23]]) ^
            SboxCUDA[((unsigned char *) cw)[24]] ^
            SboxCUDA[((unsigned char *) cw)[29]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[23]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[24]]) ^
            SboxCUDA[((unsigned char *) cw)[29]] ^
            SboxCUDA[((unsigned char *) cw)[18]]) << 24)
          ^ rkey[18];

  cw[3] = (MUL2(SboxCUDA[((unsigned char *) cw)[28]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[17]]) ^
           SboxCUDA[((unsigned char *) cw)[22]] ^
           SboxCUDA[((unsigned char *) cw)[27]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[17]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[22]]) ^
            SboxCUDA[((unsigned char *) cw)[27]] ^
            SboxCUDA[((unsigned char *) cw)[28]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[22]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[27]]) ^
            SboxCUDA[((unsigned char *) cw)[28]] ^
            SboxCUDA[((unsigned char *) cw)[17]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[27]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[28]]) ^
            SboxCUDA[((unsigned char *) cw)[17]] ^
            SboxCUDA[((unsigned char *) cw)[22]]) << 24)
          ^ rkey[19];

  //round 5
  cw[4] = (MUL2(SboxCUDA[((unsigned char *) cw)[0]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[5]]) ^
           SboxCUDA[((unsigned char *) cw)[10]] ^
           SboxCUDA[((unsigned char *) cw)[15]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[5]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[10]]) ^
            SboxCUDA[((unsigned char *) cw)[15]] ^
            SboxCUDA[((unsigned char *) cw)[0]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[10]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[15]]) ^
            SboxCUDA[((unsigned char *) cw)[0]] ^
            SboxCUDA[((unsigned char *) cw)[5]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[15]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[0]]) ^
            SboxCUDA[((unsigned char *) cw)[5]] ^
            SboxCUDA[((unsigned char *) cw)[10]]) << 24)
          ^ rkey[20];

  cw[5] = (MUL2(SboxCUDA[((unsigned char *) cw)[4]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[9]]) ^
           SboxCUDA[((unsigned char *) cw)[14]] ^
           SboxCUDA[((unsigned char *) cw)[3]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[9]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[14]]) ^
            SboxCUDA[((unsigned char *) cw)[3]] ^
            SboxCUDA[((unsigned char *) cw)[4]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[14]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[3]]) ^
            SboxCUDA[((unsigned char *) cw)[4]] ^
            SboxCUDA[((unsigned char *) cw)[9]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[3]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[4]]) ^
            SboxCUDA[((unsigned char *) cw)[9]] ^
            SboxCUDA[((unsigned char *) cw)[14]]) << 24)
          ^ rkey[21];

  cw[6] = (MUL2(SboxCUDA[((unsigned char *) cw)[8]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[13]]) ^
           SboxCUDA[((unsigned char *) cw)[2]] ^
           SboxCUDA[((unsigned char *) cw)[7]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[13]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[2]]) ^
            SboxCUDA[((unsigned char *) cw)[7]] ^
            SboxCUDA[((unsigned char *) cw)[8]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[2]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[7]]) ^
            SboxCUDA[((unsigned char *) cw)[8]] ^
            SboxCUDA[((unsigned char *) cw)[13]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[7]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[8]]) ^
            SboxCUDA[((unsigned char *) cw)[13]] ^
            SboxCUDA[((unsigned char *) cw)[2]]) << 24)
          ^ rkey[22];

  cw[7] = (MUL2(SboxCUDA[((unsigned char *) cw)[12]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[1]]) ^
           SboxCUDA[((unsigned char *) cw)[6]] ^
           SboxCUDA[((unsigned char *) cw)[11]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[1]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[6]]) ^
            SboxCUDA[((unsigned char *) cw)[11]] ^
            SboxCUDA[((unsigned char *) cw)[12]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[6]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[11]]) ^
            SboxCUDA[((unsigned char *) cw)[12]] ^
            SboxCUDA[((unsigned char *) cw)[1]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[11]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[12]]) ^
            SboxCUDA[((unsigned char *) cw)[1]] ^
            SboxCUDA[((unsigned char *) cw)[6]]) << 24)
          ^ rkey[23];
//round 6
  cw[0] = (MUL2(SboxCUDA[((unsigned char *) cw)[16]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[21]]) ^
           SboxCUDA[((unsigned char *) cw)[26]] ^
           SboxCUDA[((unsigned char *) cw)[31]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[21]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[26]]) ^
            SboxCUDA[((unsigned char *) cw)[31]] ^
            SboxCUDA[((unsigned char *) cw)[16]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[26]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[31]]) ^
            SboxCUDA[((unsigned char *) cw)[16]] ^
            SboxCUDA[((unsigned char *) cw)[21]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[31]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[16]]) ^
            SboxCUDA[((unsigned char *) cw)[21]] ^
            SboxCUDA[((unsigned char *) cw)[26]]) << 24)
          ^ rkey[24];

  cw[1] = (MUL2(SboxCUDA[((unsigned char *) cw)[20]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[25]]) ^
           SboxCUDA[((unsigned char *) cw)[30]] ^
           SboxCUDA[((unsigned char *) cw)[19]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[25]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[30]]) ^
            SboxCUDA[((unsigned char *) cw)[19]] ^
            SboxCUDA[((unsigned char *) cw)[20]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[30]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[19]]) ^
            SboxCUDA[((unsigned char *) cw)[20]] ^
            SboxCUDA[((unsigned char *) cw)[25]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[19]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[20]]) ^
            SboxCUDA[((unsigned char *) cw)[25]] ^
            SboxCUDA[((unsigned char *) cw)[30]]) << 24)
          ^ rkey[25];

  cw[2] = (MUL2(SboxCUDA[((unsigned char *) cw)[24]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[29]]) ^
           SboxCUDA[((unsigned char *) cw)[18]] ^
           SboxCUDA[((unsigned char *) cw)[23]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[29]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[18]]) ^
            SboxCUDA[((unsigned char *) cw)[23]] ^
            SboxCUDA[((unsigned char *) cw)[24]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[18]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[23]]) ^
            SboxCUDA[((unsigned char *) cw)[24]] ^
            SboxCUDA[((unsigned char *) cw)[29]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[23]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[24]]) ^
            SboxCUDA[((unsigned char *) cw)[29]] ^
            SboxCUDA[((unsigned char *) cw)[18]]) << 24)
          ^ rkey[26];

  cw[3] = (MUL2(SboxCUDA[((unsigned char *) cw)[28]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[17]]) ^
           SboxCUDA[((unsigned char *) cw)[22]] ^
           SboxCUDA[((unsigned char *) cw)[27]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[17]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[22]]) ^
            SboxCUDA[((unsigned char *) cw)[27]] ^
            SboxCUDA[((unsigned char *) cw)[28]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[22]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[27]]) ^
            SboxCUDA[((unsigned char *) cw)[28]] ^
            SboxCUDA[((unsigned char *) cw)[17]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[27]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[28]]) ^
            SboxCUDA[((unsigned char *) cw)[17]] ^
            SboxCUDA[((unsigned char *) cw)[22]]) << 24)
          ^ rkey[27];

  //round 7
  cw[4] = (MUL2(SboxCUDA[((unsigned char *) cw)[0]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[5]]) ^
           SboxCUDA[((unsigned char *) cw)[10]] ^
           SboxCUDA[((unsigned char *) cw)[15]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[5]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[10]]) ^
            SboxCUDA[((unsigned char *) cw)[15]] ^
            SboxCUDA[((unsigned char *) cw)[0]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[10]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[15]]) ^
            SboxCUDA[((unsigned char *) cw)[0]] ^
            SboxCUDA[((unsigned char *) cw)[5]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[15]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[0]]) ^
            SboxCUDA[((unsigned char *) cw)[5]] ^
            SboxCUDA[((unsigned char *) cw)[10]]) << 24)
          ^ rkey[28];

  cw[5] = (MUL2(SboxCUDA[((unsigned char *) cw)[4]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[9]]) ^
           SboxCUDA[((unsigned char *) cw)[14]] ^
           SboxCUDA[((unsigned char *) cw)[3]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[9]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[14]]) ^
            SboxCUDA[((unsigned char *) cw)[3]] ^
            SboxCUDA[((unsigned char *) cw)[4]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[14]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[3]]) ^
            SboxCUDA[((unsigned char *) cw)[4]] ^
            SboxCUDA[((unsigned char *) cw)[9]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[3]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[4]]) ^
            SboxCUDA[((unsigned char *) cw)[9]] ^
            SboxCUDA[((unsigned char *) cw)[14]]) << 24)
          ^ rkey[29];

  cw[6] = (MUL2(SboxCUDA[((unsigned char *) cw)[8]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[13]]) ^
           SboxCUDA[((unsigned char *) cw)[2]] ^
           SboxCUDA[((unsigned char *) cw)[7]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[13]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[2]]) ^
            SboxCUDA[((unsigned char *) cw)[7]] ^
            SboxCUDA[((unsigned char *) cw)[8]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[2]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[7]]) ^
            SboxCUDA[((unsigned char *) cw)[8]] ^
            SboxCUDA[((unsigned char *) cw)[13]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[7]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[8]]) ^
            SboxCUDA[((unsigned char *) cw)[13]] ^
            SboxCUDA[((unsigned char *) cw)[2]]) << 24)
          ^ rkey[30];

  cw[7] = (MUL2(SboxCUDA[((unsigned char *) cw)[12]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[1]]) ^
           SboxCUDA[((unsigned char *) cw)[6]] ^
           SboxCUDA[((unsigned char *) cw)[11]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[1]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[6]]) ^
            SboxCUDA[((unsigned char *) cw)[11]] ^
            SboxCUDA[((unsigned char *) cw)[12]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[6]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[11]]) ^
            SboxCUDA[((unsigned char *) cw)[12]] ^
            SboxCUDA[((unsigned char *) cw)[1]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[11]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[12]]) ^
            SboxCUDA[((unsigned char *) cw)[1]] ^
            SboxCUDA[((unsigned char *) cw)[6]]) << 24)
          ^ rkey[31];
//round 8
  cw[0] = (MUL2(SboxCUDA[((unsigned char *) cw)[16]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[21]]) ^
           SboxCUDA[((unsigned char *) cw)[26]] ^
           SboxCUDA[((unsigned char *) cw)[31]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[21]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[26]]) ^
            SboxCUDA[((unsigned char *) cw)[31]] ^
            SboxCUDA[((unsigned char *) cw)[16]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[26]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[31]]) ^
            SboxCUDA[((unsigned char *) cw)[16]] ^
            SboxCUDA[((unsigned char *) cw)[21]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[31]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[16]]) ^
            SboxCUDA[((unsigned char *) cw)[21]] ^
            SboxCUDA[((unsigned char *) cw)[26]]) << 24)
          ^ rkey[32];

  cw[1] = (MUL2(SboxCUDA[((unsigned char *) cw)[20]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[25]]) ^
           SboxCUDA[((unsigned char *) cw)[30]] ^
           SboxCUDA[((unsigned char *) cw)[19]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[25]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[30]]) ^
            SboxCUDA[((unsigned char *) cw)[19]] ^
            SboxCUDA[((unsigned char *) cw)[20]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[30]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[19]]) ^
            SboxCUDA[((unsigned char *) cw)[20]] ^
            SboxCUDA[((unsigned char *) cw)[25]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[19]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[20]]) ^
            SboxCUDA[((unsigned char *) cw)[25]] ^
            SboxCUDA[((unsigned char *) cw)[30]]) << 24)
          ^ rkey[33];

  cw[2] = (MUL2(SboxCUDA[((unsigned char *) cw)[24]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[29]]) ^
           SboxCUDA[((unsigned char *) cw)[18]] ^
           SboxCUDA[((unsigned char *) cw)[23]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[29]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[18]]) ^
            SboxCUDA[((unsigned char *) cw)[23]] ^
            SboxCUDA[((unsigned char *) cw)[24]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[18]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[23]]) ^
            SboxCUDA[((unsigned char *) cw)[24]] ^
            SboxCUDA[((unsigned char *) cw)[29]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[23]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[24]]) ^
            SboxCUDA[((unsigned char *) cw)[29]] ^
            SboxCUDA[((unsigned char *) cw)[18]]) << 24)
          ^ rkey[34];

  cw[3] = (MUL2(SboxCUDA[((unsigned char *) cw)[28]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[17]]) ^
           SboxCUDA[((unsigned char *) cw)[22]] ^
           SboxCUDA[((unsigned char *) cw)[27]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[17]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[22]]) ^
            SboxCUDA[((unsigned char *) cw)[27]] ^
            SboxCUDA[((unsigned char *) cw)[28]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[22]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[27]]) ^
            SboxCUDA[((unsigned char *) cw)[28]] ^
            SboxCUDA[((unsigned char *) cw)[17]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[27]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[28]]) ^
            SboxCUDA[((unsigned char *) cw)[17]] ^
            SboxCUDA[((unsigned char *) cw)[22]]) << 24)
          ^ rkey[35];

  //round 9
  cw[4] = (MUL2(SboxCUDA[((unsigned char *) cw)[0]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[5]]) ^
           SboxCUDA[((unsigned char *) cw)[10]] ^
           SboxCUDA[((unsigned char *) cw)[15]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[5]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[10]]) ^
            SboxCUDA[((unsigned char *) cw)[15]] ^
            SboxCUDA[((unsigned char *) cw)[0]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[10]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[15]]) ^
            SboxCUDA[((unsigned char *) cw)[0]] ^
            SboxCUDA[((unsigned char *) cw)[5]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[15]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[0]]) ^
            SboxCUDA[((unsigned char *) cw)[5]] ^
            SboxCUDA[((unsigned char *) cw)[10]]) << 24)
          ^ rkey[36];

  cw[5] = (MUL2(SboxCUDA[((unsigned char *) cw)[4]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[9]]) ^
           SboxCUDA[((unsigned char *) cw)[14]] ^
           SboxCUDA[((unsigned char *) cw)[3]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[9]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[14]]) ^
            SboxCUDA[((unsigned char *) cw)[3]] ^
            SboxCUDA[((unsigned char *) cw)[4]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[14]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[3]]) ^
            SboxCUDA[((unsigned char *) cw)[4]] ^
            SboxCUDA[((unsigned char *) cw)[9]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[3]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[4]]) ^
            SboxCUDA[((unsigned char *) cw)[9]] ^
            SboxCUDA[((unsigned char *) cw)[14]]) << 24)
          ^ rkey[37];

  cw[6] = (MUL2(SboxCUDA[((unsigned char *) cw)[8]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[13]]) ^
           SboxCUDA[((unsigned char *) cw)[2]] ^
           SboxCUDA[((unsigned char *) cw)[7]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[13]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[2]]) ^
            SboxCUDA[((unsigned char *) cw)[7]] ^
            SboxCUDA[((unsigned char *) cw)[8]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[2]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[7]]) ^
            SboxCUDA[((unsigned char *) cw)[8]] ^
            SboxCUDA[((unsigned char *) cw)[13]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[7]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[8]]) ^
            SboxCUDA[((unsigned char *) cw)[13]] ^
            SboxCUDA[((unsigned char *) cw)[2]]) << 24)
          ^ rkey[38];

  cw[7] = (MUL2(SboxCUDA[((unsigned char *) cw)[12]]) ^
           MUL3(SboxCUDA[((unsigned char *) cw)[1]]) ^
           SboxCUDA[((unsigned char *) cw)[6]] ^
           SboxCUDA[((unsigned char *) cw)[11]]
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[1]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[6]]) ^
            SboxCUDA[((unsigned char *) cw)[11]] ^
            SboxCUDA[((unsigned char *) cw)[12]]) << 8
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[6]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[11]]) ^
            SboxCUDA[((unsigned char *) cw)[12]] ^
            SboxCUDA[((unsigned char *) cw)[1]]) << 16
           |
           (MUL2(SboxCUDA[((unsigned char *) cw)[11]]) ^
            MUL3(SboxCUDA[((unsigned char *) cw)[12]]) ^
            SboxCUDA[((unsigned char *) cw)[1]] ^
            SboxCUDA[((unsigned char *) cw)[6]]) << 24)
          ^ rkey[39];

  cb[0] = SboxCUDA[cb[16]];
  cb[1] = SboxCUDA[cb[21]];
  cb[2] = SboxCUDA[cb[26]];
  cb[3] = SboxCUDA[cb[31]];
  ((int *) ct)[thread_id << 2] = cw[0] ^ rkey[40];
  cb[4] = SboxCUDA[cb[20]];
  cb[5] = SboxCUDA[cb[25]];
  cb[6] = SboxCUDA[cb[30]];
  cb[7] = SboxCUDA[cb[19]];
  ((int *) ct)[thread_id << 2 | 1] = cw[1] ^ rkey[41];
  cb[8] = SboxCUDA[cb[24]];
  cb[9] = SboxCUDA[cb[29]];
  cb[10] = SboxCUDA[cb[18]];
  cb[11] = SboxCUDA[cb[23]];
  ((int *) ct)[thread_id << 2 | 2] = cw[2] ^ rkey[42];
  cb[12] = SboxCUDA[cb[28]];
  cb[13] = SboxCUDA[cb[17]];
  cb[14] = SboxCUDA[cb[22]];
  cb[15] = SboxCUDA[cb[27]];
  ((int *) ct)[thread_id << 2 | 3] = cw[3] ^ rkey[43];
/*  if (thread_id == 0) {
    printf("state0: 0x%x\n", ((int *) ct)[thread_id << 2]);
    printf("state1: 0x%x\n", ((int *) ct)[thread_id << 2|1]);
    printf("state2: 0x%x\n", ((int *) ct)[thread_id << 2|2]);
    printf("state3: 0x%x\n", ((int *) ct)[thread_id << 2|3]);
  }*/
}

void launch_aes_kernel(unsigned char *pt, int *rk, unsigned char *ct, long int size) {
  //This function launches the AES kernel.
  //Please modify this function for AES kernel.
  //In this function, you need to allocate the device memory and so on.
  unsigned char *d_ct, *d_pt;
//  int *d_pt;
  long int size2 = size >> 6;
  cudaStream_t stream[Stream];
  dim3 dim_grid(GRIDSIZE >> 6, 1, 1), dim_block(BLOCKSIZE, 1, 1);
  cudaHostRegister(pt, size, cudaHostRegisterDefault);
  cudaHostRegister(ct, size, cudaHostRegisterDefault);

  cudaMalloc((void **) &d_pt, size);
  cudaMalloc((void **) &d_ct, size);
  cudaMemcpyToSymbol(rkey, rk, 176);

  cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking);
  cudaMemcpyAsync(d_pt, pt, size2, cudaMemcpyHostToDevice, stream[0]);
//  cudaBindTexture(NULL, pt_texture, d_pt);

  int i;
  for (i = 0; i < Stream; i++) {
    device_aes_encrypt <<< dim_grid, dim_block, 0, stream[i] >>> (d_pt + size2 * i, d_ct + size2 * i);
    cudaMemcpyAsync(ct + size2 * i, d_ct + size2 * i, size2, cudaMemcpyDeviceToHost, stream[i]);
    if (i != Stream - 1) {
      cudaStreamCreateWithFlags(&stream[i + 1], cudaStreamNonBlocking);
      cudaMemcpyAsync(d_pt + size2 * (i + 1), pt + size2 * (i + 1), size2, cudaMemcpyHostToDevice, stream[i + 1]);
    }
  }

//  cudaUnbindTexture(pt_texture);
  int stm;
  for (stm = 0; stm < Stream; stm++) {
    cudaStreamDestroy(stream[stm]);
  }
  cudaHostUnregister(pt);
  cudaHostUnregister(ct);
  cudaFree(d_pt);
  cudaFree(d_ct);
}












