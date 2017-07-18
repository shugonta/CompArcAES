#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

__constant__ int rkey[44];
__constant__ unsigned char SboxCUDA[256] = {
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

__device__ void SubShift(int *state){
  unsigned char cb[NBb];
  cb[0] = SboxCUDA[((unsigned char *) state)[0]];
  cb[1] = SboxCUDA[((unsigned char *) state)[5]];
  cb[2] = SboxCUDA[((unsigned char *) state)[10]];
  cb[3] = SboxCUDA[((unsigned char *) state)[15]];
  cb[4] = SboxCUDA[((unsigned char *) state)[4]];
  cb[5] = SboxCUDA[((unsigned char *) state)[9]];
  cb[6] = SboxCUDA[((unsigned char *) state)[14]];
  cb[7] = SboxCUDA[((unsigned char *) state)[3]];
  cb[8] = SboxCUDA[((unsigned char *) state)[8]];
  cb[9] = SboxCUDA[((unsigned char *) state)[13]];
  cb[10] = SboxCUDA[((unsigned char *) state)[2]];
  cb[11] = SboxCUDA[((unsigned char *) state)[7]];
  cb[12] = SboxCUDA[((unsigned char *) state)[12]];
  cb[13] = SboxCUDA[((unsigned char *) state)[1]];
  cb[14] = SboxCUDA[((unsigned char *) state)[6]];
  cb[15] = SboxCUDA[((unsigned char *) state)[11]];
  memcpy(state, cb, sizeof(unsigned char) * NBb);
}
/*
__device__ void SubBytesCUDA(int *state) {
  unsigned char cb[NBb];
  cb[0] = SboxCUDA[((unsigned char *) state)[0]];
  cb[1] = SboxCUDA[((unsigned char *) state)[1]];
  cb[2] = SboxCUDA[((unsigned char *) state)[2]];
  cb[3] = SboxCUDA[((unsigned char *) state)[3]];
  cb[4] = SboxCUDA[((unsigned char *) state)[4]];
  cb[5] = SboxCUDA[((unsigned char *) state)[5]];
  cb[6] = SboxCUDA[((unsigned char *) state)[6]];
  cb[7] = SboxCUDA[((unsigned char *) state)[7]];
  cb[8] = SboxCUDA[((unsigned char *) state)[8]];
  cb[9] = SboxCUDA[((unsigned char *) state)[9]];
  cb[10] = SboxCUDA[((unsigned char *) state)[10]];
  cb[11] = SboxCUDA[((unsigned char *) state)[11]];
  cb[12] = SboxCUDA[((unsigned char *) state)[12]];
  cb[13] = SboxCUDA[((unsigned char *) state)[13]];
  cb[14] = SboxCUDA[((unsigned char *) state)[14]];
  cb[15] = SboxCUDA[((unsigned char *) state)[15]];
  memcpy(state, cb, sizeof(unsigned char) * NBb);
}

__device__ void ShiftRowsCUDA(int *state) {
  unsigned char cb[NBb];

  cb[0] = ((unsigned char *) state)[0];
  cb[1] = ((unsigned char *) state)[5];
  cb[2] = ((unsigned char *) state)[10];
  cb[3] = ((unsigned char *) state)[15];
  cb[4] = ((unsigned char *) state)[4];
  cb[5] = ((unsigned char *) state)[9];
  cb[6] = ((unsigned char *) state)[14];
  cb[7] = ((unsigned char *) state)[3];
  cb[8] = ((unsigned char *) state)[8];
  cb[9] = ((unsigned char *) state)[13];
  cb[10] = ((unsigned char *) state)[2];
  cb[11] = ((unsigned char *) state)[7];
  cb[12] = ((unsigned char *) state)[12];
  cb[13] = ((unsigned char *) state)[1];
  cb[14] = ((unsigned char *) state)[6];
  cb[15] = ((unsigned char *) state)[11];



*//*  cw[0] =
          ((unsigned char *) state)[0] |
          ((unsigned char *) state)[5] << 8 |
          ((unsigned char *) state)[10] << 16 |
          ((unsigned char *) state)[15] << 24;
  cw[1] =
          ((unsigned char *) state)[4] |
          ((unsigned char *) state)[9] << 8 |
          ((unsigned char *) state)[14] << 16 |
          ((unsigned char *) state)[3] << 24;
  cw[2] =
          ((unsigned char *) state)[8] |
          ((unsigned char *) state)[13] << 8 |
          ((unsigned char *) state)[2] << 16 |
          ((unsigned char *) state)[7] << 24;
  cw[3] =
          ((unsigned char *) state)[12] |
          ((unsigned char *) state)[1] << 8 |
          ((unsigned char *) state)[6] << 16 |
          ((unsigned char *) state)[11] << 24;*//*

  memcpy(state, cb, sizeof(int) * NB);
}*/

__device__ int mul3CUDA(unsigned char dt) {
  int x;
  x = dt << 1;
  if (x & 0x100)
    x = (x ^ 0x1b) & 0xff;
  x ^= dt;

  return (x);
}

__device__ int mul2CUDA(unsigned char dt) {
  int x;
  x = dt << 1;
  if (x & 0x100)
    x = (x ^ 0x1b) & 0xff;

  return (x);
}

__device__ void MixAddRound(int * state, int n){
  int cw[NB], key[NB];
  memcpy(key, &(rkey[n]), sizeof(int) * NB);
  memcpy(cw, state, sizeof(int) * NB);
  cw[0] =(mul2CUDA(((unsigned char *) cw)[0]) ^
          mul3CUDA(((unsigned char *) cw)[1]) ^
          ((unsigned char *) cw)[2] ^
          ((unsigned char *) cw)[3]
          |
          (mul2CUDA(((unsigned char *) cw)[1]) ^
           mul3CUDA(((unsigned char *) cw)[2]) ^
           ((unsigned char *) cw)[3] ^
           ((unsigned char *) cw)[0]) << 8
          |
          (mul2CUDA(((unsigned char *) cw)[2]) ^
           mul3CUDA(((unsigned char *) cw)[3]) ^
           ((unsigned char *) cw)[0] ^
           ((unsigned char *) cw)[1]) << 16
          |
          (mul2CUDA(((unsigned char *) cw)[3]) ^
           mul3CUDA(((unsigned char *) cw)[0]) ^
           ((unsigned char *) cw)[1] ^
           ((unsigned char *) cw)[2]) << 24)
         ^ key[0];

  cw[1] = (mul2CUDA(((unsigned char *) cw)[4]) ^
           mul3CUDA(((unsigned char *) cw)[5]) ^
           ((unsigned char *) cw)[6] ^
           ((unsigned char *) cw)[7]
           |
           (mul2CUDA(((unsigned char *) cw)[5]) ^
            mul3CUDA(((unsigned char *) cw)[6]) ^
            ((unsigned char *) cw)[7] ^
            ((unsigned char *) cw)[4]) << 8
           |
           (mul2CUDA(((unsigned char *) cw)[6]) ^
            mul3CUDA(((unsigned char *) cw)[7]) ^
            ((unsigned char *) cw)[4] ^
            ((unsigned char *) cw)[5]) << 16
           |
           (mul2CUDA(((unsigned char *) cw)[7]) ^
            mul3CUDA(((unsigned char *) cw)[4]) ^
            ((unsigned char *) cw)[5] ^
            ((unsigned char *) cw)[6]) << 24)
          ^ key[1];

  cw[2] = (mul2CUDA(((unsigned char *) cw)[8]) ^
           mul3CUDA(((unsigned char *) cw)[9]) ^
           ((unsigned char *) cw)[10] ^
           ((unsigned char *) cw)[11]
           |
           (mul2CUDA(((unsigned char *) cw)[9]) ^
            mul3CUDA(((unsigned char *) cw)[10]) ^
            ((unsigned char *) cw)[11] ^
            ((unsigned char *) cw)[8]) << 8
           |
           (mul2CUDA(((unsigned char *) cw)[10]) ^
            mul3CUDA(((unsigned char *) cw)[11]) ^
            ((unsigned char *) cw)[8] ^
            ((unsigned char *) cw)[9]) << 16
           |
           (mul2CUDA(((unsigned char *) cw)[11]) ^
            mul3CUDA(((unsigned char *) cw)[8]) ^
            ((unsigned char *) cw)[9] ^
            ((unsigned char *) cw)[10]) << 24)
          ^ key[2];

  cw[3] = (mul2CUDA(((unsigned char *) cw)[12]) ^
           mul3CUDA(((unsigned char *) cw)[13]) ^
           ((unsigned char *) cw)[14] ^
           ((unsigned char *) cw)[15]
           |
           (mul2CUDA(((unsigned char *) cw)[13]) ^
            mul3CUDA(((unsigned char *) cw)[14]) ^
            ((unsigned char *) cw)[15] ^
            ((unsigned char *) cw)[12]) << 8
           |
           (mul2CUDA(((unsigned char *) cw)[14]) ^
            mul3CUDA(((unsigned char *) cw)[15]) ^
            ((unsigned char *) cw)[12] ^
            ((unsigned char *) cw)[13]) << 16
           |
           (mul2CUDA(((unsigned char *) cw)[15]) ^
            mul3CUDA(((unsigned char *) cw)[12]) ^
            ((unsigned char *) cw)[13] ^
            ((unsigned char *) cw)[14]) << 24)
          ^ key[3];
  memcpy(state, cw, sizeof(int) * NB);
}

__device__ void MixColumnsCUDA(int *state) {
  int cw[NB];
  memcpy(cw, state, sizeof(int) * NB);
  cw[0] = mul2CUDA(((unsigned char *) cw)[0]) ^
          mul3CUDA(((unsigned char *) cw)[1]) ^
          ((unsigned char *) cw)[2] ^
          ((unsigned char *) cw)[3]
          |
          (mul2CUDA(((unsigned char *) cw)[1]) ^
           mul3CUDA(((unsigned char *) cw)[2]) ^
           ((unsigned char *) cw)[3] ^
           ((unsigned char *) cw)[0]) << 8
          |
          (mul2CUDA(((unsigned char *) cw)[2]) ^
           mul3CUDA(((unsigned char *) cw)[3]) ^
           ((unsigned char *) cw)[0] ^
           ((unsigned char *) cw)[1]) << 16
          |
          (mul2CUDA(((unsigned char *) cw)[3]) ^
           mul3CUDA(((unsigned char *) cw)[0]) ^
           ((unsigned char *) cw)[1] ^
           ((unsigned char *) cw)[2]) << 24;

  cw[1] = mul2CUDA(((unsigned char *) cw)[4]) ^
          mul3CUDA(((unsigned char *) cw)[5]) ^
          ((unsigned char *) cw)[6] ^
          ((unsigned char *) cw)[7]
          |
          (mul2CUDA(((unsigned char *) cw)[5]) ^
           mul3CUDA(((unsigned char *) cw)[6]) ^
           ((unsigned char *) cw)[7] ^
           ((unsigned char *) cw)[4]) << 8
          |
          (mul2CUDA(((unsigned char *) cw)[6]) ^
           mul3CUDA(((unsigned char *) cw)[7]) ^
           ((unsigned char *) cw)[4] ^
           ((unsigned char *) cw)[5]) << 16
          |
          (mul2CUDA(((unsigned char *) cw)[7]) ^
           mul3CUDA(((unsigned char *) cw)[4]) ^
           ((unsigned char *) cw)[5] ^
           ((unsigned char *) cw)[6]) << 24;

  cw[2] = mul2CUDA(((unsigned char *) cw)[8]) ^
          mul3CUDA(((unsigned char *) cw)[9]) ^
          ((unsigned char *) cw)[10] ^
          ((unsigned char *) cw)[11]
          |
          (mul2CUDA(((unsigned char *) cw)[9]) ^
           mul3CUDA(((unsigned char *) cw)[10]) ^
           ((unsigned char *) cw)[11] ^
           ((unsigned char *) cw)[8]) << 8
          |
          (mul2CUDA(((unsigned char *) cw)[10]) ^
           mul3CUDA(((unsigned char *) cw)[11]) ^
           ((unsigned char *) cw)[8] ^
           ((unsigned char *) cw)[9]) << 16
          |
          (mul2CUDA(((unsigned char *) cw)[11]) ^
           mul3CUDA(((unsigned char *) cw)[8]) ^
           ((unsigned char *) cw)[9] ^
           ((unsigned char *) cw)[10]) << 24;

  cw[3] = mul2CUDA(((unsigned char *) cw)[12]) ^
          mul3CUDA(((unsigned char *) cw)[13]) ^
          ((unsigned char *) cw)[14] ^
          ((unsigned char *) cw)[15]
          |
          (mul2CUDA(((unsigned char *) cw)[13]) ^
           mul3CUDA(((unsigned char *) cw)[14]) ^
           ((unsigned char *) cw)[15] ^
           ((unsigned char *) cw)[12]) << 8
          |
          (mul2CUDA(((unsigned char *) cw)[14]) ^
           mul3CUDA(((unsigned char *) cw)[15]) ^
           ((unsigned char *) cw)[12] ^
           ((unsigned char *) cw)[13]) << 16
          |
          (mul2CUDA(((unsigned char *) cw)[15]) ^
           mul3CUDA(((unsigned char *) cw)[12]) ^
           ((unsigned char *) cw)[13] ^
           ((unsigned char *) cw)[14]) << 24;
  memcpy(state, cw, sizeof(int) * NB);
}


__device__ void AddRoundKeyCUDA(int *state, int *w, int n) {
  int cw[NB], key[NB];
  memcpy(cw, state, sizeof(int) * NB);
  memcpy(key, &(rkey[n]), sizeof(int) * NB);
  cw[0] ^= key[0];
  cw[1] ^= key[1];
  cw[2] ^= key[2];
  cw[3] ^= key[3];
  memcpy(state, cw, sizeof(int) * NB);
}

__device__ void CipherCUDA(int *pt, unsigned char *ct, int *rkey) {
  int rnd, threadId = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int* state = pt;
  unsigned char cb[NBb], cb2[NBb];
  int* cw = (int *)cb;
  int* cw2 = (int *)cb2;
//  int state[NB];
//  memcpy(state, pt, sizeof(int) * NB);

  cw2[0] = state[0] ^ rkey[0];
  cw2[1] = state[1] ^ rkey[1];
  cw2[2] = state[2] ^ rkey[2];
  cw2[3] = state[3] ^ rkey[3];

//  round 1
  cw[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
           SboxCUDA[((unsigned char *) cw2)[10]] ^
           SboxCUDA[((unsigned char *) cw2)[15]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            SboxCUDA[((unsigned char *) cw2)[15]] ^
            SboxCUDA[((unsigned char *) cw2)[0]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            SboxCUDA[((unsigned char *) cw2)[0]] ^
            SboxCUDA[((unsigned char *) cw2)[5]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
            SboxCUDA[((unsigned char *) cw2)[5]] ^
            SboxCUDA[((unsigned char *) cw2)[10]]) << 24)
          ^ rkey[4];

  cw[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
           SboxCUDA[((unsigned char *) cw2)[14]] ^
           SboxCUDA[((unsigned char *) cw2)[3]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            SboxCUDA[((unsigned char *) cw2)[3]] ^
            SboxCUDA[((unsigned char *) cw2)[4]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            SboxCUDA[((unsigned char *) cw2)[4]] ^
            SboxCUDA[((unsigned char *) cw2)[9]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
            SboxCUDA[((unsigned char *) cw2)[9]] ^
            SboxCUDA[((unsigned char *) cw2)[14]]) << 24)
          ^ rkey[5];

  cw[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
           SboxCUDA[((unsigned char *) cw2)[2]] ^
           SboxCUDA[((unsigned char *) cw2)[7]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            SboxCUDA[((unsigned char *) cw2)[7]] ^
            SboxCUDA[((unsigned char *) cw2)[8]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            SboxCUDA[((unsigned char *) cw2)[8]] ^
            SboxCUDA[((unsigned char *) cw2)[13]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
            SboxCUDA[((unsigned char *) cw2)[13]] ^
            SboxCUDA[((unsigned char *) cw2)[2]]) << 24)
          ^ rkey[6];

  cw[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
           SboxCUDA[((unsigned char *) cw2)[6]] ^
           SboxCUDA[((unsigned char *) cw2)[11]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            SboxCUDA[((unsigned char *) cw2)[11]] ^
            SboxCUDA[((unsigned char *) cw2)[12]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            SboxCUDA[((unsigned char *) cw2)[12]] ^
            SboxCUDA[((unsigned char *) cw2)[1]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
            SboxCUDA[((unsigned char *) cw2)[1]] ^
            SboxCUDA[((unsigned char *) cw2)[6]]) << 24)
          ^ rkey[7];
//  round2
  cw2[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
            SboxCUDA[((unsigned char *) cw)[10]] ^
            SboxCUDA[((unsigned char *) cw)[15]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             SboxCUDA[((unsigned char *) cw)[15]] ^
             SboxCUDA[((unsigned char *) cw)[0]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             SboxCUDA[((unsigned char *) cw)[0]] ^
             SboxCUDA[((unsigned char *) cw)[5]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
             SboxCUDA[((unsigned char *) cw)[5]] ^
             SboxCUDA[((unsigned char *) cw)[10]]) << 24)
           ^ rkey[8];

  cw2[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
            SboxCUDA[((unsigned char *) cw)[14]] ^
            SboxCUDA[((unsigned char *) cw)[3]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             SboxCUDA[((unsigned char *) cw)[3]] ^
             SboxCUDA[((unsigned char *) cw)[4]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             SboxCUDA[((unsigned char *) cw)[4]] ^
             SboxCUDA[((unsigned char *) cw)[9]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
             SboxCUDA[((unsigned char *) cw)[9]] ^
             SboxCUDA[((unsigned char *) cw)[14]]) << 24)
           ^ rkey[9];

  cw2[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
            SboxCUDA[((unsigned char *) cw)[2]] ^
            SboxCUDA[((unsigned char *) cw)[7]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             SboxCUDA[((unsigned char *) cw)[7]] ^
             SboxCUDA[((unsigned char *) cw)[8]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             SboxCUDA[((unsigned char *) cw)[8]] ^
             SboxCUDA[((unsigned char *) cw)[13]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
             SboxCUDA[((unsigned char *) cw)[13]] ^
             SboxCUDA[((unsigned char *) cw)[2]]) << 24)
           ^ rkey[10];

  cw2[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
            SboxCUDA[((unsigned char *) cw)[6]] ^
            SboxCUDA[((unsigned char *) cw)[11]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             SboxCUDA[((unsigned char *) cw)[11]] ^
             SboxCUDA[((unsigned char *) cw)[12]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             SboxCUDA[((unsigned char *) cw)[12]] ^
             SboxCUDA[((unsigned char *) cw)[1]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
             SboxCUDA[((unsigned char *) cw)[1]] ^
             SboxCUDA[((unsigned char *) cw)[6]]) << 24)
           ^ rkey[11];

//  round 3
  cw[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
           SboxCUDA[((unsigned char *) cw2)[10]] ^
           SboxCUDA[((unsigned char *) cw2)[15]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            SboxCUDA[((unsigned char *) cw2)[15]] ^
            SboxCUDA[((unsigned char *) cw2)[0]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            SboxCUDA[((unsigned char *) cw2)[0]] ^
            SboxCUDA[((unsigned char *) cw2)[5]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
            SboxCUDA[((unsigned char *) cw2)[5]] ^
            SboxCUDA[((unsigned char *) cw2)[10]]) << 24)
          ^ rkey[12];

  cw[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
           SboxCUDA[((unsigned char *) cw2)[14]] ^
           SboxCUDA[((unsigned char *) cw2)[3]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            SboxCUDA[((unsigned char *) cw2)[3]] ^
            SboxCUDA[((unsigned char *) cw2)[4]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            SboxCUDA[((unsigned char *) cw2)[4]] ^
            SboxCUDA[((unsigned char *) cw2)[9]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
            SboxCUDA[((unsigned char *) cw2)[9]] ^
            SboxCUDA[((unsigned char *) cw2)[14]]) << 24)
          ^ rkey[13];

  cw[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
           SboxCUDA[((unsigned char *) cw2)[2]] ^
           SboxCUDA[((unsigned char *) cw2)[7]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            SboxCUDA[((unsigned char *) cw2)[7]] ^
            SboxCUDA[((unsigned char *) cw2)[8]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            SboxCUDA[((unsigned char *) cw2)[8]] ^
            SboxCUDA[((unsigned char *) cw2)[13]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
            SboxCUDA[((unsigned char *) cw2)[13]] ^
            SboxCUDA[((unsigned char *) cw2)[2]]) << 24)
          ^ rkey[13];

  cw[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
           SboxCUDA[((unsigned char *) cw2)[6]] ^
           SboxCUDA[((unsigned char *) cw2)[11]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            SboxCUDA[((unsigned char *) cw2)[11]] ^
            SboxCUDA[((unsigned char *) cw2)[12]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            SboxCUDA[((unsigned char *) cw2)[12]] ^
            SboxCUDA[((unsigned char *) cw2)[1]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
            SboxCUDA[((unsigned char *) cw2)[1]] ^
            SboxCUDA[((unsigned char *) cw2)[6]]) << 24)
          ^ rkey[15];

//  round 4
  cw2[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
            SboxCUDA[((unsigned char *) cw)[10]] ^
            SboxCUDA[((unsigned char *) cw)[15]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             SboxCUDA[((unsigned char *) cw)[15]] ^
             SboxCUDA[((unsigned char *) cw)[0]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             SboxCUDA[((unsigned char *) cw)[0]] ^
             SboxCUDA[((unsigned char *) cw)[5]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
             SboxCUDA[((unsigned char *) cw)[5]] ^
             SboxCUDA[((unsigned char *) cw)[10]]) << 24)
           ^ rkey[16];

  cw2[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
            SboxCUDA[((unsigned char *) cw)[14]] ^
            SboxCUDA[((unsigned char *) cw)[3]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             SboxCUDA[((unsigned char *) cw)[3]] ^
             SboxCUDA[((unsigned char *) cw)[4]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             SboxCUDA[((unsigned char *) cw)[4]] ^
             SboxCUDA[((unsigned char *) cw)[9]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
             SboxCUDA[((unsigned char *) cw)[9]] ^
             SboxCUDA[((unsigned char *) cw)[14]]) << 24)
           ^ rkey[17];

  cw2[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
            SboxCUDA[((unsigned char *) cw)[2]] ^
            SboxCUDA[((unsigned char *) cw)[7]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             SboxCUDA[((unsigned char *) cw)[7]] ^
             SboxCUDA[((unsigned char *) cw)[8]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             SboxCUDA[((unsigned char *) cw)[8]] ^
             SboxCUDA[((unsigned char *) cw)[13]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
             SboxCUDA[((unsigned char *) cw)[13]] ^
             SboxCUDA[((unsigned char *) cw)[2]]) << 24)
           ^ rkey[18];

  cw2[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
            SboxCUDA[((unsigned char *) cw)[6]] ^
            SboxCUDA[((unsigned char *) cw)[11]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             SboxCUDA[((unsigned char *) cw)[11]] ^
             SboxCUDA[((unsigned char *) cw)[12]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             SboxCUDA[((unsigned char *) cw)[12]] ^
             SboxCUDA[((unsigned char *) cw)[1]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
             SboxCUDA[((unsigned char *) cw)[1]] ^
             SboxCUDA[((unsigned char *) cw)[6]]) << 24)
           ^ rkey[19];

//  round 5
  cw[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
           SboxCUDA[((unsigned char *) cw2)[10]] ^
           SboxCUDA[((unsigned char *) cw2)[15]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            SboxCUDA[((unsigned char *) cw2)[15]] ^
            SboxCUDA[((unsigned char *) cw2)[0]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            SboxCUDA[((unsigned char *) cw2)[0]] ^
            SboxCUDA[((unsigned char *) cw2)[5]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
            SboxCUDA[((unsigned char *) cw2)[5]] ^
            SboxCUDA[((unsigned char *) cw2)[10]]) << 24)
          ^ rkey[20];

  cw[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
           SboxCUDA[((unsigned char *) cw2)[14]] ^
           SboxCUDA[((unsigned char *) cw2)[3]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            SboxCUDA[((unsigned char *) cw2)[3]] ^
            SboxCUDA[((unsigned char *) cw2)[4]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            SboxCUDA[((unsigned char *) cw2)[4]] ^
            SboxCUDA[((unsigned char *) cw2)[9]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
            SboxCUDA[((unsigned char *) cw2)[9]] ^
            SboxCUDA[((unsigned char *) cw2)[14]]) << 24)
          ^ rkey[21];

  cw[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
           SboxCUDA[((unsigned char *) cw2)[2]] ^
           SboxCUDA[((unsigned char *) cw2)[7]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            SboxCUDA[((unsigned char *) cw2)[7]] ^
            SboxCUDA[((unsigned char *) cw2)[8]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            SboxCUDA[((unsigned char *) cw2)[8]] ^
            SboxCUDA[((unsigned char *) cw2)[13]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
            SboxCUDA[((unsigned char *) cw2)[13]] ^
            SboxCUDA[((unsigned char *) cw2)[2]]) << 24)
          ^ rkey[22];

  cw[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
           SboxCUDA[((unsigned char *) cw2)[6]] ^
           SboxCUDA[((unsigned char *) cw2)[11]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            SboxCUDA[((unsigned char *) cw2)[11]] ^
            SboxCUDA[((unsigned char *) cw2)[12]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            SboxCUDA[((unsigned char *) cw2)[12]] ^
            SboxCUDA[((unsigned char *) cw2)[1]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
            SboxCUDA[((unsigned char *) cw2)[1]] ^
            SboxCUDA[((unsigned char *) cw2)[6]]) << 24)
          ^ rkey[23];

//  round 6
  cw2[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
            SboxCUDA[((unsigned char *) cw)[10]] ^
            SboxCUDA[((unsigned char *) cw)[15]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             SboxCUDA[((unsigned char *) cw)[15]] ^
             SboxCUDA[((unsigned char *) cw)[0]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             SboxCUDA[((unsigned char *) cw)[0]] ^
             SboxCUDA[((unsigned char *) cw)[5]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
             SboxCUDA[((unsigned char *) cw)[5]] ^
             SboxCUDA[((unsigned char *) cw)[10]]) << 24)
           ^ rkey[24];

  cw2[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
            SboxCUDA[((unsigned char *) cw)[14]] ^
            SboxCUDA[((unsigned char *) cw)[3]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             SboxCUDA[((unsigned char *) cw)[3]] ^
             SboxCUDA[((unsigned char *) cw)[4]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             SboxCUDA[((unsigned char *) cw)[4]] ^
             SboxCUDA[((unsigned char *) cw)[9]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
             SboxCUDA[((unsigned char *) cw)[9]] ^
             SboxCUDA[((unsigned char *) cw)[14]]) << 24)
           ^ rkey[25];

  cw2[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
            SboxCUDA[((unsigned char *) cw)[2]] ^
            SboxCUDA[((unsigned char *) cw)[7]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             SboxCUDA[((unsigned char *) cw)[7]] ^
             SboxCUDA[((unsigned char *) cw)[8]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             SboxCUDA[((unsigned char *) cw)[8]] ^
             SboxCUDA[((unsigned char *) cw)[13]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
             SboxCUDA[((unsigned char *) cw)[13]] ^
             SboxCUDA[((unsigned char *) cw)[2]]) << 24)
           ^ rkey[26];

  cw2[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
            SboxCUDA[((unsigned char *) cw)[6]] ^
            SboxCUDA[((unsigned char *) cw)[11]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             SboxCUDA[((unsigned char *) cw)[11]] ^
             SboxCUDA[((unsigned char *) cw)[12]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             SboxCUDA[((unsigned char *) cw)[12]] ^
             SboxCUDA[((unsigned char *) cw)[1]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
             SboxCUDA[((unsigned char *) cw)[1]] ^
             SboxCUDA[((unsigned char *) cw)[6]]) << 24)
           ^ rkey[27];

  //  round 7
  cw[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
           SboxCUDA[((unsigned char *) cw2)[10]] ^
           SboxCUDA[((unsigned char *) cw2)[15]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            SboxCUDA[((unsigned char *) cw2)[15]] ^
            SboxCUDA[((unsigned char *) cw2)[0]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            SboxCUDA[((unsigned char *) cw2)[0]] ^
            SboxCUDA[((unsigned char *) cw2)[5]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
            SboxCUDA[((unsigned char *) cw2)[5]] ^
            SboxCUDA[((unsigned char *) cw2)[10]]) << 24)
          ^ rkey[28];

  cw[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
           SboxCUDA[((unsigned char *) cw2)[14]] ^
           SboxCUDA[((unsigned char *) cw2)[3]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            SboxCUDA[((unsigned char *) cw2)[3]] ^
            SboxCUDA[((unsigned char *) cw2)[4]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            SboxCUDA[((unsigned char *) cw2)[4]] ^
            SboxCUDA[((unsigned char *) cw2)[9]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
            SboxCUDA[((unsigned char *) cw2)[9]] ^
            SboxCUDA[((unsigned char *) cw2)[14]]) << 24)
          ^ rkey[29];

  cw[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
           SboxCUDA[((unsigned char *) cw2)[2]] ^
           SboxCUDA[((unsigned char *) cw2)[7]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            SboxCUDA[((unsigned char *) cw2)[7]] ^
            SboxCUDA[((unsigned char *) cw2)[8]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            SboxCUDA[((unsigned char *) cw2)[8]] ^
            SboxCUDA[((unsigned char *) cw2)[13]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
            SboxCUDA[((unsigned char *) cw2)[13]] ^
            SboxCUDA[((unsigned char *) cw2)[2]]) << 24)
          ^ rkey[30];

  cw[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
           SboxCUDA[((unsigned char *) cw2)[6]] ^
           SboxCUDA[((unsigned char *) cw2)[11]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            SboxCUDA[((unsigned char *) cw2)[11]] ^
            SboxCUDA[((unsigned char *) cw2)[12]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            SboxCUDA[((unsigned char *) cw2)[12]] ^
            SboxCUDA[((unsigned char *) cw2)[1]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
            SboxCUDA[((unsigned char *) cw2)[1]] ^
            SboxCUDA[((unsigned char *) cw2)[6]]) << 24)
          ^ rkey[31];

//  round 8
  cw2[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
            SboxCUDA[((unsigned char *) cw)[10]] ^
            SboxCUDA[((unsigned char *) cw)[15]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[5]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             SboxCUDA[((unsigned char *) cw)[15]] ^
             SboxCUDA[((unsigned char *) cw)[0]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[10]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             SboxCUDA[((unsigned char *) cw)[0]] ^
             SboxCUDA[((unsigned char *) cw)[5]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[15]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[0]]) ^
             SboxCUDA[((unsigned char *) cw)[5]] ^
             SboxCUDA[((unsigned char *) cw)[10]]) << 24)
           ^ rkey[32];

  cw2[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
            SboxCUDA[((unsigned char *) cw)[14]] ^
            SboxCUDA[((unsigned char *) cw)[3]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[9]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             SboxCUDA[((unsigned char *) cw)[3]] ^
             SboxCUDA[((unsigned char *) cw)[4]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[14]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             SboxCUDA[((unsigned char *) cw)[4]] ^
             SboxCUDA[((unsigned char *) cw)[9]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[3]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[4]]) ^
             SboxCUDA[((unsigned char *) cw)[9]] ^
             SboxCUDA[((unsigned char *) cw)[14]]) << 24)
           ^ rkey[33];

  cw2[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
            SboxCUDA[((unsigned char *) cw)[2]] ^
            SboxCUDA[((unsigned char *) cw)[7]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[13]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             SboxCUDA[((unsigned char *) cw)[7]] ^
             SboxCUDA[((unsigned char *) cw)[8]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[2]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             SboxCUDA[((unsigned char *) cw)[8]] ^
             SboxCUDA[((unsigned char *) cw)[13]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[7]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[8]]) ^
             SboxCUDA[((unsigned char *) cw)[13]] ^
             SboxCUDA[((unsigned char *) cw)[2]]) << 24)
           ^ rkey[34];

  cw2[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
            SboxCUDA[((unsigned char *) cw)[6]] ^
            SboxCUDA[((unsigned char *) cw)[11]]
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[1]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             SboxCUDA[((unsigned char *) cw)[11]] ^
             SboxCUDA[((unsigned char *) cw)[12]]) << 8
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[6]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             SboxCUDA[((unsigned char *) cw)[12]] ^
             SboxCUDA[((unsigned char *) cw)[1]]) << 16
            |
            (mul2CUDA(SboxCUDA[((unsigned char *) cw)[11]]) ^
             mul3CUDA(SboxCUDA[((unsigned char *) cw)[12]]) ^
             SboxCUDA[((unsigned char *) cw)[1]] ^
             SboxCUDA[((unsigned char *) cw)[6]]) << 24)
           ^ rkey[35];

  //  round 9
  cw[0] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
           SboxCUDA[((unsigned char *) cw2)[10]] ^
           SboxCUDA[((unsigned char *) cw2)[15]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[5]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            SboxCUDA[((unsigned char *) cw2)[15]] ^
            SboxCUDA[((unsigned char *) cw2)[0]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[10]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            SboxCUDA[((unsigned char *) cw2)[0]] ^
            SboxCUDA[((unsigned char *) cw2)[5]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[15]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[0]]) ^
            SboxCUDA[((unsigned char *) cw2)[5]] ^
            SboxCUDA[((unsigned char *) cw2)[10]]) << 24)
          ^ rkey[36];

  cw[1] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
           SboxCUDA[((unsigned char *) cw2)[14]] ^
           SboxCUDA[((unsigned char *) cw2)[3]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[9]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            SboxCUDA[((unsigned char *) cw2)[3]] ^
            SboxCUDA[((unsigned char *) cw2)[4]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[14]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            SboxCUDA[((unsigned char *) cw2)[4]] ^
            SboxCUDA[((unsigned char *) cw2)[9]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[3]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[4]]) ^
            SboxCUDA[((unsigned char *) cw2)[9]] ^
            SboxCUDA[((unsigned char *) cw2)[14]]) << 24)
          ^ rkey[37];

  cw[2] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
           SboxCUDA[((unsigned char *) cw2)[2]] ^
           SboxCUDA[((unsigned char *) cw2)[7]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[13]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            SboxCUDA[((unsigned char *) cw2)[7]] ^
            SboxCUDA[((unsigned char *) cw2)[8]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[2]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            SboxCUDA[((unsigned char *) cw2)[8]] ^
            SboxCUDA[((unsigned char *) cw2)[13]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[7]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[8]]) ^
            SboxCUDA[((unsigned char *) cw2)[13]] ^
            SboxCUDA[((unsigned char *) cw2)[2]]) << 24)
          ^ rkey[38];

  cw[3] = (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
           mul3CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
           SboxCUDA[((unsigned char *) cw2)[6]] ^
           SboxCUDA[((unsigned char *) cw2)[11]]
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[1]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            SboxCUDA[((unsigned char *) cw2)[11]] ^
            SboxCUDA[((unsigned char *) cw2)[12]]) << 8
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[6]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            SboxCUDA[((unsigned char *) cw2)[12]] ^
            SboxCUDA[((unsigned char *) cw2)[1]]) << 16
           |
           (mul2CUDA(SboxCUDA[((unsigned char *) cw2)[11]]) ^
            mul3CUDA(SboxCUDA[((unsigned char *) cw2)[12]]) ^
            SboxCUDA[((unsigned char *) cw2)[1]] ^
            SboxCUDA[((unsigned char *) cw2)[6]]) << 24)
          ^ rkey[39];

  cb2[0] = SboxCUDA[cb[0]];
  cb2[1] = SboxCUDA[cb[5]];
  cb2[2] = SboxCUDA[cb[10]];
  cb2[3] = SboxCUDA[cb[15]];
  cw2[0] ^= rkey[40];
  cb2[4] = SboxCUDA[cb[4]];
  cb2[5] = SboxCUDA[cb[9]];
  cb2[6] = SboxCUDA[cb[14]];
  cb2[7] = SboxCUDA[cb[3]];
  cw2[1] ^= rkey[41];
  cb2[8] = SboxCUDA[cb[8]];
  cb2[9] = SboxCUDA[cb[13]];
  cb2[10] = SboxCUDA[cb[2]];
  cb2[11] = SboxCUDA[cb[7]];
  cw2[2] ^= rkey[42];
  cb2[12] = SboxCUDA[cb[12]];
  cb2[13] = SboxCUDA[cb[1]];
  cb2[14] = SboxCUDA[cb[6]];
  cb2[15] = SboxCUDA[cb[11]];
  cw2[3] ^= rkey[43];
  if (threadId == 0 ) {
    printf("cw0: 0x%x\n", cw[0]);
    printf("cw1: 0x%x\n", cw[1]);
    printf("cw2: 0x%x\n", cw[2]);
    printf("cw3: 0x%x\n", cw[3]);
  }
//  SubShift(cw);
//  SubBytesCUDA(state);
//  ShiftRowsCUDA(state);
//  AddRoundKeyCUDA(cw, rkey, rnd);
  memcpy(&ct[(((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x) << 4], cb2,
         sizeof(int) * NB);

  return;
}

__global__ void device_aes_encrypt(unsigned char *pt, unsigned char *ct, long int size) {

  //This kernel executes AES encryption on a GPU.
  //Please modify this kernel!!
  int thread_id = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

 /* if (thread_id == 0)
    printf("size = %ld\n", size);
//  printf("You can use printf function to eliminate bugs in your kernel.\n");
*/
  __shared__ int state[BLOCKSIZE][NB];
  memcpy(&(state[threadIdx.x][0]), &(pt[thread_id << 4]), sizeof(unsigned char) * NBb);
  CipherCUDA(&(state[threadIdx.x][0]), ct, rkey);
//  memcpy(&ct[thread_id << 4], &state[threadIdx.x], sizeof(unsigned char) * NBb);
}

void launch_aes_kernel(unsigned char *pt, int *rk, unsigned char *ct, long int size) {

  //This function launches the AES kernel.
  //Please modify this function for AES kernel.
  //In this function, you need to allocate the device memory and so on.
  unsigned char *d_pt, *d_ct;

  dim3 dim_grid(GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z), dim_block(BLOCKSIZE, 1, 1);

  cudaMalloc((void **) &d_pt, sizeof(unsigned char) * size);
//  cudaMalloc((void **) &d_rkey, sizeof(int) * 44);
  cudaMalloc((void **) &d_ct, sizeof(unsigned char) * size);

  cudaMemset(d_pt, 0, sizeof(unsigned char) * size);
  cudaMemcpy(d_pt, pt, sizeof(unsigned char) * size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(rkey, rk, sizeof(int) * 44);
//  cudaMemcpyToSymbol(state_org, pt, sizeof(unsigned char) * size);

  device_aes_encrypt <<< dim_grid, dim_block >>> (d_pt, d_ct, size);

  cudaMemcpy(ct, d_ct, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);

  cudaFree(d_pt);
  cudaFree(d_ct);
}












