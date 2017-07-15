#ifndef __CALCULATION_H_INCLUDED__
#define __CALCULATION_H_INCLUDED__

#define NB (4)                        /* 128bit $B8GDj$H$7$F$N5,3J(B($B%G!<%?$ND9$5(B) */
#define NBb (16)

#define NK (4)                        /* 128-bit$B80%b!<%I$G$N80$ND9$5(B */
#define NR (10)                       /* $B%i%&%s%I?t(B */

/********************************************************************************/
// You have to set FILESIZE to "16*128*13*16*512" at the time of your submission.
// Please do not modify the other parts in this file.
/*******************************************************************************/
//#define FILESIZE (32)
#define FILESIZE (16*128*13*16*512)
#define BLOCKSIZE (512)
#define GRIDSIZE (128 * 13 * 16)


void SubBytes(int *);                 /* FIPS 197  P.16 Figure  6 */
void ShiftRows(int *);                /* FIPS 197  P.17 Figure  8 */
void MixColumns(int *);               /* FIPS 197  P.18 Figure  9 */
void AddRoundKey(int *, int *, int);  /* FIPS 197  P.19 Figure 10 */
int SubWord(int in);                  /* FIPS 197  P.20 Figure 11 */ /* FIPS 197  P.19  5.2 */
int RotWord(int in);                  /* FIPS 197  P.20 Figure 11 */ /* FIPS 197  P.19  5.2 */
void KeyExpansion(void *, int *);     /* FIPS 197  P.20 Figure 11 */
void Cipher(int *, int *);            /* FIPS 197  P.15 Figure  5 */

void launch_cpu_aes(unsigned char *in, int *rkey, unsigned char *out, long int size);
void launch_aes_kernel(unsigned char *pt, int *rk, unsigned char *ct, long int size);

//IDE用
#endif /* __CALCULATION_H_INCLUDED__ */

#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; int y; int z;};
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
#endif