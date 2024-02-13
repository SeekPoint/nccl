// https://blog.csdn.net/eloudy/article/details/135342579
/*

amd00@MZ32-00:~/yk_repo/NCCL/nccl/ex02$ make
g++ -g ex02.cpp -o ex02 -lnccl -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include
make: *** No rule to make target 'oneServer_multiDevice_multiThread.cpp', needed by 'oneServer_multiDevice_multiThread'.  Stop.
amd00@MZ32-00:~/yk_repo/NCCL/nccl/ex02$ make
make: *** No rule to make target 'oneServer_multiDevice_multiThread', needed by 'all'.  Stop.
amd00@MZ32-00:~/yk_repo/NCCL/nccl/ex02$ make
g++ -g ex02.cpp -o ex02 -lnccl -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include

amd00@MZ32-00:~/yk_repo/NCCL/nccl/ex02$ ./ex02


*/
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <time.h>
#include <sys/time.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void  get_seed(long long &seed)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  seed = (long long)tv.tv_sec * 1000*1000 + tv.tv_usec;//only second and usecond;
  printf("useconds:%lld\n", seed);
}

void  init_vector(float* A, int n)
{
  long long seed = 0;

  get_seed(seed);
  srand(seed);
  for(int i=0; i<n; i++)
  {
    A[i] = (rand()%100)/100.0f;
  }
}

void print_vector(float* A, float size)
{
  for(int i=0; i<size; i++)
    printf("%.2f ", A[i]);

  printf("\n");
}

void vector_add_vector(float* sum, float* A, int n)
{
  for(int i=0; i<n; i++)
  {
    sum[i] += A[i];
  }
}

int main(int argc, char* argv[])
{
  ncclComm_t comms[4];

  printf("ncclComm_t is a pointer type, sizeof(ncclComm_t)=%lu\n", sizeof(ncclComm_t));
  //managing 4 devices
  //int nDev = 4;
  int nDev = 2;
  //int size = 32*1024*1024;
  int size = 16*16;
  int devs[4] = { 0, 1, 2, 3 };

  float** sendbuff_host = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff_host = (float**)malloc(nDev * sizeof(float*));

  for(int dev=0; dev<nDev; dev++)
  {
    sendbuff_host[dev] = (float*)malloc(size*sizeof(float));
    recvbuff_host[dev] = (float*)malloc(size*sizeof(float));
    init_vector(sendbuff_host[dev], size);
    init_vector(recvbuff_host[dev], size);
  }

  //sigma(sendbuff_host[i]); i = 0, 1, ..., nDev-1
  float* result = (float*)malloc(size*sizeof(float));
  memset(result, 0, size*sizeof(float));

  for(int dev=0; dev<nDev; dev++)
  {
    vector_add_vector(result, sendbuff_host[dev], size);

    printf("sendbuff_host[%d]=\n", dev);
    print_vector(sendbuff_host[dev], size);
  }
  printf("result=\n");
  print_vector(result, size);

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[i], sendbuff_host[i], size*sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(recvbuff[i], recvbuff_host[i], size*sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  printf("blocked ncclAllReduce will be calleded\n");
  fflush(stdout);

  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));

  printf("blocked ncclAllReduce is calleded nDev =%d\n", nDev);
  fflush(stdout);
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(recvbuff_host[i], recvbuff[i], size*sizeof(float), cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  for(int i=0; i<nDev; i++) {
    printf("recvbuff_dev2host[%d]=\n", i);
    print_vector(recvbuff_host[i], size);
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}