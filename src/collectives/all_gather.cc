/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
/*
从其他节点接受数据并存储到本地recvbuf中。
 接收到的数据存储的偏移位置为i*sendcount（i为rank序列）
Each device gathers sendcount values from other GPUs into recvbuff,
 receiving data from rank i at offset i*sendcount.
 Assumes recvcount is equal to nranks*sendcount,
 which means that recvbuff should have a size of at least nranks*sendcount elements.
 In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
