/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
/*
 * 合并操作
对通信组中的每个communicator，需要分别调用collective操作。
 当操作进入到cuda stream中排队时函数就返回。
 collective需要每个进程／线程进行独立操作；
 或者在单线程中使用组语句（ncclGroupStart/ncclGroupEnd）。
 支持in-place模式（sendbuf=recvbuf）

Collective communication operations must be called separately for each ommunicator in a communicator clique.
 They return when operations have been enqueued on the CUDA stream.
 Since they may perform inter-CPU synchronization,
 each call has to be done from a different thread or process,
 or need to use Group Semantics.
*/

/*
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
数据合并操作，将数据合并到root节点（root节点是rank的root，不是device的root）
Reduces data arrays of length count in sendbuff into recvbuff using op operation.
 recvbuff may be NULL on all calls except for root device.
 root is the rank (not the CUDA device) where data will reside after the operation is complete.
 In-place operation will happen if sendbuff == recvbuff.
 */
NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
