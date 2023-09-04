/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
/*
 * �ϲ�����
��ͨ�����е�ÿ��communicator����Ҫ�ֱ����collective������
 ���������뵽cuda stream���Ŷ�ʱ�����ͷ��ء�
 collective��Ҫÿ�����̣��߳̽��ж���������
 �����ڵ��߳���ʹ������䣨ncclGroupStart/ncclGroupEnd����
 ֧��in-placeģʽ��sendbuf=recvbuf��

Collective communication operations must be called separately for each ommunicator in a communicator clique.
 They return when operations have been enqueued on the CUDA stream.
 Since they may perform inter-CPU synchronization,
 each call has to be done from a different thread or process,
 or need to use Group Semantics.
*/

/*
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
���ݺϲ������������ݺϲ���root�ڵ㣨root�ڵ���rank��root������device��root��
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
