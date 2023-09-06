/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
    const void* sendbuff;  // 用户指定要发送的数据buffer
    void* recvbuff;       // 用户指定的接收数据的buffer
    ssize_t sendbytes;    // sendbuff长度
    ssize_t recvbytes;    // recvbuff长度
};

struct ncclP2PConnect {
    int nrecv[MAXCHANNELS];  // nrecv[id]表示第id个channel会recv几个rank
    int nsend[MAXCHANNELS];  // nsend[id]表示第id个channel会send给几个rank
    int* recv;               // recv[id * nranks]开始的nrecv[id]个rank，表示第id个channel会从这几个rank recv
    int* send;               // send[id * nranks]开始的nsend[id]个rank，表示第id个channel会send给这几个rank
};

struct ncclP2Plist {
  struct ncclP2Pinfo *peerlist;
  int count;
  struct ncclP2PConnect connect;
};

#endif
