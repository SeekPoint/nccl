/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};
/*
 nccl现共有三个transport，P2P通过卡间p2p通信，SHM通过机器内共享的host内存通信，
 NET通过网络通信，nccl会依次通过这三个transport的canConnect判断是否可用，
 然后选择第一个可用的，由于rank 1不在当前机器，因此只有NET的recv可用，
 设置connector的transportComm为netTransport的recv。

 */
template <int type>
static ncclResult_t selectTransport(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(topo, graph, myInfo, peerInfo, connect, connector, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

//接下来接上节介绍下ncclTransportP2pSetup，由于当前rank为10，那么nrecv为1，peerRecv为1，nsend为1，peerSend为9；
//然后开始创建到1的通信，即通过selectTransport初始化peers[1].recv这个ncclConnector。
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t nSkippedSend = 0, nSkippedRecv = 0; /* for tracing */
  struct ncclConnect connect;
  struct ncclConnector* conn;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) { ++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<0>(comm->topo, graph, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->id));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) { ++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<1>(comm->topo, graph, comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->id));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) {++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, 1, comm->rank, conn));
    conn->connected = 1;
    CUDACHECK(cudaMemcpy(&channel->devPeers[peer].send, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
  }
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) {++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, 1, comm->rank, conn));
    conn->connected = 1;
    CUDACHECK(cudaMemcpy(&channel->devPeers[peer].recv, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
  }
  TRACE(NCCL_INIT, "nsend %d nrecv %d nSkippedSend %u nSkippedRecv %u - DONE", nsend, nrecv, nSkippedSend, nSkippedRecv);
  return ncclSuccess;
}


