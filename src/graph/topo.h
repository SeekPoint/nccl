/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include <sched.h>

#define LOC_WIDTH 5000.0
#define PASCAL_NVLINK_WIDTH 18.0
#define VOLTA_NVLINK_WIDTH 21.0
#define PCI_WIDTH 12.0           // PCI Gen3 x16
#define QPI_WIDTH 6.0
#define SKL_QPI_WIDTH 9.0
#define P9_WIDTH 32.0
#define ARM_WIDTH 6.0
#define NET_WIDTH 12.0           // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P(speed) (speed*9/12)
#define INTEL_P2P_OVERHEAD(speed) (speed*12/9)

#define NCCL_TOPO_NODE_TYPES 7
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
extern const char* topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
#define LINK_PCI 2
// Skipping 3 for PATH_PXB
// Skipping 4 for PATH_PHB
#define LINK_SYS 5
#define LINK_NET 6
extern const char* topoLinkTypeStr[];

/*其中 type 为路径的类型，一共有如下几种枚举值。
 * PATH_LOC ，
 * PATH_NVL
 * PATH_PIX
 * PATH_PXB
 * PATH_PHB
 * PATH_SYS
 * */
#define PATH_LOC 0 //为节点到自己
#define PATH_NVL 1 //表示路径上的边都是 NVLink，
#define PATH_PIX 2 //表示经过最多一个 PCIe switch，
#define PATH_PXB 3 //表示经过了多个 PCIe witch，但是没有经过 CPU，
#define PATH_PHB 4 //表示经过了 CPU，
#define PATH_SYS 5 //表示不同 numa 之间的路径。
#define PATH_NET 6
extern const char* topoPathTypeStr[];

struct ncclTopoNode;  //节点由ncclTopoNode表示
/*
图中的边由 ncclTopoLink
 */
struct ncclTopoLink {
  int type;    //区分边的类型，比如 NVLink，PCI
  float width; //表示带宽
  struct ncclTopoNode* remNode; //表示当前边连接的对端节点
};
#define NCCL_TOPO_MAX_LINKS 32
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)
/*
 * 最后计算出来节点之间的路径由 ncclTopoLinkList 表示，
 路径一共有 count 条边，这个路径的带宽是 width，
 即 count 条边中带宽最小为 width，list 为具体的边。
 */
struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  float width;
  int type;
};

#define NCCL_TOPO_CPU_INTEL_BDW 1
#define NCCL_TOPO_CPU_INTEL_SKL 2

#define NCCL_TOPO_UNDEF (-1)
/*每个节点由 ncclTopoNode 表示*/
struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      uint64_t asic;
      int port;
      float width;
      int gdrSupport;
      int collSupport;
      int maxChannels;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
  };
  //node1中的paths[type][id]就是node1到type类型的第id个node的路径
  int nlinks;  //表示该节点有几条边
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS]; //存储了具体连接的边
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES]; //存储了到其他节点的路径
  // Used during search
  uint64_t used;
};

//ncclTopoNodeSet 表示某种类型的所有节点，比如 GPU，PCI，NIC 等，ncclTopoSystem 存储了全局所有类型的节点。
struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  float maxWidth;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float width);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int64_t* id, int rr);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

#endif
