/*************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

/*
 * 通过 getPath 获取到 CPU 节点到自己的 path，然后设置 count 为 0，带宽为 LOC_WIDTH，type 为 PATH_LOC。

然后每次从 nodeList 中拿出一个节点 node，获取 node 到 baseNode 的路径 path，
 然后用 node 去更新和 node 相连的节点，遍历 node 的边 link，获取 link 对端节点 remNode，
 获取 remNode 到 baseNode 的路径 remPath，此时需要比较两个路径哪个更优，一个路径是原来的 remPath，
 另一个是 path+link 这个新路径，新路径的带宽 width 是 path 和 link 的带宽取个 min，
 如果 width 大于 remPath->width，那么 remPath 更新为 path+link。
 */
static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx\n", t, id);
  return ncclInternalError;
}

/*路径更新后需要计算 remPath 的 type，这里有个取巧的地方是上节设置边 type 和本节设置路径 type 是对应的，
 * 比如 LINK_PCI 等于 PATH_PIX，然后可以看到之前说的各种路径的 type 是怎么计算出来的。

首先计算当前 link 作为一条路径的 type，初始化为 link 的 type，比如这个边是 LINK_PCI，那么就是 LINK_PIX，
 如果 remPath 的 count 大于 3 的话 type 就会更新为 PATH_PXB（但是这里有个疑问是大于 3 可能也跨过了两个 PCIe switch），
 如果 link 有一端是 CPU，那么 type 进一步更新为 PATH_PHB，最后取个 max，remPath->type = std::max (path->type, type)。
 */
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
  }

  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  nextNodeList.count = 0;
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->width = LOC_WIDTH;
  basePath->type = PATH_LOC;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        float width = std::min(path->width, link->width);
        if (remPath->width < width) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }
          // Copy the rest of the path
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->width = width;

          // Start with path type = link type. PATH and LINK types are supposed to match.
          // Don't consider LINK_NET as we only care about the NIC->GPU path.
          int type = link->type == LINK_NET ? 0 : link->type;
          // Differentiate between one and multiple PCI switches
          if (type == PATH_PIX && (node->type == PCI || link->remNode->type == PCI) && remPath->count > 3) type = PATH_PXB;
          // Consider a path going through the CPU as PATH_PHB
          if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) type = PATH_PHB;
          // Ignore Power CPU in an NVLink path
          if (path->type == PATH_NVL && type == PATH_SYS && link->remNode->type == CPU &&
              link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) type = 0;

          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          // Disallow GPUs as intermediate steps for now
          if (remNode->type != GPU) {
            int i;
            for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
            if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
          }
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  char line[1024];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lX :", topoNodeTypeStr[node->type], node->id);
#else
  sprintf(line, "%s/%lX :", topoNodeTypeStr[node->type], node->id);
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        sprintf(line+offset, "--%s->%s/%lX", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type], remNode->id);
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%f)", line, node->paths[t][n].width);
#else
      sprintf(line+offset, "%s/%lX (%d/%f/%s) ", topoNodeTypeStr[t], system->nodes[t].nodes[n].id, node->paths[t][n].count, node->paths[t][n].width, topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  for (int i=0; i<system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes+i);
  }
  return ncclSuccess;
}

static ncclResult_t getLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu) {
  // Find the closest CPU to a GPU
  int minHops = 0;
  int localCpu = -1;
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  for (int c=0; c<system->nodes[CPU].count; c++) {
    int hops = paths[c].count;
    if (minHops == 0 || hops < minHops) {
      localCpu = c;
      minHops = hops;
    }
  }
  if (localCpu == -1) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  *retCpu = localCpu;
  return ncclSuccess;
}

static ncclResult_t addCpuStep(struct ncclTopoSystem* system, int c, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[CPU].nodes+c;
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;

  int l=0;
  // Node 1 -> CPU
  for (int i=0; i<srcNode->paths[CPU][c].count; i++) srcNode->paths[t2][i2].list[l++] = srcNode->paths[CPU][c].list[i];
  // CPU -> Node 2
  for (int i=0; i<cpuNode->paths[t2][i2].count; i++) srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type = std::max(srcNode->paths[CPU][c].type, cpuNode->paths[t2][i2].type);
  srcNode->paths[t2][i2].width = std::min(srcNode->paths[CPU][c].width, cpuNode->paths[t2][i2].width);
  return ncclSuccess;
}

// Remove/free paths for a given type
static void ncclTopoRemovePathType(struct ncclTopoSystem* system, int nodeType) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    // Remove links _to_ the given type
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      free(node->paths[nodeType]);
      node->paths[nodeType] = NULL;
    }
    // Remove links _from_ the given type
    for (int n=0; n<system->nodes[nodeType].count; n++) {
      struct ncclTopoNode* node = system->nodes[nodeType].nodes+n;
      free(node->paths[t]);
      node->paths[t] = NULL;
    }
  }
}

static const int levelsOldToNew[] = { PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS };
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) {
    int l = -1;
    if (disableEnv) {
      char* str = getenv(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1) l = 0;
      }
    }
    if (l == -1) {
      char* str = getenv(levelEnv);
      if (str) {
        for (int i=0; i<PATH_NET; i++) {
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // Old style numbering
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew)/sizeof(int) - 1;
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;
          l = levelsOldToNew[oldLevel];
        }
      }
    }
    if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
    *level = l >= 0 ? l : -2;
  }
  return ncclSuccess;
}

/*
 * 然后通过 ncclTopoCheckP2p 检查当前 GPU 节点和其他所有的 GPU 节点之间是否可以使用 p2p 通信，
 * 其实就是判断 gpu1 到 gpu2 的路径 type 是否满足 p2pLevel 的限制，
 * 默认 p2pLevel 是 PATH_SYS，如果用户没有通过环境变量设置的话就相当于没有限制，
 * 任意 gpu 之间都是支持 p2p 通信，另外如果路径类型为 PATH_NVL 的话，那么还支持 p2p read。
 *
 *
 *
 * 然后判断当前 GPU 和其他 GPU 是否可以通过 shm 通信，因为在 docker 环境中如果 shm 挂载的不一样就无法通信，
 * 如果无法通过 shm 通信的话就将 path 的 count 设置为 0，
 * 之后会删除掉对应节点（但是这里有个疑问，shm 不通的话为什么没有继续判断 p2p 是否可用）。

最后类似 GPU，然后对所有的 NIC 执行 ncclTopoSetPaths 计算出路径，
 然后遍历每个 NIC 和每个 GPU，判断是否支持 gdr。
 * */
int ncclTopoUserP2pLevel = -1;
ncclResult_t ncclTopoCheckP2p(struct ncclTopoSystem* system, int64_t id1, int64_t id2, int* p2p, int *read) {
  *p2p = 0;
  *read = 0;

  // Get GPUs from topology
  int g1, g2;
  NCCLCHECK(ncclTopoIdToIndex(system, GPU, id1, &g1));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes+g1;
  if (ncclTopoIdToIndex(system, GPU, id2, &g2) == ncclInternalError) {
    // GPU not found, we can't use p2p.
    return ncclSuccess;
  }
  struct ncclTopoLinkList* path = gpu1->paths[GPU]+g2;

  // In general, use P2P whenever we can.
  int p2pLevel = PATH_SYS;

  // User override
  if (ncclTopoUserP2pLevel == -1)
    NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));
  if (ncclTopoUserP2pLevel != -2) {
    p2pLevel = ncclTopoUserP2pLevel;
    goto compare;
  }

  // Don't use P2P through ARM CPUs
  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));
  if (arch == NCCL_TOPO_CPU_ARCH_ARM) p2pLevel = PATH_PXB;
  if (arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    if (model == NCCL_TOPO_CPU_TYPE_BDW) p2pLevel = PATH_PXB;
    else p2pLevel = PATH_PHB;
  }

compare:
  // Compute the PCI distance and compare with the p2pLevel.
  if (path->type <= p2pLevel) *p2p = 1;

  if (path->type == PATH_NVL) {
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes+g2;
    // Enable P2P Read for Ampere/NVLink only
    if ((gpu1->gpu.cudaCompCap == gpu2->gpu.cudaCompCap) && (gpu1->gpu.cudaCompCap == 80)) *read = 1;
  }

  return ncclSuccess;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
int ncclTopoUserGdrLevel = -1;
/*
 这里除了看之前判断是否支持 gdr 之外，还要看 GPU 和 NIC 之间的距离是否小于 netGdrLevel，netGdrLevel 默认是 PATH_PXB，用户也可以自定义，默认值为 PXB 的原因可见官方文档：

Even though the only theoretical requirement for GPUDirect RDMA to work between a third-party device and
 an NVIDIA GPU is that they share the same root complex, there exist bugs (mostly in chipsets)
 causing it to perform badly, or not work at all in certain setups.

We can distinguish between three situations, depending on what is on the path between the GPU and the third-party device:

PCIe switches only
single CPU/IOH
CPU/IOH <-> QPI/HT <-> CPU/IOH
The first situation, where there are only PCIe switches on the path, is optimal and yields the best performance.
 The second one, where a single CPU/IOH is involved, works,
 but yields worse performance ( especially peer-to-peer read bandwidth has been shown to be severely limited on some processor architectures ).
 Finally, the third situation, where the path traverses a QPI/HT link, may be extremely performance-limited or even not work reliably.

可以看到在只有经过 PCIe switch 的时候性能最好，在经过 CPU 的时候性能较差，在跨 numa 的时候性能很差，甚至不可用。

当 p2p 或者 gdr 不支持的时候，会通过 CPU 进行中转，通过 getLocalCpu 找到最近的 CPU c。

 然后 addCpuStep 将 i1 到 i2 的路径修改为 i1 到 c 的路径 + cpu 到 i2 的路径。

 */
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int64_t busId, int netDev, int read, int* useGdr) {
  *useGdr = 0;

  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netDev, &n));
  struct ncclTopoNode* net = system->nodes[NET].nodes+n;
  NCCLCHECK(ncclTopoIdToIndex(system, GPU, busId, &g));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;

  // Check that both the NIC and GPUs support it
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    if (gdrReadParam < 0) {
      int nvlink = 0;
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GPU.
      if (system->nodes[GPU].count == 1) nvlink = 1;
      for (int i=0; i<system->nodes[GPU].count; i++) {
        if (i == g) continue;
        if (gpu->paths[GPU][i].type == PATH_NVL) {
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = PATH_PXB;
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;
  int distance = gpu->paths[NET][n].type;
  if (distance > netGdrLevel) {
    INFO(NCCL_NET,"GPU Direct RDMA Disabled for GPU %lx / HCA %d (distance %d > %d)", busId, netDev, distance, netGdrLevel);
    return ncclSuccess;
  }

  *useGdr = 1;
  INFO(NCCL_NET,"GPU Direct RDMA Enabled for GPU %lx / HCA %d (distance %d <= %d), read %d", busId, netDev, distance, netGdrLevel, read);
  return ncclSuccess;
}

/*
首先通过 ncclTopoRemovePathType 将所有 node 中的 paths 清空。

ncclTopoSetPaths 作用就是计算出其他所有节点到 baseNode 的 path，这里遍历所有的 CPU 节点，
计算出其他所有节点到所有 CPU 节点的路径。

ncclTopoSetPaths 实现类似 SPFA，由于这个版本的 NCCL 不允许 GPU 作为路径的中间节点，
 所以在 SPFA 的过程中不会将 GPU 节点添加到队列中更新其他节点，
 相当于这个无向图没有环，因此这个场景下的 SPFA 过程也就相当于 BFS。

这里 baseNode 就是 CPU 节点，先分配 CPU 到 CPU path 的空间，nodeList 和 nextNodeList 就是队列的作用，
先将 baseNode 入队列。

getPath 函数是获取 node 中到 type 为 t 的第 id 个节点的路径 path。

如果 remNode 不是 GPU，那么将 remNode 添加到 nextNodeList，等 nodeList 遍历完之后，
将 nextNodeList 赋给 nodeList 继续遍历。

然后回到 ncclTopoComputePaths，还是使用 ncclTopoSetPaths 计算 GPU 节点到其他所有节点的距离。
 */
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclPeerInfo* peerInfos) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);

  // Set direct paths from/to CPUs. We need them in many cases.
  for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
  }

  // Set direct paths from/to GPUs.
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // Compute paths to GPU g
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));

    // Update path when we don't want to / can't use GPU Direct P2P
    for (int p=0; p<system->nodes[GPU].count; p++) {
      int p2p, read;
      NCCLCHECK(ncclTopoCheckP2p(system, system->nodes[GPU].nodes[p].id, system->nodes[GPU].nodes[g].id, &p2p, &read));
      if (p2p == 0) {
        // Divert all traffic through the CPU
        int cpu;
        NCCLCHECK(getLocalCpu(system, g, &cpu));
        NCCLCHECK(addCpuStep(system, cpu, GPU, p, GPU, g));
      }
    }

    if (peerInfos == NULL) continue;
    // Remove GPUs we can't talk to because of containers.
    struct ncclPeerInfo* dstInfo = peerInfos+system->nodes[GPU].nodes[g].gpu.rank;
    for (int p=0; p<system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = peerInfos+system->nodes[GPU].nodes[p].gpu.rank;
      int shm;
      NCCLCHECK(ncclTransports[TRANSPORT_SHM].canConnect(&shm, system, NULL, srcInfo, dstInfo));
      if (shm == 0) {
        // Mark this peer as inaccessible. We'll trim it later.
        system->nodes[GPU].nodes[p].paths[GPU][g].count = 0;
      }
    }
  }

  // Set direct paths from/to NICs.
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;
    NCCLCHECK(ncclTopoSetPaths(netNode, system));

    for (int g=0; g<system->nodes[GPU].count; g++) {
      // Update path when we dont want to / can't use GPU Direct RDMA.
      int gdr;
      NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].id, netNode->id, 0, &gdr));
      if (gdr == 0) {
        // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
        int localCpu;
        NCCLCHECK(getLocalCpu(system, g, &localCpu));
        NCCLCHECK(addCpuStep(system, localCpu, NET, n, GPU, g));
        NCCLCHECK(addCpuStep(system, localCpu, GPU, g, NET, n));
      }
    }
  }
  return ncclSuccess;
}

//接下来会通过 ncclTopoTrimSystem 删除图中不可达的 GPU 节点和用不到的 NIC。
/*
 * 首先通过类似并查集的思路将多个 GPU 节点合并成多个集合，myDomain 为当前 rank 的 GPU 所对应的集合号，
 * 然后将不属于 myDomain 集合的 GPU 节点在图中删除掉，
 * 最后判断下如果 comm 的 rank 数等于当前图中的 gpu 节点数，那么说明不需要网卡，所以也将网卡从图中删除。

得到新的图结构后再重新执行一次 ncclTopoComputePaths 就得到最终各个节点之间的路径了。
 */
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  int *domains;
  int64_t *ids;
  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECK(ncclCalloc(&ids, system->nodes[GPU].count));
  int myDomain = 0;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    domains[g] = g;
    ids[g] = gpu->id;
    for (int p=0; p<g; p++) {
      if (gpu->paths[GPU][p].count > 0) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }
    if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
  }

  int ngpus = system->nodes[GPU].count;
  for (int i=0; i<ngpus; i++) {
    if (domains[i] == myDomain) continue;
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g=0; g<system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes+g;
      if (gpu->id == ids[i]) break; else gpu=NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      free(domains);
      free(ids);
      return ncclInternalError;
    }
    NCCLCHECK(ncclTopoRemoveNode(system, GPU, g));
  }

  comm->localRanks = system->nodes[GPU].count;
  if (system->nodes[GPU].count == comm->nRanks) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
  }
  free(domains);
  free(ids);
  return ncclSuccess;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);
  free(system);
}

static ncclResult_t ncclTopoGetNchannels(struct ncclTopoSystem* system, int g /*local gpu index*/, int peerRank, int* nChannels) {
  int peer;
  struct ncclTopoLinkList* path = NULL;
  if (ncclTopoRankToIndex(system, peerRank, &peer) == ncclSuccess) {
    // Same rank
    if (g == peer) {
      *nChannels = -1;
      return ncclSuccess;
    }
    // Local rank
    path = system->nodes[GPU].nodes[peer].paths[GPU]+g;
    if (path->type == PATH_NVL) {
      int sm = system->nodes[GPU].nodes[g].gpu.cudaCompCap;
      double nvlWidth = sm < 70 ? PASCAL_NVLINK_WIDTH : VOLTA_NVLINK_WIDTH;
      *nChannels = 2*std::max(1, (int)(path->width / nvlWidth));
    } else {
      *nChannels = 2;
    }
  } else {
    // Remote rank, use network
    *nChannels = 1;
  }
  return ncclSuccess;
}

NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);

static int nextPow2(int v) {
  int pow2 = 1;
  while (pow2 < v) pow2 <<= 1;
  return pow2;
}

//p2p操作对应的channel如何创建出来的
/*
 之前在建立ringGraph的时候有搜索出一系列的环，并根据这些环建立了channel，假设现在一共有nChannels个channel，
 而p2p需要p2pnChannels个channel，那么如果p2pnChannels大于nChannles，会再创建p2pnChannels - nChannels个channel，
 其他的复用；否则直接复用即可。

对于每个send/recv操作，会使用p2pnChannelsPerPeer个channel并行发送/接收，那么当p2pnChannelsPerPeer比较小，
 p2pnChannels比较大，会导致只用了前边的几个channel，无法充分利用所有的channel，
 举个例子，p2pnChannelsPerPeer = 2，p2pnChannels = 32，rank0和rank1，rank2的通信都会使用channel[1]和channel[2]，
 为了解决这个问题，nccl使用数组p2pChannels[p2pnChannelsPerPeer]作为偏移，比如p2pChannels[0] = 0, p2pChannels[1] = 16，
 那么rank0和rank1的通信会使用channel[1]和channel[17]，rank0和rank2的通信会使用channel[2]和channel[18]，更充分的利用了channel。

为了方便理解，后续举例时假定p2pnChannels和p2pnChannelsPerPeer都为1。

 */
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) {
  comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
  comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  int minChannels = comm->p2pnChannels;
  // We need to loop through all local GPUs to have a global picture
  for (int g=0; g<comm->topo->nodes[GPU].count; g++) {
    for (int r=0; r<comm->nRanks; r++) {
      int nChannels;
      NCCLCHECK(ncclTopoGetNchannels(comm->topo, g, r, &nChannels));
      if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
    }
  }

  // Round to next pow2 nChannelsPerPeer and nChannels
  comm->p2pnChannelsPerPeer = nextPow2(minChannels);
  comm->p2pnChannels = nextPow2(comm->p2pnChannels);

  // Init channels that weren't used so far
  for (int c=comm->nChannels; c<comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));

  // We want to spread channels used when there aren't many and progressively
  // fill the whole space of nChannels. To do so we mirror the bits in the
  // nChannels space.
  for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
    int mirror = 0;
    for (int b=1, mb=(comm->p2pnChannels>>1); b<comm->p2pnChannels; b<<=1, mb>>=1) if (c & b) mirror |= mb;
    comm->p2pChannels[c] = mirror;
  }
  INFO(NCCL_INIT, "%d coll channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);
  return ncclSuccess;
}
