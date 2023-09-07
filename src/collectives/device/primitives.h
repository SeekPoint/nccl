/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)
/*为了方便理解，这里写下各个模板类型

/*
send:
UNROLL: 4,
SLICESPERCHUNK: 1,
SLICESTEPS: 1,
T: int8_t,
NRECV: 2,
NSEND: 1,
DIRECT: 1,
FUNC: FuncSum<int8_t>
recv:
UNROLL: 4,
SLICESPERCHUNK: 1,
SLICESTEPS: 1,
T: int8_t,
NRECV: 1,
NSEND: 2,
DIRECT: 1,
FUNC: FuncSum<int8_t>
*/
// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); } //sendPtr就是通过sendStep在buff里找到了接下来该使用的一块，即图五的某个黄框。004-005

/*
 * 在看实际数据发送之前，
 * 我们看下几个同步函数，barrier()用于同步整个发送或者接收线程，
 * subBarrier()负责同步发送/接收线程中的数据搬运线程（除去同步线程），
 * 其实就是通过不同的barrier同步不同的线程组。
 * */
  inline __device__ void barrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads+WARP_SIZE));
    }
  }
  inline __device__ void subBarrier() {
    if (NSEND>NRECV) {
      asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 4, %0;" :: "r"(nthreads));
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  /*队列的运行
然后分别从kernel视角和proxy视角看下这个队列如何运行的。
图三
对于kernel端，如图三，过程和单机一致，在搬运数据之前，
   通过判断sendConnHeadPtr和sendConnTailPtr之间的距离来判断队列是否已满，
   注意这里sendConnHead其实是sendConnTailPtr。
   */
  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + SLICESTEPS) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += SLICESTEPS;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += SLICESTEPS;
  }
  //每当新搬运了一块数据，就将sendConnTailPtr加一。
  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += SLICESTEPS;
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV>
  inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
    return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTSEND>
  inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
    return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
  }

    /*
  send:
  DIRECTRECV: 0
  DIRECTSEND: 1
  RECV: 0
  SEND: 1
  SRC: 1
  DST: 0
  dstPtr: NULL
     ===DIRECTSEND为1，但是sendDirectBuff为NULL，所以dsts等于sendPtr(i)
  */
  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    /*对于send操作的话，如果不是同步线程，需要执行上述waitSend的操作，直到可发送，由于只有第一个线程会执行waitSend，
     * 所以其他线程需要通过subBarrier等待第一个线程执行waitSend的过程，
     * 不然可能会出现buff已经满了，又开始发送导致数据覆盖的情况。
     * 然后通过ReduceOrCopyMulti将数据从src拷贝到dst，这个函数前边介绍过，不再赘述。
     * 接下来的barrier保证数据发送之后才更新队列指针信息，
     * 不然可能会出现队列指针已经更新，但是数据还没有拷贝结束的情况。
     * 然后通过incSend更新step。
     * 然后对于与同步线程，需要执行一下__threadfence_system再通过postSend更新tail指针，这是因为如果是机器间通信的话，
     * tail和buff可能是位于cpu的锁页内存，
     * 所以必须通过这个内存屏障保证网络通信线程在看到tail指针更新之后一定可以看到buff中正确的数据。
     * 由于postSend和执行内存屏障的线程可能不是同一个，
     * 所以这里需要通过__syncwarp同步一下当前warp。
     *
     *
     * 到这里基本就完成了单机内部ncclSend/ncclRecv的过程，
     * 主要就是两步，先通过peerlist将用户的操作记录下来，
     * 根据记录生成kernel所需要的参数，然后启动kernel执行拷贝即可。
     * 对于不同卡的情况，send将数据从用户指定的sendbuff拷贝到nccl p2p transport的buff，
     * recv将数据从buff拷贝到用户指定的recvbuff，
     * buff在这里其实就是一个fifo，nccl通过head，tail指针来完成对发送和接收过程的协调；
     * 对于同卡的情况直接通过kernel将数据从sendbuff拷贝到recvbuff即可。
     */
    bool syncThread = tid >= nthreads;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  //然后开始load recv的ncclConnInfo，保存下来recvBuff和step等信息，由于在p2p的setup过程中支持p2pread，
  // 因此conn->direct没有设置NCCL_DIRECT_GPU，所以不会进入第一个if。
  // 每个warp的第一个线程保存了ncclConnInfo，将recvConnTail和recvConnHead初始化为recvStep。
  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvBuff[i] = (const T*)conn->buffs[NCCL_PROTO_SIMPLE];
    recvStep[i] = conn->step;
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    recvDirectBuff[i] = NULL;
    if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *conn->ptrExchange = directBuff;
    }
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }

  /*第二个warp的第一个线程保存tail，并缓存tail的值；
同步线程的第一个线程保存了head
   */
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
    }
  }

  //然后load send的conn，保存step和sendBuff，每个warp的第一个线程保存conn，并将sendConnTail和sendConnHead初始化为step
  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    sendStep[i] = conn->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    sendDirectBuff[i] = NULL;
    if (DIRECT && (conn->direct & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = conn->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      barrier();
      if (tid == 0) *ptr = NULL;
    }
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  /*
   * 第一个线程保存了head，并缓存了head中值，
   * fifo是proxy用的，本节暂用不到；
   * 同步线程里的第一个线程保存了tail。
   *
然后我们来看下刚刚提到的这些变量都是干嘛的，在p2p transport的setup阶段，
   即第八节中讲的，每个rank都创建了用于协调发送接收过程的变量，
   如下所示，由于支持p2p read，所以buff位于发送端；tail位于接收端，发送端和接收端共同持有，由发送端更新，head位于发送端，
   发送端和接收端共同持有，由接收端进行更新；在ncclPrimitives的接收端，tail叫做recvConnTailPtr，head叫做recvConnHeadPtr；
   而在发送端，tail叫做sendConnTailPtr，head叫做sendConnHeadPtr。
  004-004.png
 然后看下这些变量是如何协调发送接收过程的
 004-005.png
中间黄色的框就是图四里标的buff，整个buff被划分为NCCL_STEP块，004-005.png 图五只画出来六块。
sendConnHead，sendConnTailPtr，sendStep由发送端更新，每次发送都会加一，这几个值其实是相等的（所以感觉这几个变量有些冗余）。
recvConnTail，recvConnHeadPtr，recvStep由接收端更新，每次接收都会加一，这几个值其实是相等的。

因此对于接收端，只要recvConnTail小于recvConnTailPtr，就表示有数据可以接收，并将recvConnTail加一表示又接收了一块数据。
inline __device__ void waitRecv()

对于发送端，只要sendConnHead大于sendConnenHeadPtr加NCCL_STEP就说明有剩余空间用来发送，并将sendConnHead加一表示又执行了一次发送。
  inline __device__ void waitSend(int nbytes)
   */
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
    }
    if (tid >= nthreads && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads && wid < nrecv) {
      recvConn->step = recvConnHead;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      __threadfence_system();
    }
  }
//先看下ncclPrimitives的构造函数，这里nthreads为160 - 32 = 128，其中32线程为同步线程。由于send的recvPeer为-1，所以send不会loadRecvConn，recv不会loadSendConn。
 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#include "prims_ll.h"
//#include "prims_ll128.h"

#endif
