/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "devcomm.h"
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>

// Define min for ssize_t
static __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

typedef uint64_t PackType;

// unpack x and y to elements of type T and apply FUNC to each element
template<class FUNC, typename T>
struct MULTI {
  __device__ PackType operator()(const PackType x, const PackType y) const;
};

template<class FUNC>
struct MULTI<FUNC, int8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint8_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of uint32_t.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    // for char, we do these as vector ops
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, int32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      int32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, uint32_t> {
  static_assert(sizeof(PackType) == 2 * sizeof(uint32_t),
      "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      uint32_t a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, half> {
  static_assert(sizeof(PackType) == 4 * sizeof(half),
      "PackType must be four times the size of half.");

  struct PackHalf2 {
    half2 a, b;
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    struct PackHalf2 cx, cy, cr;
    cx = *(reinterpret_cast<const struct PackHalf2*>(&x));
    cy = *(reinterpret_cast<const struct PackHalf2*>(&y));

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return *(reinterpret_cast<PackType*>(&cr));
  }
};

template<class FUNC>
struct MULTI<FUNC, float> {
  static_assert(sizeof(PackType) == 2 * sizeof(float),
      "PackType must be twice the size of float.");
  union converter {
    PackType storage;
    struct {
      float a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;

    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);

    return cr.storage;
  }
};

template<class FUNC>
struct MULTI<FUNC, double> {
  static_assert(sizeof(PackType) == sizeof(double),
      "PackType must be the same size as double.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    double rv = FUNC()(__longlong_as_double(x), __longlong_as_double(y));
    return __double_as_longlong(rv);
  }
};

template<class FUNC>
struct MULTI<FUNC, uint64_t> {
  static_assert(sizeof(PackType) == sizeof(uint64_t),
      "PackType must be the same size as uint64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    uint64_t rv = FUNC()(x, y);
    return rv;
  }
};

template<class FUNC>
struct MULTI<FUNC, int64_t> {
  static_assert(sizeof(PackType) == sizeof(int64_t),
      "PackType must be the same size as int64_t.");
  __device__ PackType operator()(const PackType x, const PackType y) const {
    int64_t rv = FUNC()((int64_t)x, (int64_t)y);
    return rv;
  }
};

template<typename T> inline __device__
T vFetch(const volatile T* ptr) {
  return *ptr;
}

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) {
  *ptr = val;
}

#if CUDART_VERSION < 9000
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r.x = ptr->x;
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ptr->x = val.x;
}
#else
template<> inline __device__
half vFetch<half>(const volatile half* ptr) {
  half r;
  r = ((half*)ptr)[0];
  return r;
}

template<> inline __device__
void vStore<half>(volatile half* ptr, const half val) {
  ((half*)ptr)[0] = val;
}
#endif

typedef ulong2 Pack128;

template<class FUNC, typename T>
struct MULTI128 {
  __device__ void operator()(Pack128& x, Pack128& y) {
    x.x = MULTI<FUNC, T>()(x.x, y.x);
    x.y = MULTI<FUNC, T>()(x.y, y.y);
  }
};

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}
inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

template<class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    const int offset, const int N) {
  for (int idx = offset+tid; idx < offset+N; idx += nthreads) {
    T val = vFetch(srcs[0]+idx);
    #pragma unroll
    for (int i=1; i<MINSRCS; i++) val = FUNC()(val, vFetch(srcs[i]+idx));
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) val = FUNC()(val, vFetch(srcs[i]+idx));

    #pragma unroll
    for (int i=0; i<MINDSTS; i++) vStore(dsts[i]+idx, val);
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) vStore(dsts[i]+idx, val);
  }
}

/*
 * ReduceCopy128bMulti使用向量化指令拷贝的过程，这里的load/store使用了内联PTX，
 * 不过感觉并没有必要。Fetch128就是从p指向的位置load一个ulong2到寄存器变量v里。
 * 这里有一个变量UNROLL，一个warp一次处理连续的UNROLL * WARP_SIZE个ulong2，其实就是类似循环展开的作用，
 * 当UNROLL为4的时候访存模式如下图 004-003.png，
 * 比如线程0的话会将4个黄框的第一个ulong2读取到寄存器变量vals，然后写到dst。
 *
特别的当UNROLL为1的时候，访存模式和ReduceCopyMulti类似，即128线程处理连续的128个ulong2，然后接着循环执行下一个128个ulong2。
 */
template<class FUNC, typename T, int UNROLL, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceCopy128bMulti( const int w, const int nw, const int t,
    int nsrcs, const T* s[MAXSRCS], int ndsts, T* d[MAXDSTS],
    const int elemOffset, const int Npack) {
  const int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;

  const Pack128* srcs[MAXSRCS];
  for (int i=0; i<MAXSRCS; i++) srcs[i] = ((const Pack128*)(s[i]+elemOffset))+offset;
  Pack128* dsts[MAXDSTS];
  for (int i=0; i<MAXDSTS; i++) dsts[i] = ((Pack128*)(d[i]+elemOffset))+offset;

  while (offset < Npack) {
    Pack128 vals[UNROLL];
    // Load and reduce
    for (int u = 0; u < UNROLL; ++u) Fetch128(vals[u], srcs[0]+u*WARP_SIZE);

    for (int i=1; i<MINSRCS; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }
    #pragma unroll 1
    for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) {
      Pack128 vals2[UNROLL];
      for (int u = 0; u < UNROLL; ++u) Fetch128(vals2[u], srcs[i]+u*WARP_SIZE);
      for (int u = 0; u < UNROLL; ++u) MULTI128<FUNC, T>()(vals[u], vals2[u]);
    }

    // Store
    for (int i = 0; i < MINDSTS; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    #pragma unroll 1
    for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) {
      for (int u = 0; u < UNROLL; ++u) Store128(dsts[i]+u*WARP_SIZE, vals[u]);
    }
    for (int i=0; i<MAXSRCS; i++) srcs[i] += inc;
    for (int i=0; i<MAXDSTS; i++) dsts[i] += inc;
    offset += inc;
  }
}

template <typename T>
__device__ int ptrAlign128(T* ptr) { return (uint64_t)ptr % alignof(Pack128); }

// Try to limit consecutive load/stores to 8.
// Use UNROLL 8 when we have a single source and a single destination, 4 otherwise
#define AUTOUNROLL (UNROLL*(4/(MINDSTS+MINSRCS)))
/*负责实际数据拷贝，将nsrcs个源数组通过FUNC规约后拷贝到ndsts个目标数组中，每个数组长度都为N。
 * ReduceOrCopyMulti会尝试使用128位向量化load/store来提高带宽利用率，并减少指令数量以提高性能，
 * 但是前提是待处理的数据是对齐的（16字节），如果src和dst不是16字节对齐的，但是对16取模后是一样的，
 * 那么可以先通过非向量化指令拷贝前面没对齐的数据，之后的数据就可以用向量化指令处理了；
 * 如果取模后也不一样，那就只能用非向量化指令进行拷贝了。
 * 整体分为三步骤，先处理前边未对齐的，然后处理中间对齐的数据，最后处理尾部数据。
 * ptrAlign128就是对16字节取模，首先通过异或判断srcs和dsts的首地址对齐是否一致，
 * 如果不一致，那么Npreamble = N，后续都需要用非向量化指令拷贝，
 否则Npreamble = (alignof(Pack128) - align) % alignof(Pack128)，
 即前面未对齐的一部分。
 */
template<int UNROLL, class FUNC, typename T, int MINSRCS, int MAXSRCS, int MINDSTS, int MAXDSTS>
__device__ __forceinline__ void ReduceOrCopyMulti(const int tid, const int nthreads,
    int nsrcs, const T* srcs[MAXSRCS], int ndsts, T* dsts[MAXDSTS],
    int N) {
  int Nrem = N;
  if (Nrem <= 0) return;

  int alignDiff = 0;
  int align = ptrAlign128(srcs[0]);
  #pragma unroll
  for (int i=1; i<MINSRCS; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  for (int i=MINSRCS; i<MAXSRCS && i<nsrcs; i++) alignDiff |= (align ^ ptrAlign128(srcs[i]));
  #pragma unroll
  for (int i=0; i<MINDSTS; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));
  for (int i=MINDSTS; i<MAXDSTS && i<ndsts; i++) alignDiff |= (align ^ ptrAlign128(dsts[i]));

  int Npreamble = alignDiff ? Nrem :
    N < alignof(Pack128) ? N :
    (alignof(Pack128) - align) % alignof(Pack128);

  // stage 1: preamble: handle any elements up to the point of everything coming
  // into alignment
  //对于未对齐的这部分数据，直接使用ReduceCopyMulti通过非向量化指令拷贝即可，128线程从src中读取连续的128个int8_t，然后存到dst，循环执行。访问模式如004-002.png
  if (Npreamble) {
    ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, 0, Npreamble);
    Nrem -= Npreamble;
    if (Nrem == 0) return;
  }
  int offset = Npreamble;

  // stage 2: fast path: use 128b loads/stores to do the bulk of the work,
  // assuming the pointers we have are all 128-bit alignable.
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  const int packFactor = sizeof(Pack128) / sizeof(T);

  // stage 2a: main loop
  //处理对齐的部分数据，这里分为两步，
  // 首先对于整除packFactor  * AUTOUNROLL * WARP_SIZE的部分数据可以开启AUTOUNROLL执行ReduceCopy128bMulti，
  // 对于剩余的部分设置AUTOUNROLL为1执行ReduceCopy128bMulti。
  int Npack2a = (Nrem / (packFactor * AUTOUNROLL * WARP_SIZE))
      * (AUTOUNROLL * WARP_SIZE); // round down
  int Nelem2a = Npack2a * packFactor;

  ReduceCopy128bMulti<FUNC, T, AUTOUNROLL, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2a);

  Nrem -= Nelem2a;
  if (Nrem == 0) return;
  offset += Nelem2a;

  // stage 2b: slightly less optimized for section when we don't have full
  // unrolling

  int Npack2b = Nrem / packFactor;
  int Nelem2b = Npack2b * packFactor;

  ReduceCopy128bMulti<FUNC, T, 1, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(w, nw, t, nsrcs, srcs, ndsts, dsts, offset, Npack2b);

  Nrem -= Nelem2b;
  if (Nrem == 0) return;
  offset += Nelem2b;

  // stage 2c: tail  最后对于不足packFactor，就是说最后凑不够128位的数据还是使用ReduceCopyMulti进行非向量化拷贝
  ReduceCopyMulti<FUNC, T, MINSRCS, MAXSRCS, MINDSTS, MAXDSTS>(tid, nthreads, nsrcs, srcs, ndsts, dsts, offset, Nrem);
}

#endif // COMMON_KERNEL_H_
