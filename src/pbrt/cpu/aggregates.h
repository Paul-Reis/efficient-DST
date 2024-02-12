// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_AGGREGATES_H
#define PBRT_CPU_AGGREGATES_H

#include <pbrt/pbrt.h>

#include <pbrt/cpu/primitive.h>
#include <pbrt/util/parallel.h>

#include <atomic>
#include <memory>
#include <vector>

namespace pbrt {

Primitive CreateAccelerator(const std::string &name, std::vector<Primitive> prims,
                            const ParameterDictionary &parameters);

struct BVHBuildNode;
struct BVHPrimitive;
struct LinearBVHNode;
struct MortonPrimitive;

// BVHAggregate Definition
class BVHAggregate {
  public:
    // BVHAggregate Public Types
    enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

    // BVHAggregate Public Methods
    BVHAggregate(std::vector<Primitive> p, int maxPrimsInNode = 1,
                 SplitMethod splitMethod = SplitMethod::SAH);

    static BVHAggregate *Create(std::vector<Primitive> prims,
                                const ParameterDictionary &parameters);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

    LinearBVHNode* getNodes();

  private:
    // BVHAggregate Private Methods
    BVHBuildNode *buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                 pstd::span<BVHPrimitive> bvhPrimitives,
                                 std::atomic<int> *totalNodes,
                                 std::atomic<int> *orderedPrimsOffset,
                                 std::vector<Primitive> &orderedPrims);
    BVHBuildNode *buildHLBVH(Allocator alloc,
                             const std::vector<BVHPrimitive> &primitiveInfo,
                             std::atomic<int> *totalNodes,
                             std::vector<Primitive> &orderedPrims);
    BVHBuildNode *emitLBVH(BVHBuildNode *&buildNodes,
                           const std::vector<BVHPrimitive> &primitiveInfo,
                           MortonPrimitive *mortonPrims, int nPrimitives, int *totalNodes,
                           std::vector<Primitive> &orderedPrims,
                           std::atomic<int> *orderedPrimsOffset, int bitIndex);
    BVHBuildNode *buildUpperSAH(Allocator alloc,
                                std::vector<BVHBuildNode *> &treeletRoots, int start,
                                int end, std::atomic<int> *totalNodes) const;
    int flattenBVH(BVHBuildNode *node, int *offset);

    // BVHAggregate Private Members
    int maxPrimsInNode;
    std::vector<Primitive> primitives;
    SplitMethod splitMethod;
    LinearBVHNode *nodes = nullptr;
};

struct KdTreeNode;
struct BoundEdge;

// KdTreeAggregate Definition
class KdTreeAggregate {
  public:
    // KdTreeAggregate Public Methods
    KdTreeAggregate(std::vector<Primitive> p, int isectCost = 5, int traversalCost = 1,
                    Float emptyBonus = 0.5, int maxPrims = 1, int maxDepth = -1);
    static KdTreeAggregate *Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters);
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;

    Bounds3f Bounds() const { return bounds; }

    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // KdTreeAggregate Private Methods
    void buildTree(int nodeNum, const Bounds3f &bounds,
                   const std::vector<Bounds3f> &primBounds,
                   pstd::span<const int> primNums, int depth,
                   std::vector<BoundEdge> edges[3], pstd::span<int> prims0,
                   pstd::span<int> prims1, int badRefines);

    // KdTreeAggregate Private Members
    int isectCost, traversalCost, maxPrims;
    Float emptyBonus;
    std::vector<Primitive> primitives;
    std::vector<int> primitiveIndices;
    KdTreeNode *nodes;
    int nAllocedNodes, nextFreeNode;
    Bounds3f bounds;
};

struct DSTBuildNode;

class DSTAggregate {
  public:
    // DSTAggregate Public Methods
    DSTAggregate(std::vector<Primitive> p, LinearBVHNode* nodes);

    static DSTAggregate *Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax);
    bool IntersectP(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax);

  private:
    // DSTAggregate Private Methodes
    DSTBuildNode *BuildRecursiveFromBVH(ThreadLocal<Allocator> &threadAllocators,
                                   LinearBVHNode* nodes,
                                   int currentNodeIndex, int currentDepth);
    void FlattenDST(DSTBuildNode *node);
    void addNodeToDepthList(DSTBuildNode *node);

    // DSTAggregate Private Members
    std::vector<Primitive> primitives;
    std::vector<uint32_t> linearDST;
    Bounds3f globalBB;

    std::vector<std::list<DSTBuildNode*>> nodesPerDepthLevel;
    int maximumDepth;
};

class StackItem {
  public:
    StackItem(int idx, float tMin, float tMax);
    StackItem();

    int idx;
    float tMin;
    float tMax;
};

float determineSAH(std::vector<int> &nodeComposition, std::vector<float> &nodesPlanes,
                   Bounds3f parentBB, Bounds3f childBB, int positionOfNextNode, float S);
float carveFiveSides(const std::vector<int> sidesToCarve,
                     std::vector<int> &nodeComposition, std::vector<float> &nodesPlanes,
                     int positionOfNextNode, float S, float Sn, Bounds3f parentBB,
                     Bounds3f childBB);
float carveThreeOrFourSides(const std::vector<int> sidesToCarve,
                            std::vector<int> &nodeComposition,
                            std::vector<float> &nodesPlanes, int positionOfNextNode,
                            float S, float Sn, Bounds3f parentBB, Bounds3f childBB);
float carveOneOrTwoSides(std::vector<int> sidesToCarve, std::vector<int> &nodeComposition,
                         std::vector<float> &nodesPlanes, int positionOfNextNode, float S,
                         float Sn, Bounds3f parentBB, Bounds3f childBB);

struct WDSTBuildNode;

class WDSTAggregate {
  public:
    WDSTAggregate(std::vector<Primitive> p, DSTBuildNode node, Bounds3f globalBB);

    static WDSTAggregate *Create(std::vector<Primitive> prims,
                                const ParameterDictionary &parameters);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    WDSTBuildNode *BuildWDSTRecursively(ThreadLocal<Allocator> &threadAllocators, DSTBuildNode *splittingNode, float S);
    void FlattenWDST(WDSTBuildNode *node);

    std::vector<Primitive> primitives;
    std::vector<uint32_t> linearWDST;
    Bounds3f globalBB;
};

Bounds3f carve(Bounds3f parentBB, Bounds3f childBB, std::vector<int> sidesToCarve,
               std::vector<float> &planes);

void printBVH(BVHBuildNode node, int depth);
void printDST(DSTBuildNode node, int depth);
}  // namespace pbrt

#endif  // PBRT_CPU_AGGREGATES_H
