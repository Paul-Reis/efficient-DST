#include "aggregates.h"
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/aggregates.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <tuple>
#include <iostream>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/BVH", treeBytes);
STAT_MEMORY_COUNTER("Memory/DST", dstBytes);
STAT_MEMORY_COUNTER("Memory/WDST", wdstBytes);

STAT_RATIO("BVH/Primitives per leaf node", totalPrimitives, totalLeafNodes);
STAT_COUNTER("BVH/Interior nodes", interiorNodes);
STAT_COUNTER("BVH/Leaf nodes", leafNodes);
STAT_COUNTER("SAH/BVH Cost * 1000000", bvhOverallSAHCost);//TODO: count SAH
STAT_PIXEL_COUNTER("BVH/Nodes visited", bvhNodesVisited);
STAT_RATIO("BVH/Average Plane Intersection Tests per Ray", bvhPlaneIntersections, bvhtRays);
STAT_RATIO("BVH/Average Triangle Intersection Tests per Ray", bvhTriangleIntersections, bvhRays1);
STAT_COUNTER("BVH/Rays that Intersect the Geometry", bvhRaysIntersect);

STAT_COUNTER("Accelerator/If 1 BVH, if 10 DST, if 100 WDST", testForDst);

STAT_RATIO("DST/Average Plane Intersection Tests per Ray", dstPlaneIntersections, dstRays);
STAT_RATIO("DST/Average Triangle Intersection Tests per Ray", dstTriangleIntersections, dstRays1);
STAT_COUNTER("SAH/DST Cost * 1000000", dstOverallSAHCost);
STAT_COUNTER("DST/Nodes", dstNumberOfNodes);
STAT_COUNTER("DST/Splitting Nodes", dstNumberOfSplittingNodes);
STAT_COUNTER("DST/Carving Nodes", dstNumberOfCarvingNodes);
STAT_COUNTER("DST/Leafs", dstNumberOfLeafs);
STAT_COUNTER("DST/Carving Node Sets", dstNumberOfCarvingNodeSets);
STAT_COUNTER("DST/Carving Node Sets Size 0", dstSetsWithZero);
STAT_COUNTER("DST/Carving Node Sets Size 1", dstSetsWithOne);
STAT_COUNTER("DST/Carving Node Sets Size 2", dstSetsWithTwo);
STAT_COUNTER("DST/Carving Node Sets Size 3", dstSetsWithThree);
STAT_COUNTER("DST/Single-Axis Carving Nodes", singleAxisCNs);
STAT_COUNTER("DST/Dual-Axis Carving Nodes", dualAxisCNs);
STAT_COUNTER("DST/Rays that Intersect the Geometry", dstRaysIntersect);

STAT_RATIO("WDST/Average Plane Intersection Tests per Ray", wdstPlaneIntersections, wdstRays);
STAT_RATIO("WDST/Average Triangle Intersection Tests per Ray", wdstTriangleIntersections, wdstRays1);
STAT_COUNTER("SAH/WDST Cost * 1000000", overallSAHCost);
STAT_COUNTER("WDST/Nodes", numberOfNodes);
STAT_COUNTER("WDST/Splitting Nodes", numberOfSplittingNodes);
STAT_COUNTER("WDST/Carving Nodes", numberOfCarvingNodes);
STAT_COUNTER("WDST/Leafs", numberOfLeafs);
STAT_COUNTER("WDST/Carving Node Sets", numberOfCarvingNodeSets);
STAT_COUNTER("WDST/Carving Node Sets Size 0", setsWithZero);
STAT_COUNTER("WDST/Carving Node Sets Size 1", setsWithOne);
STAT_COUNTER("WDST/Carving Node Sets Size 2", setsWithTwo);
STAT_COUNTER("WDST/Carving Node Sets Size 3", setsWithThree);
STAT_COUNTER("WDST/Splitting Nodes Size Small", smallSplittingNodes)
STAT_COUNTER("WDST/Splitting Nodes Size Medium", mediumSplittingNodes);
STAT_COUNTER("WDST/Splitting Nodes Size Big", bigSplittingNodes);
STAT_COUNTER("WDST/Rays that Intersect the Geometry", wdstRaysIntersect);


uint32_t IS_LEAF = 0b00000100000000000000000000000000;
uint32_t IS_CARVING = 0b10000000000000000000000000000000;
uint32_t SPLITTING_PLANE_AXIS = 0b01100000000000000000000000000000;
uint32_t CARVING_PLANE_AXIS = 0b00011000000000000000000000000000;
uint32_t CARVING_TYPE = 0b11100000000000000000000000000000;
uint32_t OFFSET = 0b00000011111111111111111111111111;
uint32_t CARVING_PLANE_AXES = SPLITTING_PLANE_AXIS;
uint32_t CORNER_TYPE = CARVING_PLANE_AXIS;
float TRAVERSAL_COST_DST_SN = 0.5;
float TRAVERSAL_COST_WDST_SMALL_SN = 0.5;
float TRAVERSAL_COST_WDST_MEDIUM_SN = 1;
float TRAVERSAL_COST_WDST_BIG_SN = 1.5;
float TRAVERSAL_COST_BVH_NODE = 1.5;

// MortonPrimitive Definition
struct MortonPrimitive {
    int primitiveIndex;
    uint32_t mortonCode;
};

// LBVHTreelet Definition
struct LBVHTreelet {
    size_t startIndex, nPrimitives;
    BVHBuildNode *buildNodes;
};

// BVHAggregate Utility Functions
static void RadixSort(std::vector<MortonPrimitive> *v) {
    std::vector<MortonPrimitive> tempVector(v->size());
    constexpr int bitsPerPass = 6;
    constexpr int nBits = 30;
    static_assert((nBits % bitsPerPass) == 0,
                  "Radix sort bitsPerPass must evenly divide nBits");
    constexpr int nPasses = nBits / bitsPerPass;
    for (int pass = 0; pass < nPasses; ++pass) {
        // Perform one pass of radix sort, sorting _bitsPerPass_ bits
        int lowBit = pass * bitsPerPass;
        // Set in and out vector references for radix sort pass
        std::vector<MortonPrimitive> &in = (pass & 1) ? tempVector : *v;
        std::vector<MortonPrimitive> &out = (pass & 1) ? *v : tempVector;

        // Count number of zero bits in array for current radix sort bit
        constexpr int nBuckets = 1 << bitsPerPass;
        int bucketCount[nBuckets] = {0};
        constexpr int bitMask = (1 << bitsPerPass) - 1;
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            CHECK_GE(bucket, 0);
            CHECK_LT(bucket, nBuckets);
            ++bucketCount[bucket];
        }

        // Compute starting index in output array for each bucket
        int outIndex[nBuckets];
        outIndex[0] = 0;
        for (int i = 1; i < nBuckets; ++i)
            outIndex[i] = outIndex[i - 1] + bucketCount[i - 1];

        // Store sorted values in output array
        for (const MortonPrimitive &mp : in) {
            int bucket = (mp.mortonCode >> lowBit) & bitMask;
            out[outIndex[bucket]++] = mp;
        }
    }
    // Copy final result from _tempVector_, if needed
    if (nPasses & 1)
        std::swap(*v, tempVector);
}

// BVHSplitBucket Definition
struct BVHSplitBucket {
    int count = 0;
    Bounds3f bounds;
};

// BVHPrimitive Definition
struct BVHPrimitive {
    BVHPrimitive() {}
    BVHPrimitive(size_t primitiveIndex, const Bounds3f &bounds)
        : primitiveIndex(primitiveIndex), bounds(bounds) {}
    size_t primitiveIndex;
    Bounds3f bounds;
    // BVHPrimitive Public Methods
    Point3f Centroid() const { return .5f * bounds.pMin + .5f * bounds.pMax; }
};

// BVHBuildNode Definition
struct BVHBuildNode {
    // BVHBuildNode Public Methods
    void InitLeaf(int first, int n, const Bounds3f &b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
        ++leafNodes;
        ++totalLeafNodes;
        totalPrimitives += n;
    }

    void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
        ++interiorNodes;
    }

    Bounds3f bounds;
    BVHBuildNode *children[2];
    int splitAxis, firstPrimOffset, nPrimitives;
};

// LinearBVHNode Definition
struct alignas(32) LinearBVHNode {
    Bounds3f bounds;
    union {
        int primitivesOffset;   // leaf
        int secondChildOffset;  // interior
    };
    uint16_t nPrimitives;  // 0 -> interior node
    uint8_t axis;          // interior node: xyz
};

// BVHAggregate Method Definitions
BVHAggregate::BVHAggregate(std::vector<Primitive> prims, int maxPrimsInNode,
                           SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)),
      primitives(std::move(prims)),
      splitMethod(splitMethod) {
    CHECK(!primitives.empty());
    // Build BVH from _primitives_
    // Initialize _bvhPrimitives_ array for primitives
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i].Bounds());

    // Build BVH for primitives using _bvhPrimitives_
    // Declare _Allocator_s used for BVH construction
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    std::vector<Primitive> orderedPrims(primitives.size());
    BVHBuildNode *root;
    // Build BVH according to selected _splitMethod_
    std::atomic<int> totalNodes{0};
    if (splitMethod == SplitMethod::HLBVH) {
        root = buildHLBVH(alloc, bvhPrimitives, &totalNodes, orderedPrims);
    } else {
        std::atomic<int> orderedPrimsOffset{0};
        root = buildRecursive(threadAllocators, pstd::span<BVHPrimitive>(bvhPrimitives),
                              &totalNodes, &orderedPrimsOffset, orderedPrims);
        CHECK_EQ(orderedPrimsOffset.load(), orderedPrims.size());
    }
    primitives.swap(orderedPrims);

    // Convert BVH into compact representation in _nodes_ array
    bvhPrimitives.resize(0);
    LOG_VERBOSE("BVH created with %d nodes for %d primitives (%.2f MB)",
                totalNodes.load(), (int)primitives.size(),
                float(totalNodes.load() * sizeof(LinearBVHNode)) / (1024.f * 1024.f));
    treeBytes += totalNodes * sizeof(LinearBVHNode) + sizeof(*this) +
                 primitives.size() * sizeof(primitives[0]);
    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    
    /*
    std::cout << "Primitive List:" << '\n';
    for (int i = 0; i < primitives.size(); i++) {
        std::cout << primitives[i].Bounds().ToString() << "    " << primitives[i].isLast()
                  << '\n';
    }
    std::cout << '\n' << '\n' << "Primitives in Tree:" << '\n';

    printBVH(*root);
    */
    float SAH = calculateOverallSAH(*root);
    bvhOverallSAHCost = SAH * 1000000;

    flattenBVH(root, &offset);
    CHECK_EQ(totalNodes.load(), offset);
}

LinearBVHNode* BVHAggregate::getNodes() {
    return nodes;
}

void BVHAggregate::printBVH(BVHBuildNode node) {
    if (node.nPrimitives > 0) {
        for (unsigned i = node.firstPrimOffset; i < node.firstPrimOffset + node.nPrimitives; i++) {
            std::cout << i << '\n';
        }
    } else {
        for (const auto &child : node.children) {
            if (child != NULL) {
                printBVH(*child);
            }
        }
    }
}

BVHBuildNode *BVHAggregate::buildRecursive(ThreadLocal<Allocator> &threadAllocators,
                                           pstd::span<BVHPrimitive> bvhPrimitives,
                                           std::atomic<int> *totalNodes,
                                           std::atomic<int> *orderedPrimsOffset,
                                           std::vector<Primitive> &orderedPrims) {
    DCHECK_NE(bvhPrimitives.size(), 0);
    Allocator alloc = threadAllocators.Get();
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();
    // Initialize _BVHBuildNode_ for primitive range
    ++*totalNodes;
    // Compute bounds of all primitives in BVH node
    Bounds3f bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);

    if (bounds.SurfaceArea() == 0 || bvhPrimitives.size() == 1) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            int index = bvhPrimitives[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[index];
        }
        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;

    } else {
        // Compute bound of primitive centroids and choose split dimension _dim_
        Bounds3f centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.Centroid());
        int dim = centroidBounds.MaxDimension();

        // Partition primitives into two sets and build children
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            // Create leaf _BVHBuildNode_
            int firstPrimOffset = orderedPrimsOffset->fetch_add(bvhPrimitives.size());
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                int index = bvhPrimitives[i].primitiveIndex;
                orderedPrims[firstPrimOffset + i] = primitives[index];
            }
            node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
            return node;

        } else {
            int mid = bvhPrimitives.size() / 2;
            // Partition primitives based on _splitMethod_
            switch (splitMethod) {
            case SplitMethod::Middle: {
                // Partition primitives through node's midpoint
                Float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
                auto midIter = std::partition(bvhPrimitives.begin(), bvhPrimitives.end(),
                                              [dim, pmid](const BVHPrimitive &pi) {
                                                  return pi.Centroid()[dim] < pmid;
                                              });
                mid = midIter - bvhPrimitives.begin();
                // For lots of prims with large overlapping bounding boxes, this
                // may fail to partition; in that case do not break and fall through
                // to EqualCounts.
                if (midIter != bvhPrimitives.begin() && midIter != bvhPrimitives.end())
                    break;
            }
            case SplitMethod::EqualCounts: {
                // Partition primitives into equally sized subsets
                mid = bvhPrimitives.size() / 2;
                std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                 bvhPrimitives.end(),
                                 [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                     return a.Centroid()[dim] < b.Centroid()[dim];
                                 });

                break;
            }
            case SplitMethod::SAH:
            default: {
                // Partition primitives using approximate SAH
                if (bvhPrimitives.size() <= 2) {
                    // Partition primitives into equally sized subsets
                    mid = bvhPrimitives.size() / 2;
                    std::nth_element(bvhPrimitives.begin(), bvhPrimitives.begin() + mid,
                                     bvhPrimitives.end(),
                                     [dim](const BVHPrimitive &a, const BVHPrimitive &b) {
                                         return a.Centroid()[dim] < b.Centroid()[dim];
                                     });

                } else {
                    // Allocate _BVHSplitBucket_ for SAH partition buckets
                    constexpr int nBuckets = 12;
                    BVHSplitBucket buckets[nBuckets];

                    // Initialize _BVHSplitBucket_ for SAH partition buckets
                    for (const auto &prim : bvhPrimitives) {
                        int b = nBuckets * centroidBounds.Offset(prim.Centroid())[dim];
                        if (b == nBuckets)
                            b = nBuckets - 1;
                        DCHECK_GE(b, 0);
                        DCHECK_LT(b, nBuckets);
                        buckets[b].count++;
                        buckets[b].bounds = Union(buckets[b].bounds, prim.bounds);
                    }

                    // Compute costs for splitting after each bucket
                    constexpr int nSplits = nBuckets - 1;
                    Float costs[nSplits] = {};
                    // Partially initialize _costs_ using a forward scan over splits
                    int countBelow = 0;
                    Bounds3f boundBelow;
                    for (int i = 0; i < nSplits; ++i) {
                        boundBelow = Union(boundBelow, buckets[i].bounds);
                        countBelow += buckets[i].count;
                        costs[i] += countBelow * boundBelow.SurfaceArea();
                    }

                    // Finish initializing _costs_ using a backward scan over splits
                    int countAbove = 0;
                    Bounds3f boundAbove;
                    for (int i = nSplits; i >= 1; --i) {
                        boundAbove = Union(boundAbove, buckets[i].bounds);
                        countAbove += buckets[i].count;
                        costs[i - 1] += countAbove * boundAbove.SurfaceArea();
                    }

                    // Find bucket to split at that minimizes SAH metric
                    int minCostSplitBucket = -1;
                    Float minCost = Infinity;
                    for (int i = 0; i < nSplits; ++i) {
                        // Compute cost for candidate split and update minimum if
                        // necessary
                        if (costs[i] < minCost) {
                            minCost = costs[i];
                            minCostSplitBucket = i;
                        }
                    }
                    // Compute leaf cost and SAH split cost for chosen split
                    Float leafCost = bvhPrimitives.size();
                    minCost = 1.f / 2.f + minCost / bounds.SurfaceArea();

                    // Either create leaf or split primitives at selected SAH bucket
                    if (bvhPrimitives.size() > maxPrimsInNode || minCost < leafCost) {
                        auto midIter = std::partition(
                            bvhPrimitives.begin(), bvhPrimitives.end(),
                            [=](const BVHPrimitive &bp) {
                                int b =
                                    nBuckets * centroidBounds.Offset(bp.Centroid())[dim];
                                if (b == nBuckets)
                                    b = nBuckets - 1;
                                return b <= minCostSplitBucket;
                            });
                        mid = midIter - bvhPrimitives.begin();
                        bvhOverallSAHCost += minCost * 1000000;
                    } else {
                        bvhOverallSAHCost += leafCost * 1000000;
                        // Create leaf _BVHBuildNode_
                        int firstPrimOffset =
                            orderedPrimsOffset->fetch_add(bvhPrimitives.size());
                        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                            int index = bvhPrimitives[i].primitiveIndex;
                            orderedPrims[firstPrimOffset + i] = primitives[index];
                        }
                        node->InitLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
                        return node;
                    }
                }

                break;
            }
            }

            BVHBuildNode *children[2];
            // Recursively build BVHs for _children_
            if (bvhPrimitives.size() > 128 * 1024) {
                // Recursively build child BVHs in parallel
                ParallelFor(0, 2, [&](int i) {
                    if (i == 0)
                        children[0] = buildRecursive(
                            threadAllocators, bvhPrimitives.subspan(0, mid), totalNodes,
                            orderedPrimsOffset, orderedPrims);
                    else
                        children[1] =
                            buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                           totalNodes, orderedPrimsOffset, orderedPrims);
                });

            } else {
                // Recursively build child BVHs sequentially
                children[0] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(0, mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
                children[1] =
                    buildRecursive(threadAllocators, bvhPrimitives.subspan(mid),
                                   totalNodes, orderedPrimsOffset, orderedPrims);
            }

            node->InitInterior(dim, children[0], children[1]);
        }
    }

    return node;
}

BVHBuildNode *BVHAggregate::buildHLBVH(Allocator alloc,
                                       const std::vector<BVHPrimitive> &bvhPrimitives,
                                       std::atomic<int> *totalNodes,
                                       std::vector<Primitive> &orderedPrims) {
    // Compute bounding box of all primitive centroids
    Bounds3f bounds;
    for (const BVHPrimitive &prim : bvhPrimitives)
        bounds = Union(bounds, prim.Centroid());

    // Compute Morton indices of primitives
    std::vector<MortonPrimitive> mortonPrims(bvhPrimitives.size());
    ParallelFor(0, bvhPrimitives.size(), [&](int64_t i) {
        // Initialize _mortonPrims[i]_ for _i_th primitive
        constexpr int mortonBits = 10;
        constexpr int mortonScale = 1 << mortonBits;
        mortonPrims[i].primitiveIndex = bvhPrimitives[i].primitiveIndex;
        Vector3f centroidOffset = bounds.Offset(bvhPrimitives[i].Centroid());
        Vector3f offset = centroidOffset * mortonScale;
        mortonPrims[i].mortonCode = EncodeMorton3(offset.x, offset.y, offset.z);
    });

    // Radix sort primitive Morton indices
    RadixSort(&mortonPrims);

    // Create LBVH treelets at bottom of BVH
    // Find intervals of primitives for each treelet
    std::vector<LBVHTreelet> treeletsToBuild;
    for (size_t start = 0, end = 1; end <= mortonPrims.size(); ++end) {
        uint32_t mask = 0b00111111111111000000000000000000;
        if (end == (int)mortonPrims.size() || ((mortonPrims[start].mortonCode & mask) !=
                                               (mortonPrims[end].mortonCode & mask))) {
            // Add entry to _treeletsToBuild_ for this treelet
            size_t nPrimitives = end - start;
            int maxBVHNodes = 2 * nPrimitives - 1;
            BVHBuildNode *nodes = alloc.allocate_object<BVHBuildNode>(maxBVHNodes);
            treeletsToBuild.push_back({start, nPrimitives, nodes});

            start = end;
        }
    }

    // Create LBVHs for treelets in parallel
    std::atomic<int> orderedPrimsOffset(0);
    ParallelFor(0, treeletsToBuild.size(), [&](int i) {
        // Generate _i_th LBVH treelet
        int nodesCreated = 0;
        const int firstBitIndex = 29 - 12;
        LBVHTreelet &tr = treeletsToBuild[i];
        tr.buildNodes = emitLBVH(
            tr.buildNodes, bvhPrimitives, &mortonPrims[tr.startIndex], tr.nPrimitives,
            &nodesCreated, orderedPrims, &orderedPrimsOffset, firstBitIndex);
        *totalNodes += nodesCreated;
    });

    // Create and return SAH BVH from LBVH treelets
    std::vector<BVHBuildNode *> finishedTreelets;
    finishedTreelets.reserve(treeletsToBuild.size());
    for (LBVHTreelet &treelet : treeletsToBuild)
        finishedTreelets.push_back(treelet.buildNodes);
    return buildUpperSAH(alloc, finishedTreelets, 0, finishedTreelets.size(), totalNodes);
}

BVHBuildNode *BVHAggregate::emitLBVH(BVHBuildNode *&buildNodes,
                                     const std::vector<BVHPrimitive> &bvhPrimitives,
                                     MortonPrimitive *mortonPrims, int nPrimitives,
                                     int *totalNodes,
                                     std::vector<Primitive> &orderedPrims,
                                     std::atomic<int> *orderedPrimsOffset, int bitIndex) {
    CHECK_GT(nPrimitives, 0);
    if (bitIndex == -1 || nPrimitives < maxPrimsInNode) {
        // Create and return leaf node of LBVH treelet
        ++*totalNodes;
        BVHBuildNode *node = buildNodes++;
        Bounds3f bounds;
        int firstPrimOffset = orderedPrimsOffset->fetch_add(nPrimitives);
        for (int i = 0; i < nPrimitives; ++i) {
            int primitiveIndex = mortonPrims[i].primitiveIndex;
            orderedPrims[firstPrimOffset + i] = primitives[primitiveIndex];
            bounds = Union(bounds, bvhPrimitives[primitiveIndex].bounds);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;

    } else {
        int mask = 1 << bitIndex;
        // Advance to next subtree level if there is no LBVH split for this bit
        if ((mortonPrims[0].mortonCode & mask) ==
            (mortonPrims[nPrimitives - 1].mortonCode & mask))
            return emitLBVH(buildNodes, bvhPrimitives, mortonPrims, nPrimitives,
                            totalNodes, orderedPrims, orderedPrimsOffset, bitIndex - 1);

        // Find LBVH split point for this dimension
        int splitOffset = FindInterval(nPrimitives, [&](int index) {
            return ((mortonPrims[0].mortonCode & mask) ==
                    (mortonPrims[index].mortonCode & mask));
        });
        ++splitOffset;
        CHECK_LE(splitOffset, nPrimitives - 1);
        CHECK_NE(mortonPrims[splitOffset - 1].mortonCode & mask,
                 mortonPrims[splitOffset].mortonCode & mask);

        // Create and return interior LBVH node
        (*totalNodes)++;
        BVHBuildNode *node = buildNodes++;
        BVHBuildNode *lbvh[2] = {
            emitLBVH(buildNodes, bvhPrimitives, mortonPrims, splitOffset, totalNodes,
                     orderedPrims, orderedPrimsOffset, bitIndex - 1),
            emitLBVH(buildNodes, bvhPrimitives, &mortonPrims[splitOffset],
                     nPrimitives - splitOffset, totalNodes, orderedPrims,
                     orderedPrimsOffset, bitIndex - 1)};
        int axis = bitIndex % 3;
        node->InitInterior(axis, lbvh[0], lbvh[1]);
        return node;
    }
}

int BVHAggregate::flattenBVH(BVHBuildNode *node, int *offset) {
    LinearBVHNode *linearNode = &nodes[*offset];
    linearNode->bounds = node->bounds;
    int nodeOffset = (*offset)++;
    if (node->nPrimitives > 0) {
        CHECK(!node->children[0] && !node->children[1]);
        CHECK_LT(node->nPrimitives, 65536);
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->nPrimitives = node->nPrimitives;
    } else {
        // Create interior flattened BVH node
        linearNode->axis = node->splitAxis;
        linearNode->nPrimitives = 0;
        flattenBVH(node->children[0], offset);
        linearNode->secondChildOffset = flattenBVH(node->children[1], offset);
    }
    return nodeOffset;
}

float BVHAggregate::calculateOverallSAH(BVHBuildNode node) {
    float SAH = 0;
    if (node.nPrimitives) {
        SAH = node.nPrimitives;
    } else {
        float S = node.bounds.SurfaceArea();
        SAH = TRAVERSAL_COST_BVH_NODE +
              (node.children[0]->bounds.SurfaceArea() / S) * calculateOverallSAH(*node.children[0]) +
              (node.children[1]->bounds.SurfaceArea() / S) * calculateOverallSAH(*node.children[1]);
    }
    return SAH;
}

Bounds3f BVHAggregate::Bounds() const {
    CHECK(nodes);
    return nodes[0].bounds;
}

pstd::optional<ShapeIntersection> BVHAggregate::Intersect(const Ray &ray,
                                                          Float tMax) const {
    if (!nodes)
        return {};
    pstd::optional<ShapeIntersection> si;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {int(invDir.x < 0), int(invDir.y < 0), int(invDir.z < 0)};
    if (Bounds().IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
        bvhRays1++;
        bvhtRays++;
    }
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    int nodesVisited = 0;
    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        // Check ray against BVH node
        bvhPlaneIntersections += 6;
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < node->nPrimitives; ++i) {
                    // Check for intersection with primitive in BVH node
                    pstd::optional<ShapeIntersection> primSi =
                        primitives[node->primitivesOffset + i].Intersect(ray, tMax);
                    bvhTriangleIntersections++;
                    if (primSi) {
                        si = primSi;
                        tMax = si->tHit;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];

            } else {
                // Put far BVH node on _nodesToVisit_ stack, advance to near node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    bvhNodesVisited += nodesVisited;
    return si;
}

bool BVHAggregate::IntersectP(const Ray &ray, Float tMax) const {
    if (!nodes)
        return false;
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0),
                       static_cast<int>(invDir.z < 0)};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesVisited = 0;

    while (true) {
        ++nodesVisited;
        const LinearBVHNode *node = &nodes[currentNodeIndex];
        if (node->bounds.IntersectP(ray.o, ray.d, tMax, invDir, dirIsNeg)) {
            // Process BVH node _node_ for traversal
            if (node->nPrimitives > 0) {
                for (int i = 0; i < node->nPrimitives; ++i) {
                    if (primitives[node->primitivesOffset + i].IntersectP(ray, tMax)) {
                        bvhNodesVisited += nodesVisited;
                        return true;
                    }
                }
                if (toVisitOffset == 0)
                    break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node->axis] != 0) {
                    /// second child first
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0)
                break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    bvhNodesVisited += nodesVisited;
    return false;
}

BVHBuildNode *BVHAggregate::buildUpperSAH(Allocator alloc,
                                          std::vector<BVHBuildNode *> &treeletRoots,
                                          int start, int end,
                                          std::atomic<int> *totalNodes) const {
    CHECK_LT(start, end);
    int nNodes = end - start;
    if (nNodes == 1)
        return treeletRoots[start];
    (*totalNodes)++;
    BVHBuildNode *node = alloc.new_object<BVHBuildNode>();

    // Compute bounds of all nodes under this HLBVH node
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
        bounds = Union(bounds, treeletRoots[i]->bounds);

    // Compute bound of HLBVH node centroids, choose split dimension _dim_
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        Point3f centroid =
            (treeletRoots[i]->bounds.pMin + treeletRoots[i]->bounds.pMax) * 0.5f;
        centroidBounds = Union(centroidBounds, centroid);
    }
    int dim = centroidBounds.MaxDimension();
    // FIXME: if this hits, what do we need to do?
    // Make sure the SAH split below does something... ?
    CHECK_NE(centroidBounds.pMax[dim], centroidBounds.pMin[dim]);

    // Allocate _BVHSplitBucket_ for SAH partition buckets
    constexpr int nBuckets = 12;
    struct BVHSplitBucket {
        int count = 0;
        Bounds3f bounds;
    };
    BVHSplitBucket buckets[nBuckets];

    // Initialize _BVHSplitBucket_ for HLBVH SAH partition buckets
    for (int i = start; i < end; ++i) {
        Float centroid =
            (treeletRoots[i]->bounds.pMin[dim] + treeletRoots[i]->bounds.pMax[dim]) *
            0.5f;
        int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                            (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
        if (b == nBuckets)
            b = nBuckets - 1;
        CHECK_GE(b, 0);
        CHECK_LT(b, nBuckets);
        buckets[b].count++;
        buckets[b].bounds = Union(buckets[b].bounds, treeletRoots[i]->bounds);
    }

    // Compute costs for splitting after each bucket
    Float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
        Bounds3f b0, b1;
        int count0 = 0, count1 = 0;
        for (int j = 0; j <= i; ++j) {
            b0 = Union(b0, buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for (int j = i + 1; j < nBuckets; ++j) {
            b1 = Union(b1, buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) /
                              bounds.SurfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    Float minCost = cost[0];
    int minCostSplitBucket = 0;
    for (int i = 1; i < nBuckets - 1; ++i) {
        if (cost[i] < minCost) {
            minCost = cost[i];
            minCostSplitBucket = i;
        }
    }

    // Split nodes and create interior HLBVH SAH node
    BVHBuildNode **pmid = std::partition(
        &treeletRoots[start], &treeletRoots[end - 1] + 1, [=](const BVHBuildNode *node) {
            Float centroid = (node->bounds.pMin[dim] + node->bounds.pMax[dim]) * 0.5f;
            int b = nBuckets * ((centroid - centroidBounds.pMin[dim]) /
                                (centroidBounds.pMax[dim] - centroidBounds.pMin[dim]));
            if (b == nBuckets)
                b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            return b <= minCostSplitBucket;
        });
    int mid = pmid - &treeletRoots[0];
    CHECK_GT(mid, start);
    CHECK_LT(mid, end);
    node->InitInterior(dim,
                       this->buildUpperSAH(alloc, treeletRoots, start, mid, totalNodes),
                       this->buildUpperSAH(alloc, treeletRoots, mid, end, totalNodes));
    return node;
}

BVHAggregate *BVHAggregate::Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters) {
    std::string splitMethodName = parameters.GetOneString("splitmethod", "sah");
    BVHAggregate::SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = BVHAggregate::SplitMethod::SAH;
    else if (splitMethodName == "hlbvh")
        splitMethod = BVHAggregate::SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = BVHAggregate::SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = BVHAggregate::SplitMethod::EqualCounts;
    else {
        Warning(R"(BVH split method "%s" unknown.  Using "sah".)", splitMethodName);
        splitMethod = BVHAggregate::SplitMethod::SAH;
    }

    int maxPrimsInNode = parameters.GetOneInt("maxnodeprims", 4);
    return new BVHAggregate(std::move(prims), maxPrimsInNode, splitMethod);
}

// KdNodeToVisit Definition
struct KdNodeToVisit {
    const KdTreeNode *node;
    Float tMin, tMax;
};

// KdTreeNode Definition
struct alignas(8) KdTreeNode {
    // KdTreeNode Methods
    void InitLeaf(pstd::span<const int> primNums, std::vector<int> *primitiveIndices);

    void InitInterior(int axis, int aboveChild, Float s) {
        split = s;
        flags = axis | (aboveChild << 2);
    }

    Float SplitPos() const { return split; }
    int nPrimitives() const { return flags >> 2; }
    int SplitAxis() const { return flags & 3; }
    bool IsLeaf() const { return (flags & 3) == 3; }
    int AboveChild() const { return flags >> 2; }

    union {
        Float split;                 // Interior
        int onePrimitiveIndex;       // Leaf
        int primitiveIndicesOffset;  // Leaf
    };

  private:
    uint32_t flags;
};

// EdgeType Definition
enum class EdgeType { Start, End };

// BoundEdge Definition
struct BoundEdge {
    // BoundEdge Public Methods
    BoundEdge() {}

    BoundEdge(Float t, int primNum, bool starting) : t(t), primNum(primNum) {
        type = starting ? EdgeType::Start : EdgeType::End;
    }

    Float t;
    int primNum;
    EdgeType type;
};

STAT_PIXEL_COUNTER("Kd-Tree/Nodes visited", kdNodesVisited);

// KdTreeAggregate Method Definitions
KdTreeAggregate::KdTreeAggregate(std::vector<Primitive> p, int isectCost,
                                 int traversalCost, Float emptyBonus, int maxPrims,
                                 int maxDepth)
    : isectCost(isectCost),
      traversalCost(traversalCost),
      maxPrims(maxPrims),
      emptyBonus(emptyBonus),
      primitives(std::move(p)) {
    // Build kd-tree aggregate
    nextFreeNode = nAllocedNodes = 0;
    if (maxDepth <= 0)
        maxDepth = std::round(8 + 1.3f * Log2Int(int64_t(primitives.size())));
    // Compute bounds for kd-tree construction
    std::vector<Bounds3f> primBounds;
    primBounds.reserve(primitives.size());
    for (Primitive &prim : primitives) {
        Bounds3f b = prim.Bounds();
        bounds = Union(bounds, b);
        primBounds.push_back(b);
    }

    // Allocate working memory for kd-tree construction
    std::vector<BoundEdge> edges[3];
    for (int i = 0; i < 3; ++i)
        edges[i].resize(2 * primitives.size());

    std::vector<int> prims0(primitives.size());
    std::vector<int> prims1((maxDepth + 1) * primitives.size());

    // Initialize _primNums_ for kd-tree construction
    std::vector<int> primNums(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        primNums[i] = i;

    // Start recursive construction of kd-tree
    buildTree(0, bounds, primBounds, primNums, maxDepth, edges, pstd::span<int>(prims0),
              pstd::span<int>(prims1), 0);
}

void KdTreeNode::InitLeaf(pstd::span<const int> primNums,
                          std::vector<int> *primitiveIndices) {
    flags = 3 | (primNums.size() << 2);
    // Store primitive ids for leaf node
    if (primNums.size() == 0)
        onePrimitiveIndex = 0;
    else if (primNums.size() == 1)
        onePrimitiveIndex = primNums[0];
    else {
        primitiveIndicesOffset = primitiveIndices->size();
        for (int pn : primNums)
            primitiveIndices->push_back(pn);
    }
}

void KdTreeAggregate::buildTree(int nodeNum, const Bounds3f &nodeBounds,
                                const std::vector<Bounds3f> &allPrimBounds,
                                pstd::span<const int> primNums, int depth,
                                std::vector<BoundEdge> edges[3], pstd::span<int> prims0,
                                pstd::span<int> prims1, int badRefines) {
    CHECK_EQ(nodeNum, nextFreeNode);
    // Get next free node from _nodes_ array
    if (nextFreeNode == nAllocedNodes) {
        int nNewAllocNodes = std::max(2 * nAllocedNodes, 512);
        KdTreeNode *n = new KdTreeNode[nNewAllocNodes];
        if (nAllocedNodes > 0) {
            std::memcpy(n, nodes, nAllocedNodes * sizeof(KdTreeNode));
            delete[] nodes;
        }
        nodes = n;
        nAllocedNodes = nNewAllocNodes;
    }
    ++nextFreeNode;

    // Initialize leaf node if termination criteria met
    if (primNums.size() <= maxPrims || depth == 0) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Initialize interior node and continue recursion
    // Choose split axis position for interior node
    int bestAxis = -1, bestOffset = -1;
    Float bestCost = Infinity, leafCost = isectCost * primNums.size();
    Float invTotalSA = 1 / nodeBounds.SurfaceArea();
    // Choose which axis to split along
    int axis = nodeBounds.MaxDimension();

    // Choose split along axis and attempt to partition primitives
    int retries = 0;
    size_t nPrimitives = primNums.size();
retrySplit:
    // Initialize edges for _axis_
    for (size_t i = 0; i < nPrimitives; ++i) {
        int pn = primNums[i];
        const Bounds3f &bounds = allPrimBounds[pn];
        edges[axis][2 * i] = BoundEdge(bounds.pMin[axis], pn, true);
        edges[axis][2 * i + 1] = BoundEdge(bounds.pMax[axis], pn, false);
    }
    // Sort _edges_ for _axis_
    std::sort(edges[axis].begin(), edges[axis].begin() + 2 * nPrimitives,
              [](const BoundEdge &e0, const BoundEdge &e1) -> bool {
                  return std::tie(e0.t, e0.type) < std::tie(e1.t, e1.type);
              });

    // Compute cost of all splits for _axis_ to find best
    int nBelow = 0, nAbove = primNums.size();
    for (size_t i = 0; i < 2 * primNums.size(); ++i) {
        if (edges[axis][i].type == EdgeType::End)
            --nAbove;
        Float edgeT = edges[axis][i].t;
        if (edgeT > nodeBounds.pMin[axis] && edgeT < nodeBounds.pMax[axis]) {
            // Compute child surface areas for split at _edgeT_
            Vector3f d = nodeBounds.pMax - nodeBounds.pMin;
            int otherAxis0 = (axis + 1) % 3, otherAxis1 = (axis + 2) % 3;
            Float belowSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (edgeT - nodeBounds.pMin[axis]) * (d[otherAxis0] + d[otherAxis1]));
            Float aboveSA =
                2 * (d[otherAxis0] * d[otherAxis1] +
                     (nodeBounds.pMax[axis] - edgeT) * (d[otherAxis0] + d[otherAxis1]));

            // Compute cost for split at _i_th edge
            Float pBelow = belowSA * invTotalSA, pAbove = aboveSA * invTotalSA;
            Float eb = (nAbove == 0 || nBelow == 0) ? emptyBonus : 0;
            Float cost = traversalCost +
                         isectCost * (1 - eb) * (pBelow * nBelow + pAbove * nAbove);
            // Update best split if this is lowest cost so far
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestOffset = i;
            }
        }
        if (edges[axis][i].type == EdgeType::Start)
            ++nBelow;
    }
    CHECK(nBelow == nPrimitives && nAbove == 0);

    // Try to split along another axis if no good splits were found
    if (bestAxis == -1 && retries < 2) {
        ++retries;
        axis = (axis + 1) % 3;
        goto retrySplit;
    }

    // Create leaf if no good splits were found
    if (bestCost > leafCost)
        ++badRefines;
    if ((bestCost > 4 * leafCost && nPrimitives < 16) || bestAxis == -1 ||
        badRefines == 3) {
        nodes[nodeNum].InitLeaf(primNums, &primitiveIndices);
        return;
    }

    // Classify primitives with respect to split
    int n0 = 0, n1 = 0;
    for (int i = 0; i < bestOffset; ++i)
        if (edges[bestAxis][i].type == EdgeType::Start)
            prims0[n0++] = edges[bestAxis][i].primNum;
    for (int i = bestOffset + 1; i < 2 * nPrimitives; ++i)
        if (edges[bestAxis][i].type == EdgeType::End)
            prims1[n1++] = edges[bestAxis][i].primNum;

    // Recursively initialize kd-tree node's children
    Float tSplit = edges[bestAxis][bestOffset].t;
    Bounds3f bounds0 = nodeBounds, bounds1 = nodeBounds;
    bounds0.pMax[bestAxis] = bounds1.pMin[bestAxis] = tSplit;
    buildTree(nodeNum + 1, bounds0, allPrimBounds, prims0.subspan(0, n0), depth - 1,
              edges, prims0, prims1.subspan(n1), badRefines);
    int aboveChild = nextFreeNode;
    nodes[nodeNum].InitInterior(bestAxis, aboveChild, tSplit);
    buildTree(aboveChild, bounds1, allPrimBounds, prims1.subspan(0, n1), depth - 1, edges,
              prims0, prims1.subspan(n1), badRefines);
}

pstd::optional<ShapeIntersection> KdTreeAggregate::Intersect(const Ray &ray,
                                                             Float rayTMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, rayTMax, &tMin, &tMax))
        return {};

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxToVisit = 64;
    KdNodeToVisit toVisit[maxToVisit];
    int toVisitIndex = 0;
    int nodesVisited = 0;

    // Traverse kd-tree nodes in order for ray
    pstd::optional<ShapeIntersection> si;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        // Bail out if we found a hit closer than the current node
        if (rayTMax < tMin)
            break;

        ++nodesVisited;
        if (!node->IsLeaf()) {
            // Visit kd-tree interior node
            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get node child pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next child node, possibly enqueue other child
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;

                node = firstChild;
                tMax = tSplit;
            }

        } else {
            // Check for intersections inside leaf node
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                // Check one primitive inside leaf node
                pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                if (primSi) {
                    si = primSi;
                    rayTMax = si->tHit;
                }

            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int index = primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &p = primitives[index];
                    // Check one primitive inside leaf node
                    pstd::optional<ShapeIntersection> primSi = p.Intersect(ray, rayTMax);
                    if (primSi) {
                        si = primSi;
                        rayTMax = si->tHit;
                    }
                }
            }

            // Grab next node to visit from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        }
    }
    kdNodesVisited += nodesVisited;
    return si;
}

bool KdTreeAggregate::IntersectP(const Ray &ray, Float raytMax) const {
    // Compute initial parametric range of ray inside kd-tree extent
    Float tMin, tMax;
    if (!bounds.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
        return false;

    // Prepare to traverse kd-tree for ray
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    constexpr int maxTodo = 64;
    KdNodeToVisit toVisit[maxTodo];
    int toVisitIndex = 0;
    int nodesVisited = 0;
    const KdTreeNode *node = &nodes[0];
    while (node) {
        ++nodesVisited;
        if (node->IsLeaf()) {
            // Check for shadow ray intersections inside leaf node
            int nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                const Primitive &p = primitives[node->onePrimitiveIndex];
                if (p.IntersectP(ray, raytMax)) {
                    kdNodesVisited += nodesVisited;
                    return true;
                }
            } else {
                for (int i = 0; i < nPrimitives; ++i) {
                    int primitiveIndex =
                        primitiveIndices[node->primitiveIndicesOffset + i];
                    const Primitive &prim = primitives[primitiveIndex];
                    if (prim.IntersectP(ray, raytMax)) {
                        kdNodesVisited += nodesVisited;
                        return true;
                    }
                }
            }

            // Grab next node to process from todo list
            if (toVisitIndex > 0) {
                --toVisitIndex;
                node = toVisit[toVisitIndex].node;
                tMin = toVisit[toVisitIndex].tMin;
                tMax = toVisit[toVisitIndex].tMax;
            } else
                break;
        } else {
            // Process kd-tree interior node

            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            Float tSplit = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get node children pointers for ray
            const KdTreeNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] < node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst != 0) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            } else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next child node, possibly enqueue other child
            if (tSplit > tMax || tSplit <= 0)
                node = firstChild;
            else if (tSplit < tMin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                toVisit[toVisitIndex].node = secondChild;
                toVisit[toVisitIndex].tMin = tSplit;
                toVisit[toVisitIndex].tMax = tMax;
                ++toVisitIndex;
                node = firstChild;
                tMax = tSplit;
            }
        }
    }
    kdNodesVisited += nodesVisited;
    return false;
}

KdTreeAggregate *KdTreeAggregate::Create(std::vector<Primitive> prims,
                                         const ParameterDictionary &parameters) {
    int isectCost = parameters.GetOneInt("intersectcost", 5);
    int travCost = parameters.GetOneInt("traversalcost", 1);
    Float emptyBonus = parameters.GetOneFloat("emptybonus", 0.5f);
    int maxPrims = parameters.GetOneInt("maxprims", 1);
    int maxDepth = parameters.GetOneInt("maxdepth", -1);
    return new KdTreeAggregate(std::move(prims), isectCost, travCost, emptyBonus,
                               maxPrims, maxDepth);
}

Primitive CreateAccelerator(const std::string &name, std::vector<Primitive> prims,
                            const ParameterDictionary &parameters) {
    Primitive accel = nullptr;
    if (name == "bvh") {
        testForDst += 1;
        //std::cout << "Bounding Volume Hierarchy" << '\n';
        accel = BVHAggregate::Create(std::move(prims), parameters);
    } else if (name == "kdtree")
        accel = KdTreeAggregate::Create(std::move(prims), parameters);
    else if (name == "dst") {
        accel = DSTAggregate::Create(std::move(prims), parameters);
        testForDst += 10;
        //std::cout << "Dual-Split Tree" << '\n';
    } else if (name == "wdst") {
        testForDst += 100;
        //std::cout << "Wide Dual-Split Tree" << '\n';
        accel = WDSTAggregate::Create(std::move(prims), parameters);
    } else 
        ErrorExit("%s: accelerator type unknown.", name);

    if (!accel)
        ErrorExit("%s: unable to create accelerator.", name);

    parameters.ReportUnused();
    return accel;
}

//DST Node Deffinition
struct DSTBuildNode {
    void InitLeaf(int triangleOffset, int depthLevel, int nTriangles, Bounds3f BB) { 
        int header = 0b000001;
        flags = triangleOffset | (header << 26);
        this->depthLevel = depthLevel;
        this->nTriangles = nTriangles;
        BoundingBox = BB;
        this->triangleOffset = triangleOffset;
    }

    void InitSingleAxisCarving(int planeAxis, DSTBuildNode *child, bool leaf,
                               int triangleOffset, int depthLevel, float plane1,
                               float plane2, Bounds3f BB, int nTriangles) {
        singleAxisCNs++;
        int header = 0b110000;
        header |= leaf;
        header |= planeAxis << 1;
        if (!leaf) {
            children[0] = child;
            triangleOffset = 0;
        }
        flags = header << 26 | triangleOffset;
        this->depthLevel = depthLevel;
        this->plane1 = plane2;
        this->plane2 = plane1;
        this->triangleOffset = triangleOffset;
        BoundingBox = BB;
        this->nTriangles = nTriangles;
    }

    void InitDoubleAxisCarving(int planeAxesAndCornerType, DSTBuildNode *child, bool leaf,
                               int triangleOffset, int depthLevel, float plane1,
                               float plane2, Bounds3f BB, int nTriangles) {
        dualAxisCNs++;
        int header = 0b100000;
        header |= leaf;
        header |= planeAxesAndCornerType << 1;
        if (!leaf) {
            children[0] = child;
            triangleOffset = 0;
        }
        flags = header << 26 | triangleOffset;
        this->depthLevel = depthLevel;
        this->plane1 = plane1;
        this->plane2 = plane2;
        this->triangleOffset = triangleOffset;
        BoundingBox = BB;
        this->nTriangles = nTriangles;
    }

    void InitSplittingNode(int planeAxis, DSTBuildNode* firstChild, DSTBuildNode* secondChild, int depthLevel, float plane1, float plane2, Bounds3f BB) {
        int header = 0b000000;
        header |= planeAxis << 3;
        flags = header << 26;
        children[0] = firstChild;
        children[1] = secondChild;
        this->depthLevel = depthLevel;
        this->plane1 = plane1;
        this->plane2 = plane2;
        BoundingBox = BB;
    }

    void SetOffset(int offset) { this->offset = offset; }


    bool IsLeaf() const { return (flags & IS_LEAF && !IsCarving()); }
    bool IsCarving() const { return (flags & IS_CARVING); }
    bool IsSplitting() const { return !IsLeaf() && !IsCarving(); }
    int Offset() const { return offset; }
    int GetFlag() const { return flags; }
    int Plane1AsInt() { return *reinterpret_cast<int*>(&plane1); }
    int Plane2AsInt() { return *reinterpret_cast<int*>(&plane2); }
    int GetDepthLevel() const { return depthLevel; }
    int GetHeader() const { return flags >> 26; }
    int GetSplittingPlanesAxis() const { return (flags & SPLITTING_PLANE_AXIS) >> 29; }
    std::vector<int> GetCarvedSides() const { 
        if (flags >> 29 == 0b110) {
            int axis = (flags & CARVING_PLANE_AXIS) >> 27;
            std::vector<int> carvedSides;
            if (plane2 != 0.f) {
                carvedSides.push_back(axis + 1);
            }
            if (plane1 != 0.f) {
                carvedSides.push_back(axis + 4);
            }
            return carvedSides;
        } else {
            int axes = ((flags & CARVING_PLANE_AXES) >> 29) + 1;
            int cornerType = (flags & CORNER_TYPE) >> 27;
            if (axes == 1) {
                return std::vector<int>{1 + ((cornerType & 0b01) * 3),
                                        2 + ((cornerType & 0b10) * 3)};
            } else if (axes == 2) {
                return std::vector<int>{1 + ((cornerType & 0b01) * 3),
                                        3 + ((cornerType & 0b10) * 3)};
            } else {
                return std::vector<int>{2 + ((cornerType & 0b01) * 3),
                                        3 + ((cornerType & 0b10) * 3)};
            }
        }
    }
    int offsetToFirstChild() const {
        if (children[0] == NULL)
            return triangleOffset;
        else return children[0]->Offset(); }
    Bounds3f GetBB() const {
        return BoundingBox; }
    int NTriangles() const { return nTriangles; }
    bool isCarvingLeaf() const { return IsCarving() && (flags & IS_LEAF); }

    DSTBuildNode* children[2] = {NULL, NULL};
    float plane1 = NULL;
    float plane2 = NULL;

    private:
        int triangleOffset = 0;
        int offset = NULL;
        int depthLevel = NULL;
        uint32_t flags;
        Bounds3f BoundingBox;
        int nTriangles = 0;
};

std::vector<DSTBuildNode> gloabalDSTNodes;

DSTAggregate::DSTAggregate(std::vector<Primitive> prims, LinearBVHNode *BVHNodes)
    : primitives(std::move(prims)) {
    CHECK(!primitives.empty());

    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });

    DSTBuildNode *root = alloc.new_object<DSTBuildNode>();

    globalBB = BVHNodes[0].bounds;
    root = BuildRecursiveFromBVH(threadAllocators, BVHNodes, 0, 0);
    numberOfDSTNodes = 0;
    
    nodesPerDepthLevel.resize(maximumDepth + 1);
    addNodeToDepthList(root);
    int offset = 0;
    for (int i = 0; i < nodesPerDepthLevel.size(); i++) {
        while (nodesPerDepthLevel[i].size() != 0) {
            nodesPerDepthLevel[i].front()->SetOffset(offset);
            if (nodesPerDepthLevel[i].front()->IsLeaf()) {
                offset++;
            } else {
                offset = offset + 3;
            }
            dstNumberOfNodes++;
            numberOfDSTNodes++;
            nodesPerDepthLevel[i].pop_front();
        }
    }

    /* std::cout << "Primitive List:" << '\n';
    for (int i = 0; i < primitives.size(); i++) {
        std::cout << primitives[i].Bounds().ToString() << "    " << primitives[i].isLast() << '\n';
    }
    std::cout << '\n' << '\n' << "Primitives in Tree:" << '\n';
    */
    float SAH = this->calculateOverallSAH(*root, globalBB, 0);
    dstOverallSAHCost = SAH * 1000000;

    dstBytes = offset * 4;
    linearDST.resize(offset);
    FlattenDST(root);

    buildNodes.resize(numberOfDSTNodes);
    addBuildNode(*root, 0);
    gloabalDSTNodes = buildNodes;
    /*
    std::cout << '\n' << "Primitive in liear Tree:" << '\n';
    int idx = 0;
    while (idx < linearDST.size()) {
        int entry = linearDST[idx];
        if (entry & IS_LEAF) {
            std::cout << (entry & OFFSET) << '\n';
            if (entry & IS_CARVING)
                idx += 3;
            else
                idx++;
        } else
            idx += 3;
    }*/
}

void DSTAggregate::FlattenDST(DSTBuildNode *node) {
    if (node == NULL)
        return;
    uint32_t offset = node->Offset();
    if (node->IsLeaf()) {
        linearDST[offset] = node->GetFlag();
        dstNumberOfLeafs++;
    } else {
        uint32_t flag = node->GetFlag();
        flag |= node->offsetToFirstChild();
        if (node->IsSplitting()) {
            int leftChildSize = (!node->children[0]->IsLeaf()) * 2 + 1;
            flag |= leftChildSize << 27;
            dstNumberOfSplittingNodes++;
        } else {
            if (node->isCarvingLeaf())
                dstNumberOfLeafs++;
            else 
                dstNumberOfCarvingNodes++;
        }
        linearDST[offset] = flag;
        linearDST[offset + 1] = node->Plane1AsInt();
        linearDST[offset + 2] = node->Plane2AsInt();
        if (node->children[0] != NULL)
            FlattenDST(node->children[0]);
        if (node->children[1] != NULL)
            FlattenDST(node->children[1]);
    }
}

int DSTAggregate::addBuildNode(DSTBuildNode node, int index) {
    buildNodes[index] = node;
    if (node.IsSplitting()) {
        index = addBuildNode(*node.children[0], index + 1);
        index = addBuildNode(*node.children[1], index + 1);
    } else if (node.IsCarving() && !node.isCarvingLeaf()) {
        index = addBuildNode(*node.children[0], index + 1);
    }
    return index;
}

DSTAggregate *DSTAggregate::Create(std::vector<Primitive> prims,
                                   const ParameterDictionary &parameters) {
    //Takes the parameters string and parses it into the necessary parameters for the constructor
    BVHAggregate bvh = *BVHAggregate::Create(prims, parameters);
    return new DSTAggregate(std::move(bvh.primitives), bvh.getNodes());
}

Bounds3f DSTAggregate::Bounds() const {
    return globalBB;
}

pstd::optional<ShapeIntersection> DSTAggregate::Intersect(const Ray &ray,
                                                          Float globalTMax) const {
    float delta = 0.001;
    pstd::optional<ShapeIntersection> si;
    // For code documentation look at DST supplemental materials Listing 7. DST Traversal Kernel
    float tMin = 0;
    float tMax = globalTMax;

    unsigned headerOffset;
    unsigned header5;
    bool leaf;
    unsigned idx = 0;

    const Vector3f invDir = {1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z};
    const int dirSgn[3] = {invDir[0] < 0, invDir[1] < 0, invDir[2] < 0};
    if (!globalBB.IntersectP(ray.o, ray.d, tMax, invDir, dirSgn))
        return {};
    dstRays++;
    dstRays1++;
    if (tMin < 0)
        tMin = 0;

    StackItem* stack = new StackItem[maximumDepth + 1];
    int stackPtr = -1;

    while (true) {
        headerOffset = linearDST[idx];
        header5 = headerOffset >> 27;
        leaf = (headerOffset >> 26) & 1;

        if (header5 >> 4 == 0) {
            if (leaf) {
                idx = headerOffset & OFFSET;
                goto leaf;
            }
            float splitPlanes[2];
            uint32_t temp1 = linearDST[idx + 1];
            splitPlanes[0] = *reinterpret_cast<float*>(&temp1);
            uint32_t temp2 = linearDST[idx + 2];
            splitPlanes[1] = *reinterpret_cast<float*>(&temp2);
            idx = headerOffset & OFFSET;

            unsigned axis = header5 >> 2;
            unsigned sign = dirSgn[axis];
            unsigned diff = header5 & 3; 
            float ts1 = (splitPlanes[sign] - ray.o[axis]) * invDir[axis];
            float ts2 = (splitPlanes[sign ^ 1] - ray.o[axis]) * invDir[axis];

            sign *= diff;
            dstPlaneIntersections += 2;
            if (tMax + delta >= ts2) {
                float tNext = std::max(tMin, ts2);
                if (tMin <= ts1 + delta) {
                    stack[++stackPtr] = StackItem(idx + (sign ^ diff), tNext, tMax);
                    idx += sign;
                    tMax = std::min(tMax, ts1);
                } else {
                    idx += sign ^ diff;
                    tMin = tNext;
                }
                continue;
            } else {
                if (tMin <= ts1 + delta) {
                    idx += sign;
                    tMax = std::min(tMax, ts1);
                    continue;
                } else {
                    goto pop;
                }
            }
        } else {
            float carvePlanes[2];
            uint32_t temp1 = linearDST[idx + 1];
            carvePlanes[0] = *reinterpret_cast<float*>(&temp1);
            uint32_t temp2 = linearDST[idx + 2];
            carvePlanes[1] = *reinterpret_cast<float*>(&temp2);

            char carveType1 = (header5 >> 2) & 3;
            char carveType2 = header5 & 3;

            if (carveType1 == 2) {
                unsigned sign = dirSgn[carveType2];
                if (carvePlanes[sign] != 0) {
                    float ts1 = (carvePlanes[sign] - ray.o[carveType2]) * invDir[carveType2];
                    tMax = std::min(ts1, tMax);
                }
                if (carvePlanes[sign ^ 1] != 0) {
                    float ts2 = (carvePlanes[sign ^ 1] - ray.o[carveType2]) * invDir[carveType2];
                    tMin = std::max(ts2, tMin);
                }
                dstPlaneIntersections++;
                if (tMin > tMax + delta)
                    goto pop;
                int offset = headerOffset & OFFSET;

                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            } else {
                unsigned axis1 = carveType1 >> 1;
                unsigned axis2 = (carveType1 & 1) + 1;

                float tMin0, tMin1, tMax0, tMax1;
                tMin0 = (carvePlanes[0] - ray.o[axis1]) * invDir[axis1];
                tMax0 = tMax;
                if (dirSgn[axis1] == (carveType2 >> 1)) {
                    tMax0 = tMin0;
                    tMin0 = tMin;
                }
                tMin1 = (carvePlanes[1] - ray.o[axis2]) * invDir[axis2];
                tMax1 = tMax;
                if (dirSgn[axis2] == (carveType2 & 1)) {
                    tMax1 = tMin1;
                    tMin1 = tMin;
                }

                tMin = std::max(tMin, std::max(tMin0, tMin1));
                tMax = std::min(tMax, std::min(tMax0, tMax1));
                dstPlaneIntersections++;
                if (tMin > tMax + delta)
                    goto pop;
                int offset = headerOffset & OFFSET;
                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            }
        }

    leaf:
        for (unsigned i = idx;; i++) {
            dstTriangleIntersections++;
            pstd::optional<ShapeIntersection> primSi =
                primitives[i].Intersect(ray, tMax + delta);
            if (primSi.has_value()) {
                si = primSi;
                tMax = si->tHit;
            }
            // Test if last Triangle in node
            if (primitives[i].isLast()) {
                break;
            }
        }
    pop:
        while (true) {
            if (stackPtr == -1) {
                return si;
            }
            StackItem item = stack[stackPtr--];
            idx = item.idx;
            tMin = item.tMin;
            if (!si.has_value()) {
                tMax = item.tMax;
                break;
            } else if (tMin < si->tHit) {
                tMax = std::min(si->tHit, item.tMax);
                break;
            }
        }
    }
    return si;
}

bool DSTAggregate::IntersectP(const Ray &ray, Float globalTMax) const {
    dstRays++;
    //For code documentation look at DST supplemental materials Listing 7. DST Traversal Kernel
    float tMin = 0;
    float tMax = globalTMax;

    unsigned headerOffset;
    unsigned header5;
    bool leaf;
    unsigned idx = 0;

    const Vector3f invDir = {1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z}; 
    const int dirSgn[3] = {invDir[0] < 0, invDir[1] < 0, invDir[2] < 0};
    if (!globalBB.IntersectP(ray.o, ray.d, tMax, invDir, dirSgn))
        return false;
    if (tMin < 0)
        tMin = 0;

    StackItem* stack = new StackItem[maximumDepth];
    int stackPtr = -1;

    while (true) {
        headerOffset = linearDST[idx];
        header5 = headerOffset >> 27;
        leaf = (headerOffset >> 26) & 1;
        
        if (header5 >> 4 == 0) {
            if (leaf) {
                idx = headerOffset & OFFSET;
                goto leaf;
            }
            float splitPlanes[2];
            uint32_t temp1 = linearDST[idx + 1];
            splitPlanes[0] = *reinterpret_cast<float*>(&temp1);
            uint32_t temp2 = linearDST[idx + 2];
            splitPlanes[1] = *reinterpret_cast<float*>(&temp2);
            idx = headerOffset & OFFSET;

            unsigned axis = header5 >> 2;
            unsigned sign = dirSgn[axis];
            unsigned diff = header5 & 3;
            float ts1 = (splitPlanes[sign] - ray.o[axis]) * invDir[axis];
            float ts2 = (splitPlanes[sign ^ 1] - ray.o[axis]) * invDir[axis];

            sign *= diff;
            dstPlaneIntersections += 2;
            if (tMax >= ts2) {
                float tNext = std::max(tMin, ts2);
                if (tMin <= ts1) {
                    //stack.stack[] instead of stack[] in documentation??
                    stack[++stackPtr] = StackItem(idx + (sign ^ diff), tNext, tMax);
                    idx += sign;
                    tMax = std::min(tMax, ts1);
                } else {
                    idx += sign;
                    tMin = tNext;
                }
                continue;
            } else {
                if (tMin <= ts1) {
                    idx += sign;
                    tMax = std::min(tMax, ts1);
                    continue;
                } else {
                    goto pop;
                }
            }
        } else {
            float carvePlanes[2];
            uint32_t temp1 = linearDST[idx + 1];
            carvePlanes[0] = *reinterpret_cast<float*>(&temp1);
            uint32_t temp2 = linearDST[idx + 2];
            carvePlanes[1] = *reinterpret_cast<float*>(&temp2);

            char carveType1 = header5 >> 2 & 3;
            char carveType2 = header5 & 3;

            if (carveType1 == 2) {
                unsigned sign = dirSgn[carveType2];
                float ts1 = (carvePlanes[sign] - ray.o[carveType2]) * invDir[carveType2];
                float ts2 = (carvePlanes[sign ^ 1] - ray.o[carveType2]) * invDir[carveType2];
                tMax = std::min(ts1, tMax);
                tMin = std::max(ts2, tMin);
                dstPlaneIntersections++;
                if (tMin > tMax)
                    goto pop;
                int offset = headerOffset & OFFSET;

                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            } else {
                unsigned axis1 = carveType1 >> 1;
                unsigned axis2 = (carveType1 & 1) + 1;

                float tMin0, tMin1, tMax0, tMax1;
                tMin0 = (carvePlanes[0] - ray.o[axis1]) * invDir[axis1];
                tMax0 = tMax;
                if (dirSgn[axis1] == (carveType2 >> 1)) {
                    tMax0 = tMin0;
                    tMin0 = tMin;
                }
                tMin1 = (carvePlanes[1] - ray.o[axis2]) * invDir[axis2];
                tMax1 = tMax;
                if (dirSgn[axis2] == (carveType2 & 1)) {
                    tMax1 = tMin1;
                    tMin1 = tMin;
                }

                tMin = std::max(tMin, std::max(tMin0, tMin1));
                tMax = std::min(tMax, std::min(tMax0, tMax1));
                dstPlaneIntersections++;
                if (tMin > tMax) 
                    goto pop;
                int offset = headerOffset & OFFSET;
                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            }
        }

    leaf:
        for (unsigned i = idx;; i++) {
            dstTriangleIntersections++;
            if (primitives[i].Intersect(ray, tMax))
                return true;
            // Test if last Triangle in node
            if (primitives[i].isLast()) {
                break;
            }
        }
    pop: 
        while (true) {
            if (stackPtr == -1) {
                return false;
            }
            StackItem item = stack[stackPtr--];
            idx = item.idx;
            tMin = item.tMin;
            tMax = std::min(globalTMax, item.tMax);
            break;
        }
    }
    return false;
}

float DSTAggregate::calculateOverallSAH(DSTBuildNode node, Bounds3f parentBoundingBox, int depth) {
    float SAH = 0.f;
    Bounds3f boundingBox(parentBoundingBox);
    int header = (node.GetFlag() >> 27) & 15;
    if (node.IsLeaf()) {
        SAH = node.NTriangles();
    } else if (node.IsCarving()) {
        float traversalCost = 0.f;
        if (header >> 2 == 2) {
            traversalCost = 0.3;
            if ((header & 3) == 0) {
                if (node.plane2 != 0)
                boundingBox.pMin.x = node.plane2;
                if (node.plane1 != 0)
                boundingBox.pMax.x = node.plane1;
            } else if ((header & 3) == 1) {
                if (node.plane2 != 0)
                boundingBox.pMin.y = node.plane2;
                if (node.plane1 != 0)
                boundingBox.pMax.y = node.plane1;
            } else if ((header & 3) == 2) {
                if (node.plane2 != 0)
                boundingBox.pMin.z = node.plane2;
                if (node.plane1 != 0)
                boundingBox.pMax.z = node.plane1;
            }
        } else if (header >> 2 == 0) { //xy
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.plane1 != 0)
                boundingBox.pMax.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.y = node.plane2;
            } else if ((header & 3) == 1) {
                if (node.plane1 != 0)
                boundingBox.pMax.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.y = node.plane2;
            } else if ((header & 3) == 2) {
                if (node.plane1 != 0)
                boundingBox.pMin.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.y = node.plane2;
            } else if ((header & 3) == 3) {
                if (node.plane1 != 0)
                boundingBox.pMin.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.y = node.plane2;
            }
        } else if (header >> 2 == 1) {  //xz
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.plane1 != 0)
                boundingBox.pMax.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.z = node.plane2;
            } else if ((header & 3) == 1) {
                if (node.plane1 != 0)
                boundingBox.pMax.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.z = node.plane2;
            } else if ((header & 3) == 2) {
                if (node.plane1 != 0)
                boundingBox.pMin.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.z = node.plane2;
            } else if ((header & 3) == 3) {
                if (node.plane1 != 0)
                boundingBox.pMin.x = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.z = node.plane2;
            }
        } else if (header >> 2 == 3) {  //yz
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.plane1 != 0)
                boundingBox.pMax.y = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.z = node.plane2;
            } else if ((header & 3) == 1) {
                if (node.plane1 != 0)
                boundingBox.pMax.y = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.z = node.plane2;
            } else if ((header & 3) == 2) {
                if (node.plane1 != 0)
                boundingBox.pMin.y = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMax.z = node.plane2;
            } else if ((header & 3) == 3) {
                if (node.plane1 != 0)
                boundingBox.pMin.y = node.plane1;
                if (node.plane2 != 0)
                boundingBox.pMin.z = node.plane2;
            }
        }
        if (node.isCarvingLeaf()) {
            SAH = traversalCost + (boundingBox.SurfaceArea() / parentBoundingBox.SurfaceArea()) * node.NTriangles();
        } else {
            SAH = traversalCost + (boundingBox.SurfaceArea() / parentBoundingBox.SurfaceArea()) * calculateOverallSAH(*node.children[0], boundingBox, depth + 1);
        }
    } else {
        Bounds3f boundingBox1(boundingBox);
        Bounds3f boundingBox2(boundingBox);
        if (header >> 2 == 0) {
            boundingBox1.pMax.x = node.plane1;
            boundingBox2.pMin.x = node.plane2;
        } else if (header >> 2 == 1) {
            boundingBox1.pMax.y = node.plane1;
            boundingBox2.pMin.y = node.plane2;
        } else if (header >> 2 == 2) {
            boundingBox1.pMax.z = node.plane1;
            boundingBox2.pMin.z = node.plane2;
        }
        float S = parentBoundingBox.SurfaceArea();
        SAH = TRAVERSAL_COST_DST_SN +
               (boundingBox1.SurfaceArea() / S) * calculateOverallSAH(*node.children[0], boundingBox1, depth + 1) +
               (boundingBox2.SurfaceArea() / S) * calculateOverallSAH(*node.children[1], boundingBox2, depth + 1);
    }
    return SAH;
}

DSTBuildNode *DSTAggregate::BuildRecursiveFromBVH(ThreadLocal<Allocator> &threadAllocators,
                                             LinearBVHNode* nodes,
                                             int currentNodeIndex, int currentDepth) {
    
    Allocator alloc = threadAllocators.Get();
    DSTBuildNode *node = alloc.new_object<DSTBuildNode>();
    LinearBVHNode parentNodeBVH = nodes[currentNodeIndex];
    LinearBVHNode leftChildNodeBVH = nodes[currentNodeIndex + 1];
    LinearBVHNode rightChildNodeBVH = nodes[parentNodeBVH.secondChildOffset];
    int indexToLeftBVHChild = currentNodeIndex + 1;
    int IndexToRightBVHChild = parentNodeBVH.secondChildOffset;

    if (parentNodeBVH.nPrimitives > 0) {
        node->InitLeaf(parentNodeBVH.primitivesOffset, currentDepth, parentNodeBVH.nPrimitives, parentNodeBVH.bounds);
        primitives[parentNodeBVH.primitivesOffset + parentNodeBVH.nPrimitives - 1].setLast();
        if (currentDepth > maximumDepth)
            maximumDepth = currentDepth;
        return node;
    }

    //We determine the composition of splitting and carving nodes with the lowest SAH
    //The first 6 int of the nodeComposition are for up to 3 carving nodes of the left child and the 7th to 12th int are for up to 3 carving nodes of the right child
    //The 13th int is for the splitting axis, each carving node has its first int 0=no node, 1=sigle-axis, 2=double-axis and the second int for the plane and corner int
    std::vector<int> bestNodeComposition(13, 0);
    std::vector<float> bestNodesPlanes(14, 0.f);
    std::vector<Bounds3f> bestNodesBB(6);
    float bestSAH = FLT_MAX;
    Bounds3f parentBounds = parentNodeBVH.bounds;
    bool switchChildren = false;
    for (int i = 0; i <= 2; i++) {
        bool switchChildrenLocal = false;
        std::vector<int> nodeComposition(13, 0);
        std::vector<float> nodesPlanes(14, 0.f);
        std::vector<Bounds3f> nodesBB(7);
        Bounds3f leftChildBounds = parentBounds;
        Bounds3f rightChildBounds = parentBounds;
        nodeComposition[12] = i;
        if (i == 0) {
            float leftCenterX = (leftChildNodeBVH.bounds.pMin.x + leftChildNodeBVH.bounds.pMax.x) / 2;
            float rightCenterX = (rightChildNodeBVH.bounds.pMin.x + rightChildNodeBVH.bounds.pMax.x) / 2;
            if (leftCenterX < rightCenterX) {
                leftChildBounds.pMax.x = leftChildNodeBVH.bounds.pMax.x;
                rightChildBounds.pMin.x = rightChildNodeBVH.bounds.pMin.x;
                nodesPlanes[12] = leftChildNodeBVH.bounds.pMax.x;
                nodesPlanes[13] = rightChildNodeBVH.bounds.pMin.x;
            } else {
                switchChildrenLocal = true;
                leftChildBounds.pMax.x = rightChildNodeBVH.bounds.pMax.x;
                rightChildBounds.pMin.x = leftChildNodeBVH.bounds.pMin.x;
                nodesPlanes[12] = rightChildNodeBVH.bounds.pMax.x;
                nodesPlanes[13] = leftChildNodeBVH.bounds.pMin.x;
            }
        } else if (i == 1) {
            float leftCenterY = (leftChildNodeBVH.bounds.pMin.y + leftChildNodeBVH.bounds.pMax.y) / 2;
            float rightCenterY = (rightChildNodeBVH.bounds.pMin.y + rightChildNodeBVH.bounds.pMax.y) / 2;
            if (leftCenterY < rightCenterY) {
                leftChildBounds.pMax.y = leftChildNodeBVH.bounds.pMax.y;
                rightChildBounds.pMin.y = rightChildNodeBVH.bounds.pMin.y;
                nodesPlanes[12] = leftChildNodeBVH.bounds.pMax.y;
                nodesPlanes[13] = rightChildNodeBVH.bounds.pMin.y;
            } else {
                switchChildrenLocal = true;
                leftChildBounds.pMax.y = rightChildNodeBVH.bounds.pMax.y;
                rightChildBounds.pMin.y = leftChildNodeBVH.bounds.pMin.y;
                nodesPlanes[12] = rightChildNodeBVH.bounds.pMax.y;
                nodesPlanes[13] = leftChildNodeBVH.bounds.pMin.y;
            }
        } else {
            float leftCenterZ = (leftChildNodeBVH.bounds.pMin.z + leftChildNodeBVH.bounds.pMax.z) / 2;
            float rightCenterZ = (rightChildNodeBVH.bounds.pMin.z + rightChildNodeBVH.bounds.pMax.z) / 2;
            if (leftCenterZ < rightCenterZ) {
                leftChildBounds.pMax.z = leftChildNodeBVH.bounds.pMax.z;
                rightChildBounds.pMin.z = rightChildNodeBVH.bounds.pMin.z;
                nodesPlanes[12] = leftChildNodeBVH.bounds.pMax.z;
                nodesPlanes[13] = rightChildNodeBVH.bounds.pMin.z;
            } else {
                switchChildrenLocal = true;
                leftChildBounds.pMax.z = rightChildNodeBVH.bounds.pMax.z;
                rightChildBounds.pMin.z = leftChildNodeBVH.bounds.pMin.z;
                nodesPlanes[12] = rightChildNodeBVH.bounds.pMax.z;
                nodesPlanes[13] = leftChildNodeBVH.bounds.pMin.z;
            }
        }
        nodesBB[0] = leftChildBounds;
        nodesBB[1] = rightChildBounds;
        float sumSAH;
        if (switchChildrenLocal)
            sumSAH = determineSAH(nodeComposition, nodesPlanes, nodesBB, leftChildBounds, rightChildNodeBVH.bounds, 0, parentBounds.SurfaceArea(), nodes, parentNodeBVH.secondChildOffset) +
                     determineSAH(nodeComposition, nodesPlanes, nodesBB, rightChildBounds,
                                  leftChildNodeBVH.bounds, 6, parentBounds.SurfaceArea(),
                                  nodes, currentDepth + 1);
        else
            sumSAH = determineSAH(nodeComposition, nodesPlanes, nodesBB, leftChildBounds, leftChildNodeBVH.bounds, 0, parentBounds.SurfaceArea(), nodes, currentDepth + 1) +
                determineSAH(nodeComposition, nodesPlanes, nodesBB, rightChildBounds, rightChildNodeBVH.bounds, 6, parentBounds.SurfaceArea(), nodes, parentNodeBVH.secondChildOffset);
        if (sumSAH < bestSAH) {
            switchChildren = switchChildrenLocal;
            bestNodeComposition = nodeComposition;
            bestNodesPlanes = nodesPlanes;
            bestNodesBB = nodesBB;
            bestSAH = sumSAH;
        }
    }

    if (switchChildren) {
        LinearBVHNode temp = leftChildNodeBVH;
        leftChildNodeBVH = rightChildNodeBVH;
        rightChildNodeBVH = temp;
        int tempI = indexToLeftBVHChild;
        indexToLeftBVHChild = IndexToRightBVHChild;
        IndexToRightBVHChild = tempI;
    }
    
    std::vector<float> irrelevantPlanes = {0.f, 0.f};
    Bounds3f leftBBAfterSplit = carve(parentNodeBVH.bounds, leftChildNodeBVH.bounds, {bestNodeComposition[12] + 4}, irrelevantPlanes);
    Bounds3f rightBBAfterSplit = carve(parentNodeBVH.bounds, rightChildNodeBVH.bounds, {bestNodeComposition[12] + 1}, irrelevantPlanes);

    bool lastLeftChildIsLeaf = leftChildNodeBVH.nPrimitives;
    bool lastRightChildIsLeaf = rightChildNodeBVH.nPrimitives;

    int nextCarvingNode = 0;
    DSTBuildNode *nextNodeP1 = alloc.new_object<DSTBuildNode>();
    DSTBuildNode *leftChildNode = alloc.new_object<DSTBuildNode>();
    int depthLevel = currentDepth;
    int setSize = 0;
    while (bestNodeComposition[nextCarvingNode] && nextCarvingNode < 6) {
        setSize++;
        DSTBuildNode *nextNode = alloc.new_object<DSTBuildNode>();
        depthLevel++;
        Bounds3f carvingNodeBB;
        if (nextCarvingNode == 0 && (!lastLeftChildIsLeaf || bestNodeComposition[2]))
            carvingNodeBB = leftBBAfterSplit;
        else
            carvingNodeBB = leftChildNodeBVH.bounds;
        if (bestNodeComposition[nextCarvingNode] == 3) {
            if (bestNodeComposition[nextCarvingNode + 2] && nextCarvingNode < 4) {
                nextNodeP1->InitSingleAxisCarving(
                    bestNodeComposition[nextCarvingNode + 1], nextNode, 0, 0, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB, leftChildNodeBVH.nPrimitives);
            } else {
                nextNodeP1->InitSingleAxisCarving(
                    bestNodeComposition[nextCarvingNode + 1], nextNode,
                    lastLeftChildIsLeaf, leftChildNodeBVH.primitivesOffset, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    leftChildNodeBVH.nPrimitives);
            }
        } else {
            int planeAxesAndCornerType = (bestNodeComposition[nextCarvingNode] - 1) << 2 |
                                         bestNodeComposition[nextCarvingNode + 1];
            if (bestNodeComposition[nextCarvingNode + 2] && nextCarvingNode < 4) {
                nextNodeP1->InitDoubleAxisCarving(
                    planeAxesAndCornerType, nextNode, 0, 0, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    leftChildNodeBVH.nPrimitives);
            } else {
                nextNodeP1->InitDoubleAxisCarving(
                    planeAxesAndCornerType, nextNode,
                    lastLeftChildIsLeaf, leftChildNodeBVH.primitivesOffset, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    leftChildNodeBVH.nPrimitives);
            }
        }
        if (nextCarvingNode == 0)
            leftChildNode = nextNodeP1;
        nextNodeP1 = nextNode;
        nextCarvingNode = nextCarvingNode + 2;
    }
    if (lastLeftChildIsLeaf)
        primitives[leftChildNodeBVH.primitivesOffset + leftChildNodeBVH.nPrimitives - 1].setLast();
    if (depthLevel + 1 > maximumDepth)
        maximumDepth = currentDepth + 1;
    if (nextCarvingNode == 0)
        leftChildNode = BuildRecursiveFromBVH(threadAllocators, nodes, indexToLeftBVHChild, depthLevel + 1);
    else  if (!lastLeftChildIsLeaf)
        *nextNodeP1 = *BuildRecursiveFromBVH(threadAllocators, nodes, indexToLeftBVHChild, depthLevel + 1);
    if (setSize == 0) {
        dstSetsWithZero++;
    } else if (setSize == 1) {
        dstSetsWithOne++;
    } else if(setSize == 2) {
        dstSetsWithTwo++;
    } else {
        dstSetsWithThree++;
    }

    nextCarvingNode = 6;
    DSTBuildNode *nextNodeP2 = alloc.new_object<DSTBuildNode>();
    DSTBuildNode *rightChildNode = alloc.new_object<DSTBuildNode>();
    depthLevel = currentDepth;
    setSize = 0;
    while (bestNodeComposition[nextCarvingNode] && nextCarvingNode < 12) {
        setSize++;
        DSTBuildNode *nextNode = alloc.new_object<DSTBuildNode>();
        depthLevel++;
        Bounds3f carvingNodeBB;
        if (nextCarvingNode == 6 && (!lastRightChildIsLeaf || bestNodeComposition[8]))
            carvingNodeBB = rightBBAfterSplit;
        else
            carvingNodeBB = rightChildNodeBVH.bounds;
        if (bestNodeComposition[nextCarvingNode] == 3) {
            if (bestNodeComposition[nextCarvingNode + 2] && nextCarvingNode < 10) {
                nextNodeP2->InitSingleAxisCarving(
                    bestNodeComposition[nextCarvingNode + 1], nextNode, 0, 0, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    rightChildNodeBVH.nPrimitives);
            } else {
                nextNodeP2->InitSingleAxisCarving(
                    bestNodeComposition[nextCarvingNode + 1], nextNode,
                    lastRightChildIsLeaf, rightChildNodeBVH.primitivesOffset, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    rightChildNodeBVH.nPrimitives);
            }
        } else {
            int planeAxesAndCornerType = (bestNodeComposition[nextCarvingNode] - 1) << 2 |
                                         bestNodeComposition[nextCarvingNode + 1];
            if (bestNodeComposition[nextCarvingNode + 2] && nextCarvingNode < 10) {
                nextNodeP2->InitDoubleAxisCarving(
                    planeAxesAndCornerType, nextNode, 0, 0, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    rightChildNodeBVH.nPrimitives);
            } else {
                nextNodeP2->InitDoubleAxisCarving(
                    planeAxesAndCornerType, nextNode,
                    lastRightChildIsLeaf, rightChildNodeBVH.primitivesOffset, depthLevel,
                    bestNodesPlanes[nextCarvingNode],
                    bestNodesPlanes[nextCarvingNode + 1], carvingNodeBB,
                    rightChildNodeBVH.nPrimitives);
            }
        }
        if (nextCarvingNode == 6)
            rightChildNode = nextNodeP2;
        nextNodeP2 = nextNode; 
        nextCarvingNode = nextCarvingNode + 2;
    }
    if (lastRightChildIsLeaf)
        primitives[rightChildNodeBVH.primitivesOffset + rightChildNodeBVH.nPrimitives - 1].setLast();
    if (depthLevel + 1 > maximumDepth)
        maximumDepth = currentDepth + 1;
    if (nextCarvingNode == 6)
        rightChildNode = BuildRecursiveFromBVH(
            threadAllocators, nodes, IndexToRightBVHChild, depthLevel + 1);
    else if (!lastRightChildIsLeaf)
        *nextNodeP2 = *BuildRecursiveFromBVH(threadAllocators, nodes, IndexToRightBVHChild, depthLevel + 1);
    if (setSize == 0) {
        dstSetsWithZero++;
    } else if (setSize == 1) {
        dstSetsWithOne++;
    } else if (setSize == 2) {
        dstSetsWithTwo++;
    } else {
        dstSetsWithThree++;
    }
    dstNumberOfCarvingNodeSets += 2;
    node->InitSplittingNode(bestNodeComposition[12], leftChildNode, rightChildNode,
                            currentDepth, bestNodesPlanes[12], bestNodesPlanes[13], parentNodeBVH.bounds);

    return node;
}

float determineSAH(std::vector<int> &nodeComposition, std::vector<float> &nodesPlanes, std::vector<Bounds3f> &nodesBB, Bounds3f parentBB, Bounds3f childBB,
                   int positionOfNextNode, float S, LinearBVHNode *nodes,
                   int currentNodeIndex) {
    std::vector<int> sidesToCarve;
    float delta = 0.0001;
    //sides to carve must be ordered first by x,y,z and then by min, max
    if (parentBB.pMin.x + delta < childBB.pMin.x)
        sidesToCarve.push_back(1);
    if (parentBB.pMax.x > childBB.pMax.x + delta)
        sidesToCarve.push_back(4);
    if (parentBB.pMin.y + delta < childBB.pMin.y)
        sidesToCarve.push_back(2);
    if (parentBB.pMax.y > childBB.pMax.y + delta)
        sidesToCarve.push_back(5);
    if (parentBB.pMin.z + delta < childBB.pMin.z)
        sidesToCarve.push_back(3);
    if (parentBB.pMax.z > childBB.pMax.z + delta)
        sidesToCarve.push_back(6);
    float SAH = 0;
    if (sidesToCarve.size() >= 5) {
        SAH =
            carveFiveSides(sidesToCarve, nodeComposition, nodesPlanes, nodesBB, positionOfNextNode,
                           S, parentBB.SurfaceArea(), parentBB, childBB, nodes, currentNodeIndex);
    } else if (sidesToCarve.size() >= 3) {
        SAH = carveThreeOrFourSides(sidesToCarve, nodeComposition,
                                    nodesPlanes, nodesBB, positionOfNextNode, S, parentBB.SurfaceArea(),
                                    parentBB, childBB, nodes, currentNodeIndex);
    } else if (sidesToCarve.size() >= 1) {
        SAH = carveOneOrTwoSides(sidesToCarve, nodeComposition, nodesPlanes, positionOfNextNode, S, parentBB.SurfaceArea(), parentBB,
                                 childBB, nodes, currentNodeIndex);
    }
    return SAH;
}

float carveOneOrTwoSides(std::vector<int> sidesToCarve, std::vector<int> &nodeComposition,
                         std::vector<float> &nodesPlanes, int positionOfNextNode, float S,
                         float Sn, Bounds3f parentBB, Bounds3f childBB,
                         LinearBVHNode *nodes, int currentNodeIndex) {
    //1=xmin, 2=ymin, 3=zmin, 4=xmax, 5=ymax, 6=zmax, 0=noMoreCut
    int firstSideToCarve = sidesToCarve[0];
    int secondSideToCarve = 0;
    if (sidesToCarve.size() > 1)
        secondSideToCarve = sidesToCarve[1];

    std::vector<float> planes = {0.f, 0.f};
    Bounds3f carvedBB =
        carve(parentBB, childBB, sidesToCarve, planes);
    nodesPlanes[positionOfNextNode] = planes[0];
    if (sidesToCarve.size() > 1)
        nodesPlanes[positionOfNextNode + 1] = planes[1];
    else 
        nodesPlanes[positionOfNextNode + 1] = NULL;

    //Only one side must be carved -> I choose a single axis carving node
    if (secondSideToCarve == 0 || secondSideToCarve - firstSideToCarve == 3) {
        nodeComposition[positionOfNextNode] = 3; //set first node composition entry of a CN to its first nodeType +1 (so 0 can be dafault and ignored)
        //second node composition entry can be 0 as its not checked for default value
        if (firstSideToCarve < 4) {//carve min value
            nodeComposition[positionOfNextNode + 1] = firstSideToCarve - 1;
        } else {//carve max value
            //the first plane always carves min value, the second plane always max value
            nodeComposition[positionOfNextNode + 1] = firstSideToCarve - 4;
            nodesPlanes[positionOfNextNode + 1] = nodesPlanes[positionOfNextNode];
            nodesPlanes[positionOfNextNode] = NULL;
        }
        return 0.3 * (Sn / S) + (carvedBB.SurfaceArea() / S) * getNTrianglesInSubTree(nodes, currentNodeIndex);
    }
    if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 2) { //xy
        nodeComposition[positionOfNextNode] = 1; // we want 0 but we set value +1 so default can be 0
        nodeComposition[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
    } else if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 0) { //xz
        nodeComposition[positionOfNextNode] = 2;
        nodeComposition[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
    } else if (firstSideToCarve % 3 == 2 && secondSideToCarve % 3 == 0) { //yz
        nodeComposition[positionOfNextNode] = 4; //3 is for single-axis CN therefore 4
        nodeComposition[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
    }
    return 0.5 * (Sn / S) + (carvedBB.SurfaceArea() / S) * getNTrianglesInSubTree(nodes, currentNodeIndex);
}

float carveThreeOrFourSides(const std::vector<int> sidesToCarve,
                            std::vector<int> &nodeComposition,
                            std::vector<float> &nodesPlanes,
                            std::vector<Bounds3f> &nodesBB, int positionOfNextNode,
                            float S, float Sn, Bounds3f parentBB, Bounds3f childBB,
                            LinearBVHNode *nodes, int currentNodeIndex) {
    float bestSAH = FLT_MAX;
    std::vector<int> bestNodeComposition = nodeComposition;
    std::vector<float> bestNodePlanes = nodesPlanes;
    std::vector<Bounds3f> bestNodeBB = nodesBB;
    for (int i = 0; i < sidesToCarve.size(); i++) {
        for (int j = i + 1; j < sidesToCarve.size(); j++) {
            int firstSideToCarve = sidesToCarve[i];
            int secondSideToCarve = sidesToCarve[j];
            std::vector<int> carvingSides = {firstSideToCarve, secondSideToCarve};
            std::vector<int> sidesToCarveCopy(sidesToCarve);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + j);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + i);
            std::vector<int> nodeCompositionCopy(nodeComposition);
            std::vector<float> nodesPlanesCopy(nodesPlanes);
            std::vector<Bounds3f> nodesBBCopy(nodesBB);
            std::vector<float> planes = {0.f, 0.f};
            Bounds3f carvedBB = carve(parentBB, childBB, carvingSides, planes);
            nodesBBCopy[positionOfNextNode / 2] = carvedBB;
            nodesPlanesCopy[positionOfNextNode] = planes[0];
            nodesPlanesCopy[positionOfNextNode + 1] = planes[1];
            float SnCarved = carvedBB.SurfaceArea();

            if (secondSideToCarve - firstSideToCarve == 3) { 
                nodeCompositionCopy[positionOfNextNode] = 3;
                nodeCompositionCopy[positionOfNextNode + 1] = firstSideToCarve - 1;
                float SAH = 0.3 * (Sn / S) +
                            carveOneOrTwoSides(sidesToCarveCopy, nodeCompositionCopy,
                                               nodesPlanesCopy, positionOfNextNode + 2, S,
                                               SnCarved, carvedBB, childBB, nodes,
                                               currentNodeIndex);
                if (SAH < bestSAH) {
                    bestNodeComposition = nodeCompositionCopy;
                    bestNodePlanes = nodesPlanesCopy;
                    bestNodeBB = nodesBBCopy;
                    bestSAH = SAH;
                }
            } else {
                if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 2) {  // xy
                    nodeCompositionCopy[positionOfNextNode] = 1;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                } else if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 0) {  // xz
                    nodeCompositionCopy[positionOfNextNode] = 2;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                } else if (firstSideToCarve % 3 == 2 && secondSideToCarve % 3 == 0) {  // yz
                    nodeCompositionCopy[positionOfNextNode] = 4;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                }
                float SAH = 0.5 * (Sn / S) +
                            carveOneOrTwoSides(sidesToCarveCopy, nodeCompositionCopy,
                                               nodesPlanesCopy, positionOfNextNode + 2, S,
                                               SnCarved, carvedBB, childBB, nodes,
                                               currentNodeIndex);
                if (SAH < bestSAH) {
                    bestNodeComposition = nodeCompositionCopy;
                    bestNodePlanes = nodesPlanesCopy;
                    bestNodeBB = nodesBBCopy;
                    bestSAH = SAH;
                }
            }
        }
    }
    nodeComposition = bestNodeComposition;
    nodesPlanes = bestNodePlanes;
    nodesBB = bestNodeBB;
    return bestSAH;
}

float carveFiveSides(const std::vector<int> sidesToCarve,
                     std::vector<int> &nodeComposition, std::vector<float> &nodesPlanes,
                     std::vector<Bounds3f> &nodesBB,
                     int positionOfNextNode, float S, float Sn, Bounds3f parentBB, Bounds3f childBB, LinearBVHNode *nodes,
                     int currentNodeIndex) {
    float bestSAH = FLT_MAX;
    std::vector<int> bestNodeComposition = nodeComposition;
    std::vector<float> bestNodePlanes = nodesPlanes;
    std::vector<Bounds3f> bestNodeBB = nodesBB;
    for (int i = 0; i < sidesToCarve.size(); i++) {
        for (int j = i + 1; j < sidesToCarve.size(); j++) {
            int firstSideToCarve = sidesToCarve[i];
            int secondSideToCarve = sidesToCarve[j];
            std::vector<int> carvingSides = {firstSideToCarve, secondSideToCarve};
            std::vector<int> sidesToCarveCopy(sidesToCarve);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + j);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + i);
            std::vector<int> nodeCompositionCopy(nodeComposition);
            std::vector<float> nodesPlanesCopy(nodesPlanes);
            std::vector<Bounds3f> nodesBBCopy(nodesBB);
            std::vector<float> planes = {0.f, 0.f};
            Bounds3f carvedBB = carve(parentBB, childBB, carvingSides, planes);
            nodesBBCopy[positionOfNextNode / 2] = carvedBB;
            nodesPlanesCopy[positionOfNextNode] = planes[0];
            nodesPlanesCopy[positionOfNextNode + 1] = planes[1];
            float SnCarved = carvedBB.SurfaceArea();

            if (secondSideToCarve - firstSideToCarve == 3) {
                nodeCompositionCopy[positionOfNextNode] = 3;
                nodeCompositionCopy[positionOfNextNode + 1] = firstSideToCarve - 1;
                float SAH = 0.3 * (Sn / S) +
                            carveThreeOrFourSides(sidesToCarveCopy, nodeCompositionCopy,
                                                  nodesPlanesCopy, nodesBBCopy, positionOfNextNode + 2, S, SnCarved,
                                carvedBB, childBB, nodes, currentNodeIndex);
                if (SAH < bestSAH) {
                    bestNodeComposition = nodeCompositionCopy;
                    bestNodePlanes = nodesPlanesCopy;
                    bestNodeBB = nodesBBCopy;
                    bestSAH = SAH;
                }
            } else {
                if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 2) {  // xy
                    nodeCompositionCopy[positionOfNextNode] = 1;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                } else if (firstSideToCarve % 3 == 1 && secondSideToCarve % 3 == 0) {  // xz
                    nodeCompositionCopy[positionOfNextNode] = 2;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                } else if (firstSideToCarve % 3 == 2 && secondSideToCarve % 3 == 0) {  // yz
                    nodeCompositionCopy[positionOfNextNode] = 4;
                    nodeCompositionCopy[positionOfNextNode + 1] = 2 * (firstSideToCarve < 4) + (secondSideToCarve < 4);
                }
                float SAH = 0.5 * (Sn / S) +
                            carveThreeOrFourSides(sidesToCarveCopy, nodeCompositionCopy,
                                                  nodesPlanesCopy, nodesBBCopy,
                                                  positionOfNextNode + 2, S, SnCarved,
                                carvedBB, childBB, nodes, currentNodeIndex);
                if (SAH < bestSAH) {
                    bestNodeComposition = nodeCompositionCopy;
                    bestNodePlanes = nodesPlanesCopy;
                    bestNodeBB = nodesBBCopy;
                    bestSAH = SAH;
                }
            }
        }
    }
    nodeComposition = bestNodeComposition;
    nodesPlanes = bestNodePlanes;
    nodesBB = bestNodeBB;
    return bestSAH;
}

Bounds3f carve(Bounds3f parentBB, Bounds3f childBB, std::vector<int> sidesToCarve, std::vector<float> &planes) {
    Bounds3f carvedBB = parentBB;
    int i = 0;
    for (int sideToCarve : sidesToCarve) {
        if (sideToCarve == 1) {
            carvedBB.pMin.x = childBB.pMin.x;
            planes[i] = childBB.pMin.x;
        } else if (sideToCarve == 2) {
            carvedBB.pMin.y = childBB.pMin.y;
            planes[i] = childBB.pMin.y;
        } else if (sideToCarve == 3) {
            carvedBB.pMin.z = childBB.pMin.z;
            planes[i] = childBB.pMin.z;
        } else if (sideToCarve == 4) {
            carvedBB.pMax.x = childBB.pMax.x;
            planes[i] = childBB.pMax.x;
        } else if (sideToCarve == 5) {
            carvedBB.pMax.y = childBB.pMax.y;
            planes[i] = childBB.pMax.y;
        } else if (sideToCarve == 6) {
            carvedBB.pMax.z = childBB.pMax.z;
            planes[i] = childBB.pMax.z;
        }
        i++;
    }
    return carvedBB;
}

int getNTrianglesInSubTree(LinearBVHNode *nodes, int currentNodeIndex) {
    LinearBVHNode parentNode = nodes[currentNodeIndex];
    LinearBVHNode leftChildNode = nodes[currentNodeIndex + 1];
    LinearBVHNode rightChildNode = nodes[parentNode.secondChildOffset];
    int nTriangles = 0;
    if (leftChildNode.nPrimitives) {
        nTriangles += leftChildNode.nPrimitives;
    } else {
        nTriangles += getNTrianglesInSubTree(nodes, currentNodeIndex + 1);
    }
    if (rightChildNode.nPrimitives) {
        nTriangles += rightChildNode.nPrimitives;
    } else {
        nTriangles += getNTrianglesInSubTree(nodes, parentNode.secondChildOffset);
    }
    return nTriangles;
}

int getNTrianglesInSubTree(DSTBuildNode node) {
    int nTriangles = 0;
    if (node.IsLeaf()) {
        nTriangles += node.NTriangles();
    } else if (node.IsCarving()) {
        if (node.isCarvingLeaf())
            nTriangles += node.NTriangles();
        else
            nTriangles += getNTrianglesInSubTree(*node.children[0]);
    } else {
        nTriangles += getNTrianglesInSubTree(*node.children[0]);
        nTriangles += getNTrianglesInSubTree(*node.children[1]);
    }
    return nTriangles;
}

void DSTAggregate::addNodeToDepthList(DSTBuildNode* node) {
    nodesPerDepthLevel[node->GetDepthLevel()].push_back(node);
    for (DSTBuildNode* child : node->children) {
        if (child != NULL)
            addNodeToDepthList(child);
    }
}

StackItem::StackItem(int idx, float tMin, float tMax) {
    this->idx = idx;
    this->tMax = tMax;
    this->tMin = tMin;
}

StackItem::StackItem() {
    this->idx = 0;
    this->tMax = Infinity;
    this->tMin = -Infinity;
}

struct WDSTBuildNode {
  public:
    void InitLeaf(int triangleOffset, int depthLevel, int nPrimitives) {
        header = 0b000001;
        this->triangleOffset = triangleOffset;
        this->depthLevel = depthLevel;
        this->nPrimitives = nPrimitives;
    }

    void InitExistingCarving(float plane1, float plane2, WDSTBuildNode *child, int header,
                             int triangleOffset, Bounds3f BB, int depthLevel,
                             int nPrimitives) {
        this->planes[0] = plane1;
        this->planes[1] = plane2;
        this->children[0] = child;
        this->header = header;
        this->triangleOffset = triangleOffset;
        this->depthLevel = depthLevel;
        this->nPrimitives = nPrimitives;
        boundingBox = BB;
    }

    void InitNewCarving(float plane1, float plane2, WDSTBuildNode *child, int header,
                             int triangleOffset, Bounds3f BB, int depthLevel, int nPrimitives) {
        this->header = header;
        if ((header & 0b011000) == 0b010000) {
            this->planes[0] = plane2;
            this->planes[1] = plane1;
        } else {
            this->planes[0] = plane1;
            this->planes[1] = plane2;
        }
        this->children[0] = child;
        this->triangleOffset = triangleOffset;
        this->depthLevel = depthLevel;
        this->nPrimitives = nPrimitives;
        boundingBox = BB;
    }

    bool IsLeaf() const { return (header & 0b000001); }
    bool IsCarving() const { return (header & 0b100000); }
    int offsetToFirstChild() const {
        if (children[0] == NULL)
            return triangleOffset;
        else
            return children[0]->offset; 
    }
    int size() const {
        if (header >> 5)
            return 2;
        if (header & 1)
            return 0;
        int size = 3;
        if (children[1] != NULL)
            size += 2;
        if (children[3] != NULL)
            size += 2;
        return size;
    }
    int GetDepthLevel() const { return depthLevel; }
    int PlaneAsInt(int idx) { return *reinterpret_cast<int *>(&planes[idx]); }

    uint32_t triangleOffset;
    std::array<WDSTBuildNode*, 4> children;
    std::array<float, 6> planes;
    Bounds3f boundingBox;
    int offset;
    int header;
    int axes; //wird nur für SN gebraucht, kann man eventuell noch eleganter speichern
    int depthLevel;
    int nPrimitives;
};

WDSTAggregate::WDSTAggregate(std::vector<Primitive> prims, DSTBuildNode node, Bounds3f globalBB)
    : primitives(std::move(prims)) {
    CHECK(!primitives.empty());

    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    using Resource = pstd::pmr::monotonic_buffer_resource;
    std::vector<std::unique_ptr<Resource>> threadBufferResources;
    ThreadLocal<Allocator> threadAllocators([&threadBufferResources]() {
        threadBufferResources.push_back(std::make_unique<Resource>());
        auto ptr = threadBufferResources.back().get();
        return Allocator(ptr);
    });
    maximumDepth = 0;

    WDSTBuildNode *root;

    this->globalBB = globalBB;
    root = BuildWDSTRecursively(threadAllocators, &node, globalBB.SurfaceArea(), 0);

    nodesPerDepthLevel.resize(maximumDepth + 1);
    addNodeToDepthList(root);
    int offset = 0;
    for (int i = 0; i < nodesPerDepthLevel.size(); i++) {
        while (nodesPerDepthLevel[i].size() != 0) {
            nodesPerDepthLevel[i].front()->offset = offset;
            offset += nodesPerDepthLevel[i].front()->size() + 1;
            nodesPerDepthLevel[i].pop_front();
            numberOfNodes++;
        }
    }

    /*
    std::cout << "Primitive List:" << '\n';
    for (int i = 0; i < primitives.size(); i++) {
        std::cout << primitives[i].Bounds().ToString() << "    " << primitives[i].isLast()
    << '\n';
    }
    std::cout << '\n' << '\n' << "Primitives in Tree:" << '\n';*/

    float SAH = this->calculateOverallSAH(*root, globalBB, 0);
    overallSAHCost = SAH * 1000000;

    wdstBytes = offset * 4;
    linearWDST.resize(offset);
    FlattenWDST(root);
}

float WDSTAggregate::calculateOverallSAH(WDSTBuildNode node, Bounds3f parentBoundingBox, int depth) {
    int header = node.header >> 1;
    Bounds3f boundingBox(parentBoundingBox);
    float SAH = 0.f;
    if (node.IsLeaf() && !node.IsCarving()) {
        SAH = node.nPrimitives;
    } else if (node.IsCarving()) {
        float traversalCost = 0.f;
        if (((header >> 2) & 3) == 2) {
            traversalCost = 0.3;
            if ((header & 3) == 0) {
                if (node.planes[1] != 0)
                    boundingBox.pMin.x = node.planes[1];
                if (node.planes[0] != 0)
                    boundingBox.pMax.x = node.planes[0];
            } else if ((header & 3) == 1) {
                if (node.planes[1] != 0)
                    boundingBox.pMin.y = node.planes[1];
                if (node.planes[0] != 0)
                    boundingBox.pMax.y = node.planes[0];
            } else if ((header & 3) == 2) {
                if (node.planes[1] != 0)
                    boundingBox.pMin.z = node.planes[1];
                if (node.planes[0] != 0)
                    boundingBox.pMax.z = node.planes[0];
            }
        } else if (((header >> 2) & 3) == 0) {  // xy
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.y = node.planes[1];
            } else if ((header & 3) == 1) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.y = node.planes[1];
            } else if ((header & 3) == 2) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.y = node.planes[1];
            } else if ((header & 3) == 3) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.y = node.planes[1];
            }
        } else if (((header >> 2) & 3) == 1) {  // xz
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.z = node.planes[1];
            } else if ((header & 3) == 1) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.z = node.planes[1];
            } else if ((header & 3) == 2) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.z = node.planes[1];
            } else if ((header & 3) == 3) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.x = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.z = node.planes[1];
            }
        } else if (((header >> 2) & 3) == 3) {  // yz
            traversalCost = 0.5;
            if ((header & 3) == 0) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.y = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.z = node.planes[1];
            } else if ((header & 3) == 1) {
                if (node.planes[0] != 0)
                    boundingBox.pMax.y = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.z = node.planes[1];
            } else if ((header & 3) == 2) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.y = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMax.z = node.planes[1];
            } else if ((header & 3) == 3) {
                if (node.planes[0] != 0)
                    boundingBox.pMin.y = node.planes[0];
                if (node.planes[1] != 0)
                    boundingBox.pMin.z = node.planes[1];
            }
        }
        if (node.IsLeaf()) {
            SAH = traversalCost + (boundingBox.SurfaceArea() / parentBoundingBox.SurfaceArea()) * node.nPrimitives;
        } else {
            SAH = traversalCost + (boundingBox.SurfaceArea() / parentBoundingBox.SurfaceArea()) * calculateOverallSAH(*node.children[0], boundingBox, depth + 1);
        }
    } else {
        Bounds3f boundingBox1(boundingBox);
        Bounds3f boundingBox2(boundingBox);
        int size = header >> 2;
        int axis = ((node.axes >> (2 * (size - 1))) & 3);
        if (axis == 0) {
            boundingBox1.pMax.x = node.planes[0];
            boundingBox2.pMin.x = node.planes[1];
        } else if (axis == 1) {
            boundingBox1.pMax.y = node.planes[0];
            boundingBox2.pMin.y = node.planes[1];
        } else if (axis == 2) {
            boundingBox1.pMax.z = node.planes[0];
            boundingBox2.pMin.z = node.planes[1];
        }
        if ((header & 3) == 0) {
            float S = parentBoundingBox.SurfaceArea();
            SAH += TRAVERSAL_COST_WDST_SMALL_SN +
                   (boundingBox1.SurfaceArea() / S) * calculateOverallSAH(*node.children[0], boundingBox1, depth + 1) +
                   (boundingBox2.SurfaceArea() / S) * calculateOverallSAH(*node.children[2], boundingBox2, depth + 1);
        } else if ((header & 3) == 1) {
            int axis2 = node.axes & 3;
            Bounds3f boundingBox21(boundingBox2);
            Bounds3f boundingBox22(boundingBox2);
            if (axis2 == 0) {
                boundingBox21.pMax.x = node.planes[4];
                boundingBox22.pMin.x = node.planes[5];
            } else if (axis2 == 1) {
                boundingBox21.pMax.y = node.planes[4];
                boundingBox22.pMin.y = node.planes[5];
            } else if (axis2 == 2) {
                boundingBox21.pMax.z = node.planes[4];
                boundingBox22.pMin.z = node.planes[5];
            }
            float S = parentBoundingBox.SurfaceArea();
            SAH += TRAVERSAL_COST_WDST_MEDIUM_SN +
                   (boundingBox1.SurfaceArea() / S) * calculateOverallSAH(*node.children[0], boundingBox1, depth + 1) +
                   (boundingBox21.SurfaceArea() / S) * calculateOverallSAH(*node.children[2], boundingBox21, depth + 1) +
                   (boundingBox22.SurfaceArea() / S) * calculateOverallSAH(*node.children[3], boundingBox22, depth + 1);
        } else if ((header & 3) == 2) {
            int axis1 = node.axes & 3;
            Bounds3f boundingBox11(boundingBox1);
            Bounds3f boundingBox12(boundingBox1);
            if (axis1 == 0) {
                boundingBox11.pMax.x = node.planes[2];
                boundingBox12.pMin.x = node.planes[3];
            } else if (axis1 == 1) {
                boundingBox11.pMax.y = node.planes[2];
                boundingBox12.pMin.y = node.planes[3];
            } else if (axis1 == 2) {
                boundingBox11.pMax.z = node.planes[2];
                boundingBox12.pMin.z = node.planes[3];
            }
            float S = parentBoundingBox.SurfaceArea();
            SAH += TRAVERSAL_COST_WDST_MEDIUM_SN +
                   (boundingBox11.SurfaceArea() / S) * calculateOverallSAH(*node.children[0], boundingBox11, depth + 1) +
                   (boundingBox12.SurfaceArea() / S) * calculateOverallSAH(*node.children[1], boundingBox12, depth + 1) +
                   (boundingBox2.SurfaceArea() / S) * calculateOverallSAH(*node.children[2], boundingBox2, depth + 1);
        } else if ((header & 3) == 3) {
            int axis1 = (node.axes >> 2) & 3;
            Bounds3f boundingBox11(boundingBox1);
            Bounds3f boundingBox12(boundingBox1);
            if (axis1 == 0) {
                boundingBox11.pMax.x = node.planes[2];
                boundingBox12.pMin.x = node.planes[3];
            } else if (axis1 == 1) {
                boundingBox11.pMax.y = node.planes[2];
                boundingBox12.pMin.y = node.planes[3];
            } else if (axis1 == 2) {
                boundingBox11.pMax.z = node.planes[2];
                boundingBox12.pMin.z = node.planes[3];
            }
            int axis2 = (node.axes) & 3;
            Bounds3f boundingBox21(boundingBox2);
            Bounds3f boundingBox22(boundingBox2);
            if (axis2 == 0) {
                boundingBox21.pMax.x = node.planes[4];
                boundingBox22.pMin.x = node.planes[5];
            } else if (axis2 == 1) {
                boundingBox21.pMax.y = node.planes[4];
                boundingBox22.pMin.y = node.planes[5];
            } else if (axis2 == 2) {
                boundingBox21.pMax.z = node.planes[4];
                boundingBox22.pMin.z = node.planes[5];
            }
            float S = parentBoundingBox.SurfaceArea();
            SAH += TRAVERSAL_COST_WDST_BIG_SN +
                   (boundingBox11.SurfaceArea() / S) * calculateOverallSAH(*node.children[0], boundingBox11, depth + 1) +
                   (boundingBox12.SurfaceArea() / S) * calculateOverallSAH(*node.children[1], boundingBox12, depth + 1) +
                   (boundingBox21.SurfaceArea() / S) * calculateOverallSAH(*node.children[2], boundingBox21, depth + 1) +
                   (boundingBox22.SurfaceArea() / S) * calculateOverallSAH(*node.children[3], boundingBox22, depth + 1);
        }
    }
    return SAH;
}

WDSTAggregate *WDSTAggregate::Create(std::vector<Primitive> prims,
                                     const ParameterDictionary &parameters) {
    pstd::pmr::monotonic_buffer_resource resource;
    Allocator alloc(&resource);
    DSTAggregate dst = *DSTAggregate::Create(prims, parameters);
    Bounds3f globalBB = dst.globalBB;
    std::vector<DSTBuildNode> DSTNodesBeforReconstruction = gloabalDSTNodes;
    reconstructTree(0);
    std::vector<DSTBuildNode> DSTNodesAfterReconstruction = gloabalDSTNodes;
    DSTBuildNode node = gloabalDSTNodes[0];
    return new WDSTAggregate(std::move(dst.primitives), node, globalBB);
}

Bounds3f WDSTAggregate::Bounds() const {
    return globalBB;
}

pstd::optional<ShapeIntersection> WDSTAggregate::Intersect(const Ray &ray,
                                                           Float globalTMax) const {
    float delta = 0.001;
    pstd::optional<ShapeIntersection> si;
    float tMin = 0;
    float tMax = globalTMax;

    unsigned headerOffset;
    unsigned header5;
    bool leaf;
    unsigned idx = 0;

    const Vector3f invDir = {1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z};
    const int dirSgn[3] = {invDir[0] < 0, invDir[1] < 0, invDir[2] < 0};
    if (!globalBB.IntersectP(ray.o, ray.d, tMax, invDir, dirSgn))
        return {};
    wdstRays++;
    wdstRays1++;
    if (tMin < 0)
        tMin = 0;

    StackItem *stack = new StackItem[(maximumDepth + 1) * 3 ];
    int stackPtr = -1;

    while (true) {
        headerOffset = linearWDST[idx];
        header5 = headerOffset >> 27;
        leaf = (headerOffset >> 26) & 1;

        if (header5 >> 4 == 0) {
            if (leaf) { 
                idx = headerOffset & OFFSET;
                goto leaf;
            }
            int amountOfChildren = header5 >> 2;
            int whereDoubleChildren = header5 & 0b11;
            int axes[] = {-1, -1, -1};
            int childrenSize[] = {-1, -1, -1};
            for (int i = 0; i < amountOfChildren; i++) {
                axes[i] = (headerOffset >> (24 - 2 * i)) & 0b11;
                childrenSize[i] = ((headerOffset >> (17 - 3 * i)) & 0b111) + 1;
            }

            //amountOfChildren is one smaller than the actual amount of children for this SN, but this way it is half of the amount of planes stored in this SN
            std::vector<float> splitPlanes(amountOfChildren * 2);
            for (int i = 0; i < amountOfChildren * 2; i += 2) {
                uint32_t temp1 = linearWDST[idx + 2 + i];
                splitPlanes[i] = *reinterpret_cast<float *>(&temp1);
                uint32_t temp2 = linearWDST[idx + 3 + i];
                splitPlanes[i + 1] = *reinterpret_cast<float *>(&temp2);
            }
            idx = linearWDST[idx + 1];

            unsigned sign = dirSgn[axes[0]];
            float ts1 = (splitPlanes[sign] - ray.o[axes[0]]) * invDir[axes[0]];
            float ts2 = (splitPlanes[sign ^ 1] - ray.o[axes[0]]) * invDir[axes[0]];

            wdstPlaneIntersections += 2;
            if (tMax + delta >= ts2) {
                float tNext = std::max(tMin, ts2);
                if (tMin <= ts1 + delta) {
                    std::list<StackItem> intersectedChildren;
                    float localTMax = std::min(tMax, ts1);
                    if ((whereDoubleChildren >> (sign ^ 1)) & 0b1) { //if near area after first split is split again
                        unsigned axis = axes[1 + sign * (amountOfChildren == 3)];
                        unsigned localSign = dirSgn[axis];
                        int previousChildren = sign * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + localSign + 2 * sign * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        float ts4 = (splitPlanes[2 + (localSign ^ 1) + 2 * sign * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (localTMax + delta >= ts4) {
                            float localTNext = std::max(tMin, ts4);
                            if (tMin <= ts3 + delta) {
                                int localIdx1 = idx;
                                int childIndex1 = previousChildren + localSign;
                                for (int i = 0; i < childIndex1; i++)
                                    localIdx1 += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx1, tMin, std::min(localTMax, ts3)));
                                int localIdx2 = idx;
                                int childIndex2 = previousChildren + (localSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    localIdx2 += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx2, localTNext, localTMax));
                            } else {
                                int localIdx = idx;
                                int childIndex = previousChildren + (localSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx, localTNext, localTMax));
                            }
                        } else {
                            if (tMin <= ts3 + delta) {
                                int localIdx = idx;
                                int childIndex = previousChildren + localSign;
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx, tMin, std::min(localTMax, ts3)));
                            }
                        }
                    } else {
                        int localIdx = idx;
                        int childIndex = sign * ((whereDoubleChildren >> 1) + 1);
                        for (int i = 0; i < childIndex; i++)
                            localIdx += childrenSize[i];
                        intersectedChildren.push_back(StackItem(localIdx, tMin, std::min(tMax, ts1)));
                    }
                    float localTMin = std::max(tMin, ts2);
                    if ((whereDoubleChildren >> sign) & 0b1) {
                        unsigned axis = axes[1 + (sign ^ 1) * (amountOfChildren == 3)];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren = (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign + 2 * (sign ^ 1) * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) + 2 * (sign ^ 1) * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax + delta >= ts4) {
                            float localTNext = std::max(localTMin, ts4);
                            if (localTMin <= ts3 + delta) {
                                int localIdx1 = idx;
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    localIdx1 += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx1, localTMin, std::min(tMax, ts3)));
                                int localIdx2 = idx;
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    localIdx2 += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx2, localTNext, tMax));
                            } else {
                                int localIdx = idx;
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx, localTNext, tMax));
                            }
                        } else {
                            if (localTMin <= ts3 + delta) {
                                int localIdx = idx;
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(StackItem(localIdx, localTMin, std::min(tMax, ts3)));
                            }
                        }
                    } else {
                        int localIdx = idx;
                        int childIndex = (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        for (int i = 0; i < childIndex; i++)
                            localIdx += childrenSize[i];
                        intersectedChildren.push_back(StackItem(localIdx, tNext, tMax));
                    }
                    while (!intersectedChildren.empty()) {
                            stack[++stackPtr] = intersectedChildren.back();
                            intersectedChildren.pop_back();
                    }
                    goto pop;
                } else {
                    tMin = tNext;
                    if ((whereDoubleChildren >> sign) & 0b1) {
                        unsigned axis = axes[1 + sign * (amountOfChildren == 3)];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren = (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign + 2 * (sign ^ 1) * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) + 2 * (sign ^ 1) * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax + delta >= ts4) {
                            float localTNext = std::max(tMin, ts4);
                            if (tMin <= ts3 + delta) {
                                int farChildIdx = idx;
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    idx += childrenSize[i];
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    farChildIdx += childrenSize[i];
                                stack[++stackPtr] = StackItem(farChildIdx, localTNext, tMax);
                                tMax = std::min(tMax, ts3);
                            } else {
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMin = localTNext;
                            }
                            continue;
                        } else {
                            if (tMin <= ts3 + delta) {
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMax = std::min(tMax, ts3);
                                continue;
                            } else {
                                goto pop;
                            }
                        }
                    } else {
                        int childIndex = (sign ^ 1) * amountOfChildren;
                        for (int i = 0; i < childIndex; i++)
                            idx += childrenSize[i];
                        tMin = tNext;
                    }
                }
                continue;
            } else {
                if (tMin <= ts1 + delta) {
                    tMax = std::min(tMax, ts1);
                    if ((whereDoubleChildren >> (sign ^ 1)) & 0b1) {
                        unsigned axis = axes[1 + (sign ^ 1) * (amountOfChildren == 3)];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren = sign * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign + 2 * sign * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) + 2 * sign * (amountOfChildren == 3)] - ray.o[axis]) * invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax + delta >= ts4) {
                            float internalTNext = std::max(tMin, ts4);
                            if (tMin <= ts3 + delta) {
                                int otherIdx = idx;
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    idx += childrenSize[i];
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    otherIdx += childrenSize[i];
                                stack[++stackPtr] = StackItem(otherIdx, internalTNext, tMax);
                                tMax = std::min(tMax, ts3);
                            } else {
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMin = internalTNext;
                            }
                            continue;
                        } else {
                            if (tMin <= ts3 + delta) {
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMax = std::min(tMax, ts3);
                                continue;
                            } else {
                                goto pop;
                            }
                        }
                    } else {
                        int childIndex = sign * amountOfChildren;
                        for (int i = 0; i < childIndex; i++)
                            idx += childrenSize[i];
                        tMax = std::min(tMax, ts1);
                    }
                    continue;
                } else {
                    goto pop;
                }
            }
        } else { //Hier nur CN oder CL
            float carvePlanes[2];
            uint32_t temp1 = linearWDST[idx + 1];
            carvePlanes[0] = *reinterpret_cast<float *>(&temp1);
            uint32_t temp2 = linearWDST[idx + 2];
            carvePlanes[1] = *reinterpret_cast<float *>(&temp2);

            char carveType1 = (header5 >> 2) & 3;
            char carveType2 = header5 & 3;

            if (carveType1 == 2) {
                unsigned sign = dirSgn[carveType2];
                if (carvePlanes[sign] != 0) {
                    float ts1 = (carvePlanes[sign] - ray.o[carveType2]) * invDir[carveType2];
                    tMax = std::min(ts1, tMax);
                }
                if (carvePlanes[sign ^ 1] != 0) {
                    float ts2 = (carvePlanes[sign ^ 1] - ray.o[carveType2]) * invDir[carveType2];
                    tMin = std::max(ts2, tMin);
                }
                wdstPlaneIntersections++;
                if (tMin > tMax + delta)
                    goto pop;
                int offset = headerOffset & OFFSET;

                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            } else {
                unsigned axis1 = carveType1 >> 1;
                unsigned axis2 = (carveType1 & 1) + 1;

                float tMin0, tMin1, tMax0, tMax1;
                tMin0 = (carvePlanes[0] - ray.o[axis1]) * invDir[axis1];
                tMax0 = tMax;
                if (dirSgn[axis1] == (carveType2 >> 1)) {
                    tMax0 = tMin0;
                    tMin0 = tMin;
                }
                tMin1 = (carvePlanes[1] - ray.o[axis2]) * invDir[axis2];
                tMax1 = tMax;
                if (dirSgn[axis2] == (carveType2 & 1)) {
                    tMax1 = tMin1;
                    tMin1 = tMin;
                }

                tMin = std::max(tMin, std::max(tMin0, tMin1));
                tMax = std::min(tMax, std::min(tMax0, tMax1));
                wdstPlaneIntersections++;
                if (tMin > tMax + delta)
                    goto pop;
                int offset = headerOffset & OFFSET;
                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            }
        }

    leaf:
        for (unsigned i = idx;; i++) {
            wdstTriangleIntersections++;
            pstd::optional<ShapeIntersection> primSi = primitives[i].Intersect(ray, tMax + delta);
            if (primSi.has_value()) {
                si = primSi;
                tMax = si->tHit;
            }
            if (primitives[i].isLast()) {
                break;
            }
        }
    pop:
        while (true) {
            if (stackPtr == -1) {
                return si;
            }
            StackItem item = stack[stackPtr--];
            idx = item.idx;
            tMin = item.tMin;
            if (!si.has_value()) {
                tMax = item.tMax;
                break;
            } else if (tMin < si->tHit) {
                tMax = std::min(si->tHit, item.tMax);
                break;
            }
        }
    }
    return si;
}

bool WDSTAggregate::IntersectP(const Ray &ray, Float globalTMax) const {
    wdstRays++;
    float tMin = 0;
    float tMax = globalTMax;

    unsigned headerOffset;
    unsigned header5;
    bool leaf;
    unsigned idx = 0;

    const Vector3f invDir = {1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z};
    const int dirSgn[3] = {invDir[0] < 0, invDir[1] < 0, invDir[2] < 0};
    if (globalBB.IntersectP(ray.o, ray.d, tMax, invDir, dirSgn))
        return {};
    if (tMin < 0)
        tMin = 0;

    StackItem *stack = new StackItem[maximumDepth];
    int stackPtr = -1;

    while (true) {
        headerOffset = linearWDST[idx];
        header5 = headerOffset >> 27;
        leaf = (headerOffset >> 26) & 1;

        if (header5 >> 4 == 0) {
            if (leaf) {
                idx = headerOffset & OFFSET;
                goto leaf;
            }
            int amountOfChildren = header5 >> 2;
            int whereDoubleChildren = header5 & 0b11;
            int axes[] = {-1, -1, -1};
            int childrenSize[] = {-1, -1, -1};
            for (int i = 0; i < amountOfChildren; i++) {
                axes[i] = (headerOffset >> (24 - 2 * i)) & 0b11;
                childrenSize[i] = (headerOffset >> (17 - 3 * i)) & 0b111;
            }

            headerOffset = linearWDST[idx + 1];
            // amountOfChildren is one smaller than the actual amount of children for this
            // SN, but this way it is half of the amount of planes stored in this SN
            std::vector<float> splitPlanes(amountOfChildren * 2);
            for (int i = 0; i < amountOfChildren; i += 2) {
                splitPlanes[i] = *reinterpret_cast<float *>(linearWDST[idx + 1 + i]);
                splitPlanes[i + 1] = *reinterpret_cast<float *>(linearWDST[idx + 2 + i]);
            }
            idx = headerOffset & OFFSET;

            unsigned sign = dirSgn[axes[0]];
            float ts1 = (splitPlanes[sign] - ray.o[axes[0]]) * invDir[axes[0]];
            float ts2 = (splitPlanes[sign ^ 1] - ray.o[axes[0]]) * invDir[axes[0]];

            wdstPlaneIntersections += 2;
            if (tMax >= ts2) {
                float tNext = std::max(tMin, ts2);
                if (tMin <= ts1) {
                    std::list<StackItem> intersectedChildren;
                    float localTMax = std::min(tMax, ts2);
                    if ((whereDoubleChildren >> (sign ^ 1)) & 0b1) {
                        unsigned axis = axes[1 + sign];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren = sign * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign +
                                                 sign * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) +
                                                 sign * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (localTMax >= ts4) {
                            float localTNext = std::max(tMin, ts4);
                            if (tMin <= ts3) {
                                int localIdx1 = idx;
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    localIdx1 += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx1, tMin, std::min(localTMax, ts3)));
                                int localIdx2 = idx;
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    localIdx2 += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx2, localTNext, localTMax));
                            } else {
                                int localIdx = idx;
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx, localTNext, localTMax));
                            }
                        } else {
                            if (tMin <= ts3) {
                                int localIdx = idx;
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx, tMin, std::min(localTMax, ts3)));
                            }
                        }
                    } else {
                        int localIdx = idx;
                        int childIndex = sign * ((whereDoubleChildren >> 1) + 1);
                        for (int i = 0; i < childIndex; i++)
                            localIdx += childrenSize[i];
                        intersectedChildren.push_back(
                            StackItem(localIdx, tMin, std::min(tMax, ts1)));
                    }
                    float localTMin = std::max(tMin, ts2);
                    if ((whereDoubleChildren >> sign) & 0b1) {
                        unsigned axis = axes[1 + (sign ^ 1)];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren =
                            (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign +
                                                 (sign ^ 1) * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) +
                                                 (sign ^ 1) * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax >= ts4) {
                            float localTNext = std::max(localTMin, ts4);
                            if (tMin <= ts3) {
                                int localIdx1 = idx;
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    localIdx1 += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx1, localTMin, std::min(tMax, ts3)));
                                int localIdx2 = idx;
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    localIdx2 += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx2, localTNext, tMax));
                            } else {
                                int localIdx = idx;
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx, localTNext, tMax));
                            }
                        } else {
                            if (localTMin <= ts3) {
                                int localIdx = idx;
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    localIdx += childrenSize[i];
                                intersectedChildren.push_back(
                                    StackItem(localIdx, localTMin, std::min(tMax, ts3)));
                            }
                        }
                    } else {
                        int localIdx = idx;
                        int childIndex = (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        for (int i = 0; i < childIndex; i++)
                            localIdx += childrenSize[i];
                        intersectedChildren.push_back(StackItem(localIdx, tNext, tMax));
                    }

                    while (!intersectedChildren.empty()) {
                        stack[++stackPtr] = intersectedChildren.back();
                        intersectedChildren.pop_back();
                    }
                    goto pop;
                } else {
                    tMin = tNext;
                    if ((whereDoubleChildren >> sign) & 0b1) {
                        unsigned axis = axes[1 + sign];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren =
                            (sign ^ 1) * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign +
                                                 (sign ^ 1) * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) +
                                                 (sign ^ 1) * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax >= ts4) {
                            float internalTNext = std::max(tMin, ts4);
                            if (tMin <= ts3) {
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    idx += childrenSize[i];
                                int otherIdx = idx;
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    otherIdx += childrenSize[i];
                                stack[++stackPtr] =
                                    StackItem(otherIdx, internalTNext, tMax);
                                tMax = std::min(tMax, ts3);
                            } else {
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMin = internalTNext;
                            }
                            continue;
                        } else {
                            if (tMin <= ts3) {
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMax = std::min(tMax, ts4);
                                continue;
                            } else {
                                goto pop;
                            }
                        }
                    } else {
                        int childIndex = (sign ^ 1) * amountOfChildren;
                        for (int i = 0; i < childIndex; i++)
                            idx += childrenSize[i];
                        tMin = tNext;
                    }
                }
                continue;
            } else {
                if (tMin <= ts1) {
                    tMax = std::min(tMax, ts1);
                    if ((whereDoubleChildren >> (sign ^ 1)) & 0b1) {
                        unsigned axis = axes[1 + (sign ^ 1)];
                        unsigned internalSign = dirSgn[axis];
                        int previousChildren = sign * ((whereDoubleChildren >> 1) + 1);
                        float ts3 = (splitPlanes[2 + internalSign +
                                                 sign * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        float ts4 = (splitPlanes[2 + (internalSign ^ 1) +
                                                 sign * (amountOfChildren == 4)] -
                                     ray.o[axis]) *
                                    invDir[axis];
                        wdstPlaneIntersections += 2;
                        if (tMax >= ts4) {
                            float internalTNext = std::max(tMin, ts4);
                            if (tMin <= ts3) {
                                int childIndex1 = previousChildren + internalSign;
                                for (int i = 0; i < childIndex1; i++)
                                    idx += childrenSize[i];
                                int otherIdx = idx;
                                int childIndex2 = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex2; i++)
                                    otherIdx += childrenSize[i];
                                stack[++stackPtr] =
                                    StackItem(otherIdx, internalTNext, tMax);
                                tMax = std::min(tMax, ts3);
                            } else {
                                int childIndex = previousChildren + (internalSign ^ 1);
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMin = internalTNext;
                            }
                            continue;
                        } else {
                            if (tMin <= ts3) {
                                int childIndex = previousChildren + internalSign;
                                for (int i = 0; i < childIndex; i++)
                                    idx += childrenSize[i];
                                tMax = std::min(tMax, ts4);
                                continue;
                            } else {
                                goto pop;
                            }
                        }
                    } else {
                        int childIndex = sign * amountOfChildren;
                        for (int i = 0; i < childIndex; i++)
                            idx += childrenSize[i];
                        tMax = std::min(tMax, ts1);
                    }
                    continue;
                } else {
                    goto pop;
                }
            }
        } else {  // Hier nur CN oder CL
            float carvePlanes[2];
            carvePlanes[0] = *reinterpret_cast<float *>(linearWDST[idx + 1]);
            carvePlanes[1] = *reinterpret_cast<float *>(linearWDST[idx + 2]);

            char carveType1 = header5 >> 2 & 3;
            char carveType2 = header5 & 3;

            if (carveType1 == 2) {
                unsigned sign = dirSgn[carveType2];
                float ts1 = (carvePlanes[sign] - ray.o[carveType2]) * invDir[carveType2];
                float ts2 =
                    (carvePlanes[sign ^ 1] - ray.o[carveType2]) * invDir[carveType2];
                tMax = std::min(ts1, tMax);
                tMin = std::max(ts2, tMin);
                wdstPlaneIntersections++;
                if (tMin > tMax)
                    goto pop;
                int offset = headerOffset & OFFSET;

                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            } else {
                unsigned axis1 = carveType1 >> 1;
                unsigned axis2 = (carveType1 & 1) + 1;

                float tMin0, tMin1, tMax0, tMax1;
                tMin0 = (carvePlanes[0] - ray.o[axis1]) * invDir[axis1];
                tMax0 = tMax;
                if (dirSgn[axis1] == (carveType2 >> 1)) {
                    tMax0 = tMin0;
                    tMin0 = tMin;
                }
                tMin1 = (carvePlanes[1] - ray.o[axis2]) * invDir[axis2];
                tMax1 = tMax;
                if (dirSgn[axis2] == (carveType2 & 1)) {
                    tMax1 = tMin1;
                    tMin1 = tMin;
                }

                tMin = std::max(tMin, std::max(tMin0, tMin1));
                tMax = std::min(tMax, std::min(tMax0, tMax1));
                wdstPlaneIntersections++;
                if (tMin > tMax)
                    goto pop;
                int offset = headerOffset & OFFSET;
                if (leaf) {
                    idx = offset;
                    goto leaf;
                }
                idx = offset;
                continue;
            }
        }

    leaf:
        for (unsigned i = idx;; i++) {
            wdstTriangleIntersections++;
            if (primitives[i].Intersect(ray, tMax))
                return true;
            // Test if last Triangle in node
            if (primitives[i].isLast()) {
                break;
            }
        }
    pop:
        while (true) {
            if (stackPtr == -1) {
                return false;
            }
            StackItem item = stack[stackPtr--];
            idx = item.idx;
            tMin = item.tMin;
            tMax = std::min(globalTMax, item.tMax);
            break;
        }
    }
    return false;
}

int WDSTAggregate::reconstructTree(int index) {
    int childIndex = index;
    if (gloabalDSTNodes[index].IsSplitting()) {
        (&gloabalDSTNodes[index])->children[0] = &gloabalDSTNodes[index + 1];
        childIndex = reconstructTree(index + 1);
        (&gloabalDSTNodes[index])->children[1] = &gloabalDSTNodes[childIndex + 1];
        childIndex = reconstructTree(childIndex + 1);
    } else if (gloabalDSTNodes[index].IsCarving() && !gloabalDSTNodes[index].isCarvingLeaf()) {
        (&gloabalDSTNodes[index])->children[0] = &gloabalDSTNodes[index + 1];
        childIndex = reconstructTree(index + 1);
    }
    return childIndex;
}

WDSTBuildNode *WDSTAggregate::BuildWDSTRecursively(
    ThreadLocal<Allocator> &threadAllocators, DSTBuildNode *splittingNode, float S, int currentDepth) {
     Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *node = alloc.new_object<WDSTBuildNode>();

    if (splittingNode->IsLeaf()) {
        node->InitLeaf(splittingNode->offsetToFirstChild(), currentDepth, splittingNode->NTriangles());
        if (currentDepth > maximumDepth)
            maximumDepth++;
        return node;
    }

    node->planes[0] = splittingNode->plane1;
    node->planes[1] = splittingNode->plane2;
    node->nPrimitives = 0;
    node->axes = splittingNode->GetSplittingPlanesAxis();
    node->boundingBox = splittingNode->GetBB();
    node->depthLevel = currentDepth;
    node->header;

    if (node->offset == 57 || node->offset == 131)
        int i = 0;

    DSTBuildNode *firstUnderlyingSN = getNextRelevantNode(splittingNode->children[0]);
    DSTBuildNode *secondUnderlyingSN = getNextRelevantNode(splittingNode->children[1]);
    std::vector<float> irrelevantPlanes = {0.f, 0.f};
    Bounds3f leftBBAfterFirstSplit = carve(splittingNode->GetBB(), firstUnderlyingSN->GetBB(),
                         {splittingNode->GetSplittingPlanesAxis() + 4}, irrelevantPlanes);
    Bounds3f rightBBAfterFirstSplit = carve(splittingNode->GetBB(), secondUnderlyingSN->GetBB(),
                         {splittingNode->GetSplittingPlanesAxis() + 1}, irrelevantPlanes);
    
    if (firstUnderlyingSN->IsSplitting()) {
        node->planes[2] = firstUnderlyingSN->plane1;
        node->planes[3] = firstUnderlyingSN->plane2;
        node->axes = node->axes << 2;
        node->axes += firstUnderlyingSN->GetSplittingPlanesAxis();

        Bounds3f doubleSplitBB1 = carve(leftBBAfterFirstSplit, firstUnderlyingSN->children[0]->GetBB(),
                  {firstUnderlyingSN->GetSplittingPlanesAxis() + 4}, irrelevantPlanes);
        Bounds3f doubleSplitBB2 = carve(leftBBAfterFirstSplit, firstUnderlyingSN->children[1]->GetBB(),
                  {firstUnderlyingSN->GetSplittingPlanesAxis() + 1}, irrelevantPlanes);

        DSTBuildNode *nextNode = getNextRelevantNode(firstUnderlyingSN->children[0]); 
        float SAH = 0;
        WDSTBuildNode *child1 = determineCNConstelation(
            threadAllocators, doubleSplitBB1, nextNode, S,
            currentDepth + 1, &SAH);
        WDSTBuildNode *lowerEnd = getLowerEnd(child1);
        if (child1 == NULL) { //If there is a SN directly after the parent SN in the WDST
            node->children[0] = BuildWDSTRecursively(threadAllocators, nextNode, firstUnderlyingSN->GetBB().SurfaceArea(), currentDepth + 1);
        } else if (lowerEnd == NULL || lowerEnd->IsLeaf()) { //If there is no more SN in the subtree of child1 of the parent SN
            node->children[0] = child1;
        } else { //If there are one to three CN between the SN and the parent SN in the WDST
            lowerEnd->children[0] = BuildWDSTRecursively(threadAllocators, nextNode, lowerEnd->boundingBox.SurfaceArea(), lowerEnd->depthLevel + 1);
            node->children[0] = child1;
        }

        nextNode = getNextRelevantNode(firstUnderlyingSN->children[1]);
        WDSTBuildNode *child2 = determineCNConstelation(
            threadAllocators, doubleSplitBB2, nextNode, S,
            currentDepth + 1, &SAH);
        lowerEnd = getLowerEnd(child2);
        if (child2 == NULL) {  
            node->children[1] = BuildWDSTRecursively(
                threadAllocators, nextNode, firstUnderlyingSN->GetBB().SurfaceArea(),
                currentDepth + 1);
        } else if (lowerEnd == NULL || lowerEnd->IsLeaf()) {
            node->children[1] = child2;
        } else {
            lowerEnd->children[0] = BuildWDSTRecursively(
                threadAllocators, nextNode, lowerEnd->boundingBox.SurfaceArea(),
                lowerEnd->depthLevel + 1);
            node->children[1] = child2;
        }
    } else {
        WDSTBuildNode *child = transformDSTNode(
            threadAllocators, *splittingNode->children[0], currentDepth + 1);
        node->children[0] = child;
    }

    if (secondUnderlyingSN->IsSplitting()) {
        node->planes[4] = secondUnderlyingSN->plane1;
        node->planes[5] = secondUnderlyingSN->plane2;
        node->axes = node->axes << 2;
        node->axes += secondUnderlyingSN->GetSplittingPlanesAxis();

        Bounds3f doubleSplitBB3 = carve(rightBBAfterFirstSplit, secondUnderlyingSN->children[0]->GetBB(),
                  {secondUnderlyingSN->GetSplittingPlanesAxis() + 4}, irrelevantPlanes);
        Bounds3f doubleSplitBB4 = carve(rightBBAfterFirstSplit, secondUnderlyingSN->children[1]->GetBB(),
                  {secondUnderlyingSN->GetSplittingPlanesAxis() + 1}, irrelevantPlanes);

        DSTBuildNode *nextNode = getNextRelevantNode(secondUnderlyingSN->children[0]);
        float SAH = 0;
        WDSTBuildNode *child1 = determineCNConstelation(
            threadAllocators, doubleSplitBB3, nextNode, S,
            currentDepth + 1, &SAH);
        WDSTBuildNode *lowerEnd = getLowerEnd(child1);
        if (child1 == NULL) {
            node->children[2] = BuildWDSTRecursively(
                threadAllocators, nextNode, firstUnderlyingSN->GetBB().SurfaceArea(),
                currentDepth + 1);
        } else if (lowerEnd == NULL || lowerEnd->IsLeaf()) {
            node->children[2] = child1;
        } else {
            lowerEnd->children[0] = BuildWDSTRecursively(
                threadAllocators, nextNode, lowerEnd->boundingBox.SurfaceArea(),
                lowerEnd->depthLevel + 1);
            node->children[2] = child1;
        }

        nextNode = getNextRelevantNode(secondUnderlyingSN->children[1]);
        WDSTBuildNode *child2 = determineCNConstelation(
            threadAllocators, doubleSplitBB4, nextNode, S,
            currentDepth + 1, &SAH);
        lowerEnd = getLowerEnd(child2);
        if (child2 == NULL) {
            node->children[3] = BuildWDSTRecursively(
                threadAllocators, nextNode, firstUnderlyingSN->GetBB().SurfaceArea(),
                currentDepth + 1);
        } else if (lowerEnd == NULL || lowerEnd->IsLeaf()) {
            node->children[3] = child2;
        } else {
            lowerEnd->children[0] = BuildWDSTRecursively(
                threadAllocators, nextNode, lowerEnd->boundingBox.SurfaceArea(),
                lowerEnd->depthLevel + 1);
            node->children[3] = child2;
        }
    } else {
        WDSTBuildNode *child = transformDSTNode(
            threadAllocators, *splittingNode->children[1], currentDepth + 1);
        node->children[2] = child;
    }
    node->header = 0b001000;
    if (node->children[1] != NULL) {
        node->header = node->header + 0b001100;
    }
    if (node->children[3] != NULL) {
        node->header = node->header + 0b001010;
    }
    return node;
}


void WDSTAggregate::FlattenWDST(WDSTBuildNode *node) {
    if (node == NULL)
        return;
    uint32_t offset = node->offset;
    if (node->IsCarving()){
        uint32_t flag = node->header << 26;
        flag |= node->offsetToFirstChild();
        linearWDST[offset] = flag;
        linearWDST[offset + 1] = node->PlaneAsInt(0);
        linearWDST[offset + 2] = node->PlaneAsInt(1);
        if (node->children[0] != NULL)
            FlattenWDST(node->children[0]);
        if (node->IsLeaf())
            numberOfLeafs++;
        else
            numberOfCarvingNodes++;
    } else if (node->IsLeaf()) {
        linearWDST[offset] = node->header << 26 | node->triangleOffset;
        numberOfLeafs++;
    } else {
        int header = 8;
        if (node->children[1] != NULL)
            header += 12;
        if (node->children[3] != NULL)
            header += 10;
        int childrenSize = 0;
        int j = 0;
        for (int i = 0; i < 3 && j < 3; i++) {
            if (node->children[i] != NULL) {
                childrenSize = childrenSize << 3;
                childrenSize += node->children[i]->size();
                j++;
            }
        }
        for (int i = j; i < 3; i++) {
            childrenSize = childrenSize << 3;
        }
        uint32_t axes = node->axes;
        for (int i = (header >> 3); i < 3; i++) {
            axes = axes << 2;
        }
        linearWDST[offset] = (header << 26) + (axes << 20) + (childrenSize << 11);
        linearWDST[offset + 1] = node->children[0]->offset;
        j = 2;
        for (int i = 0; i < 6; i += 2) {
            if (node->planes[i] != 0) {
                linearWDST[offset + j] = node->PlaneAsInt(i);
                linearWDST[offset + j + 1] = node->PlaneAsInt(i + 1);
                j += 2;
            }
        }
        for (int i = 0; i < 4; i++)
            FlattenWDST(node->children[i]);
        if (header >> 3 == 1) {
            smallSplittingNodes++;
        } else if (header >> 3 == 2) {
            mediumSplittingNodes++;
        }
        else {
            bigSplittingNodes++;
        }
        numberOfSplittingNodes++;
    }
}

DSTBuildNode *WDSTAggregate::getNextRelevantNode(DSTBuildNode *node) {
    while (node->IsCarving() && !node->isCarvingLeaf()) {
        node = node->children[0];
    }
    return node;
}

WDSTBuildNode *WDSTAggregate::transformDSTNode(ThreadLocal<Allocator> &threadAllocators,
                                               DSTBuildNode node, int currentDepth) {
    Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *returnNode = alloc.new_object<WDSTBuildNode>();
    WDSTBuildNode *wdstNode = returnNode;
    while (node.IsCarving()  && !node.isCarvingLeaf()) {
        WDSTBuildNode *nextNode = alloc.new_object<WDSTBuildNode>();
        wdstNode->InitExistingCarving(node.plane1, node.plane2, nextNode, node.GetHeader(), 0, node.GetBB(), currentDepth++, node.NTriangles());
        node = *node.children[0];
        wdstNode = nextNode;
    }
    if (node.isCarvingLeaf()) {
        wdstNode->InitExistingCarving(node.plane1, node.plane2, NULL, node.GetHeader(), node.offsetToFirstChild(), node.GetBB(), currentDepth++, node.NTriangles());
    } else {
        wdstNode->InitLeaf(node.offsetToFirstChild(), currentDepth, node.NTriangles());
    }
    if (currentDepth > maximumDepth)
        maximumDepth++;
    return returnNode;
}

WDSTBuildNode *WDSTAggregate::determineCNConstelation(
    ThreadLocal<Allocator> &threadAllocators, Bounds3f parentBB,
    DSTBuildNode *nextRelevantDSTNode, float S, int currentDepth, float *SAH) {
    Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *node = alloc.new_object<WDSTBuildNode>();
    numberOfCarvingNodeSets++;
    float delta = 0.0001;

    Bounds3f childBB = nextRelevantDSTNode->GetBB();
    std::vector<int> sidesToCarve;
    if (parentBB.pMin.x + delta < childBB.pMin.x)
        sidesToCarve.push_back(1);
    if (parentBB.pMax.x > childBB.pMax.x + delta)
        sidesToCarve.push_back(4);
    if (parentBB.pMin.y + delta < childBB.pMin.y)
        sidesToCarve.push_back(2);
    if (parentBB.pMax.y > childBB.pMax.y + delta)
        sidesToCarve.push_back(5);
    if (parentBB.pMin.z + delta < childBB.pMin.z)
        sidesToCarve.push_back(3);
    if (parentBB.pMax.z > childBB.pMax.z + delta)
        sidesToCarve.push_back(6);
    if (nextRelevantDSTNode->isCarvingLeaf()) {
        std::vector<int> carvedSides = nextRelevantDSTNode->GetCarvedSides();
        if (carvedSides.size() > 0 && !(std::find(sidesToCarve.begin(), sidesToCarve.end(),
                                                 carvedSides[0]) !=
              sidesToCarve.end()))
            sidesToCarve.push_back(carvedSides[0]);
        if (carvedSides.size() > 1 && (std::find(sidesToCarve.begin(), sidesToCarve.end(),
                                                carvedSides[1]) !=
              sidesToCarve.end()))
            sidesToCarve.push_back(carvedSides[1]);
    }
    float localSAH = 0;
    if (sidesToCarve.size() == 5) {
        node = getThreeCarvingNodes(threadAllocators, sidesToCarve, parentBB,
                                    nextRelevantDSTNode, &localSAH, S, currentDepth);
        setsWithThree++;
    } else if (sidesToCarve.size() > 2) {
        node = getTwoCarvingNodes(threadAllocators, sidesToCarve, parentBB,
                                  nextRelevantDSTNode, &localSAH, S, currentDepth);
        setsWithTwo++;
    } else if (sidesToCarve.size() > 0) {
        node = getOneCarvingNode(threadAllocators, sidesToCarve, parentBB,
                                 nextRelevantDSTNode, &localSAH, S, currentDepth);
        setsWithOne++;
    } else {
        setsWithZero++;
        if (nextRelevantDSTNode->IsLeaf())
            return transformDSTNode(threadAllocators, *nextRelevantDSTNode, currentDepth);
        else {
            return NULL;
        }
    }
    *SAH = localSAH;
    return node;
}

WDSTBuildNode *WDSTAggregate::getThreeCarvingNodes(
    ThreadLocal<Allocator> &threadAllocators,
                                    std::vector<int> sidesToCarve, Bounds3f parentBB,
                                    DSTBuildNode *nextRelevantDSTNode, float *globalSAH,
                                    float S, int currentDepth) {
    Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *bestNode = alloc.new_object<WDSTBuildNode>();
    float bestSAH = FLT_MAX;

    for (int i = 0; i < sidesToCarve.size(); i++) {
        for (int j = i + 1; j < sidesToCarve.size(); j++) {
            WDSTBuildNode *node = alloc.new_object<WDSTBuildNode>();
            float SAH = 0.f;
            std::vector<float> planes = {0.f, 0.f};
            int firstSideToCarve = sidesToCarve[i];
            int secondSideToCarve = sidesToCarve[j];
            std::vector<int> carvingSides = {firstSideToCarve, secondSideToCarve};
            Bounds3f carvedParentBB =
                carve(parentBB, nextRelevantDSTNode->GetBB(), carvingSides, planes);
            std::vector<int> sidesToCarveCopy(sidesToCarve);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + j);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + i);
            int header;
            if (secondSideToCarve < firstSideToCarve) {
                firstSideToCarve = sidesToCarve[j];
                secondSideToCarve = sidesToCarve[i];
            }
            if ((firstSideToCarve - 1) % 3 > (secondSideToCarve - 1) % 3) {
                int temp = firstSideToCarve;
                firstSideToCarve = secondSideToCarve;
                secondSideToCarve = temp;
            }
            if (secondSideToCarve - firstSideToCarve == 3) {
                header = 0b110000 | (firstSideToCarve - 1) << 1;
                SAH += (parentBB.SurfaceArea() / S) * 0.3f;
            } else {
                int carveType1 = ((firstSideToCarve + secondSideToCarve) % 3);
                if (carveType1 == 2)
                    carveType1++;
                header = 0b100000 | (carveType1 << 3);
                header |= (2 * (firstSideToCarve < 4) + (secondSideToCarve < 4)) << 1;
                SAH += (parentBB.SurfaceArea() / S) * 0.5f;
            }

            node->InitNewCarving(planes[0], planes[1], getTwoCarvingNodes(threadAllocators, sidesToCarveCopy, carvedParentBB,
                                   nextRelevantDSTNode, &SAH, S, currentDepth + 1), header, 0, parentBB, currentDepth, 0);
            
            if(SAH < bestSAH) {
                bestSAH = SAH;
                bestNode = node;
            }
        }
    }
    *globalSAH += bestSAH;
    return bestNode;
}

WDSTBuildNode *WDSTAggregate::getTwoCarvingNodes(ThreadLocal<Allocator> &threadAllocators,
                                     std::vector<int> sidesToCarve, Bounds3f parentBB,
                                    DSTBuildNode *nextRelevantDSTNode, float *globalSAH, float S, int currentDepth) {
    Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *bestNode = alloc.new_object<WDSTBuildNode>();
    float bestSAH = FLT_MAX;

    for (int i = 0; i < sidesToCarve.size(); i++) {
        for (int j = i + 1; j < sidesToCarve.size(); j++) {
            WDSTBuildNode *node = alloc.new_object<WDSTBuildNode>();
            float SAH = 0.f;
            std::vector<float> planes = {0.f, 0.f};
            int firstSideToCarve = sidesToCarve[i];
            int secondSideToCarve = sidesToCarve[j];
            std::vector<int> carvingSides = {firstSideToCarve, secondSideToCarve};
            Bounds3f carvedParentBB =
                carve(parentBB, nextRelevantDSTNode->GetBB(), carvingSides, planes);
            std::vector<int> sidesToCarveCopy(sidesToCarve);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + j);
            sidesToCarveCopy.erase(sidesToCarveCopy.begin() + i);
            int header;
            if (secondSideToCarve < firstSideToCarve) {
                firstSideToCarve = sidesToCarve[j];
                secondSideToCarve = sidesToCarve[i];
            }
            if ((firstSideToCarve - 1) % 3 > (secondSideToCarve - 1) % 3) {
                int temp = firstSideToCarve;
                firstSideToCarve = secondSideToCarve;
                secondSideToCarve = temp;
            }
            if (secondSideToCarve - firstSideToCarve == 3) {
                header = 0b110000 | (firstSideToCarve - 1) << 1;
                SAH += (parentBB.SurfaceArea() / S) * 0.3f;
            } else {
                int carveType1 = ((firstSideToCarve + secondSideToCarve) % 3);
                if (carveType1 == 2)
                    carveType1++;
                header = 0b100000 | (carveType1 << 3);
                header |= (2 * (firstSideToCarve < 4) + (secondSideToCarve < 4)) << 1;
                SAH += (parentBB.SurfaceArea() / S) * 0.5f;
            }

            node->InitNewCarving( planes[0], planes[1], getOneCarvingNode(threadAllocators, sidesToCarveCopy, carvedParentBB,
                nextRelevantDSTNode, &SAH, S, currentDepth + 1), header, 0, parentBB, currentDepth, 0);

            if (SAH < bestSAH) {
                bestSAH = SAH;
                bestNode = node;
            }
        }
    }
    *globalSAH += bestSAH;
    return bestNode;
}

WDSTBuildNode *WDSTAggregate::getOneCarvingNode(ThreadLocal<Allocator> &threadAllocators,
                                 std::vector<int> sidesToCarve, Bounds3f parentBB,
                                 DSTBuildNode *nextRelevantDSTNode, float *SAH, float S, int currentDepth) {
    Allocator alloc = threadAllocators.Get();
    WDSTBuildNode *node = alloc.new_object<WDSTBuildNode>();
    std::vector<float> planes = {0.f, 0.f};
    int firstSideToCarve = sidesToCarve[0];
    int secondSideToCarve = 0;    
    if (sidesToCarve.size() > 1) {
        secondSideToCarve = sidesToCarve[1];
        if (secondSideToCarve < firstSideToCarve) {
            firstSideToCarve = sidesToCarve[1];
            secondSideToCarve = sidesToCarve[0];
        }
    }
    Bounds3f carvedParentBB =
        carve(parentBB, nextRelevantDSTNode->GetBB(), sidesToCarve, planes);
    int header;
    bool swapPlanes = false;
    if (secondSideToCarve == 0 || secondSideToCarve - firstSideToCarve == 3) {
        if (secondSideToCarve == 0 && firstSideToCarve > 3)
            swapPlanes = true;
        header = 0b110000 | ((firstSideToCarve - 1) % 3) << 1;
        *SAH += (parentBB.SurfaceArea() / S) * 0.3f + (carvedParentBB.SurfaceArea() / S) * getNTrianglesInSubTree(*nextRelevantDSTNode);
    } else {
        if ((firstSideToCarve - 1) % 3 > (secondSideToCarve - 1) % 3) {
            int temp = firstSideToCarve;
            firstSideToCarve = secondSideToCarve;
            secondSideToCarve = temp;
        }
        int carveType1 = ((firstSideToCarve + secondSideToCarve) % 3);
        if (carveType1 == 2)
            carveType1++;
        header = 0b100000 | (carveType1 << 3);
        header |= (2 * (firstSideToCarve < 4) + (secondSideToCarve < 4)) << 1;
        *SAH += (parentBB.SurfaceArea() / S) * 0.5f + (carvedParentBB.SurfaceArea() / S) * getNTrianglesInSubTree(*nextRelevantDSTNode);
    }
    
    if (swapPlanes) {
        float temp = planes[0];
        planes[0] = planes[1];
        planes[1] = temp;
    }
    if (nextRelevantDSTNode->IsLeaf() || nextRelevantDSTNode->isCarvingLeaf()) {
        header |= 1;
        node->InitNewCarving(planes[0], planes[1], NULL, header, nextRelevantDSTNode->offsetToFirstChild(), parentBB, currentDepth, nextRelevantDSTNode->NTriangles());
        *SAH += (carvedParentBB.SurfaceArea() / S) * nextRelevantDSTNode->NTriangles();
    } else {
        node->InitNewCarving(planes[0], planes[1], NULL, header, 0, parentBB, currentDepth, 0);
    }
    if (currentDepth > maximumDepth)
        maximumDepth++;
    return node;
}

WDSTBuildNode *WDSTAggregate::getLowerEnd(WDSTBuildNode *node) {
    if (node == NULL)
        return NULL;
    while ((*node).children[0] != NULL) {
        node = node->children[0];
    }
    if (node->IsLeaf()) {
        return NULL;
    }
    return node;
}

void WDSTAggregate::addNodeToDepthList(WDSTBuildNode *node) {
    nodesPerDepthLevel[node->GetDepthLevel()].push_back(node);
    for (WDSTBuildNode *child : node->children) {
        if (child != NULL)
            addNodeToDepthList(child);
    }
}
}  // namespace pbrt
