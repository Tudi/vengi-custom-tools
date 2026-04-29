/**
 * @file
 */

#include "FaceClassify.h"
#include "Progress.h"

#include "app/Async.h"
#include "color/RGBA.h"
#include "core/GLM.h"
#include "core/Log.h"
#include "core/TimeProvider.h"
#include "core/collection/DynamicArray.h"
#include "palette/Palette.h"
#include "scenegraph/SceneGraph.h"
#include "scenegraph/SceneGraphNode.h"
#include "scenegraph/SceneGraphUtil.h"
#include "voxedit-util/SceneManager.h"
#include "voxel/Connectivity.h"
#include "voxel/RawVolume.h"
#include "voxel/RawVolumeWrapper.h"
#include "voxel/Region.h"
#include "voxel/Voxel.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <glm/matrix.hpp>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace voxedit {
namespace printing {

namespace {

// Process resident-set size in GB. Reads /proc/self/statm (Linux). Cheap (~1 us).
// Used to log ground-truth memory at every fillholes lifecycle point so we stop
// guessing when the user asks "is X GB ok?".
static double rssGB() {
	FILE *f = std::fopen("/proc/self/statm", "r");
	if (!f) return 0.0;
	long sz = 0, resident = 0;
	if (std::fscanf(f, "%ld %ld", &sz, &resident) != 2) {
		std::fclose(f);
		return 0.0;
	}
	std::fclose(f);
	const long pageSize = sysconf(_SC_PAGESIZE);
	return (double)resident * (double)pageSize / (1024.0 * 1024.0 * 1024.0);
}

// Hard cap on RSS at heartbeat sites. Set high enough that legitimate cs=2 work
// fits, low enough to avoid OOM on a 125 GB machine. Checked from heartbeat
// callbacks; on overrun the caller logs and returns early instead of getting
// killed by the OOM-killer.
static constexpr double kHolefillRSSCapGB = 100.0;
static std::atomic<bool> g_rssCapTripped{false};

static bool checkRSSCap(const char *where) {
	const double rss = rssGB();
	if (rss > kHolefillRSSCapGB && !g_rssCapTripped.exchange(true)) {
		Log::error("3dprint holefill: RSS=%.1f GB exceeds cap %.1f GB at %s -- aborting",
				   rss, kHolefillRSSCapGB, where);
		return true;
	}
	return g_rssCapTripped.load(std::memory_order_relaxed);
}

static void logRSS(const char *label) {
	Log::info("3dprint holefill: [RSS=%.1f GB] %s", rssGB(), label);
}

static constexpr uint8_t kTagOuter  = 0;
static constexpr uint8_t kTagInner  = 1;
static constexpr uint8_t kTagThin   = 2;
static constexpr uint8_t kTagBuried = 3;

// State values stored in the flat visited array
static constexpr uint8_t kStateEmpty    = 0; // unvisited empty (unknown)
static constexpr uint8_t kStateExterior = 1;
static constexpr uint8_t kStateInterior = 2;
static constexpr uint8_t kStateSolid    = 3; // solid placeholder so BFS skips it
static constexpr uint8_t kStateBlocked  = 4; // hole: exterior tried to enter coarse-interior, quarantined

struct ClassifyResult {
	glm::ivec3 localPos;
	uint8_t tag;
};

struct NodeInfo {
	uint64_t nodeId;
	voxel::RawVolume *rv;
	glm::mat4 worldMat;
	glm::mat4 invWorldMat;
	glm::ivec3 cellOrigin; // snapped to coarse cell grid
	// Cached fill colour palette index. Set in a single-threaded pre-pass
	// before parallel plug detection so threads can read lock-free instead of
	// modifying the palette concurrently.
	uint8_t fillColorIdx = 0;
};

using CellHash = std::unordered_map<glm::ivec3, uint64_t, glm::hash<glm::ivec3>>;
using CellSet  = std::unordered_set<glm::ivec3, glm::hash<glm::ivec3>>;

// Flat open-addressing hash set for glm::ivec3 voxel positions. Replaces
// std::unordered_set in the sparse-BFS hot path. Why:
//   std::unordered_set is chained -- per-entry node allocations scattered
//   across hundreds of MB of heap. With 9.6M+ entries, every find() chases
//   pointers through L3 misses, costing ~500-1000ns per lookup. The sparse
//   BFS does ~30 lookups per voxel; with millions of voxels per round, this
//   dominates runtime.
//   Flat hash keeps everything in one contiguous power-of-two array with
//   linear probing. Probes hit the cache line of the bucket plus 1-2 more
//   on collision, instead of jumping to a heap node.
//
// Safety: not thread-safe. Used here in the read-only-during-parallel-scatter,
// modify-only-during-sequential-merge pattern, so no concurrent mutation.
// Sentinel: x == kEmptySentinel marks a never-used slot; x == kTombSentinel
// marks an erased slot (probes pass through, inserts can reuse). Real voxel
// positions never land at INT_MIN -- grid coords are bounded.
class FlatVoxelSet {
private:
	static constexpr uint64_t kEmptySentinel = INT_MIN;
	static constexpr uint64_t kTombSentinel  = INT_MIN + 1;

	glm::ivec3 *_data = nullptr;
	size_t _capacity = 0;
	size_t _size = 0;

	static uint64_t hashKey(const glm::ivec3 &v) {
		// Murmur-style avalanche on the 3 components folded into a 64-bit value.
		uint64_t h = (uint64_t)(uint32_t)v.x;
		h = (h ^ ((uint64_t)(uint32_t)v.y << 21)) * 0x9E3779B97F4A7C15ull;
		h = (h ^ ((uint64_t)(uint32_t)v.z << 11)) * 0x9E3779B97F4A7C15ull;
		h ^= h >> 33;
		return h;
	}

	bool insertNoGrow(const glm::ivec3 &v) {
		const size_t mask = _capacity - 1;
		size_t idx = (size_t)hashKey(v) & mask;
		size_t firstTomb = SIZE_MAX;
		while (true) {
			const glm::ivec3 &slot = _data[idx];
			if (slot.x == kEmptySentinel) {
				const size_t target = (firstTomb != SIZE_MAX) ? firstTomb : idx;
				_data[target] = v;
				++_size;
				return true;
			}
			if (slot.x == kTombSentinel) {
				if (firstTomb == SIZE_MAX) firstTomb = idx;
			} else if (slot == v) {
				return false;
			}
			idx = (idx + 1) & mask;
		}
	}

	void grow(size_t newCap) {
		glm::ivec3 *newData = new glm::ivec3[newCap];
		const glm::ivec3 emptyVal(kEmptySentinel, 0, 0);
		for (size_t i = 0; i < newCap; ++i) newData[i] = emptyVal;
		glm::ivec3 *oldData = _data;
		const size_t oldCap = _capacity;
		_data = newData;
		_capacity = newCap;
		_size = 0;
		for (size_t i = 0; i < oldCap; ++i) {
			if (oldData[i].x != kEmptySentinel && oldData[i].x != kTombSentinel) {
				insertNoGrow(oldData[i]);
			}
		}
		delete[] oldData;
	}

public:
	FlatVoxelSet() = default;
	~FlatVoxelSet() { delete[] _data; }
	FlatVoxelSet(const FlatVoxelSet &) = delete;
	FlatVoxelSet &operator=(const FlatVoxelSet &) = delete;
	FlatVoxelSet(FlatVoxelSet &&other) noexcept
		: _data(other._data), _capacity(other._capacity), _size(other._size) {
		other._data = nullptr;
		other._capacity = 0;
		other._size = 0;
	}

	void reserve(size_t expected) {
		size_t cap = 16;
		// Target ~50% max load factor to keep probe chains short.
		while (cap < expected * 2) cap *= 2;
		if (cap > _capacity) grow(cap);
	}

	// Returns true if newly inserted, false if already present.
	bool insert(const glm::ivec3 &v) {
		if ((_size + 1) * 2 > _capacity) {
			grow(_capacity ? _capacity * 2 : 16);
		}
		return insertNoGrow(v);
	}

	bool contains(const glm::ivec3 &v) const {
		if (_capacity == 0) return false;
		const size_t mask = _capacity - 1;
		size_t idx = (size_t)hashKey(v) & mask;
		while (true) {
			const glm::ivec3 &slot = _data[idx];
			if (slot.x == kEmptySentinel) return false;
			if (slot.x != kTombSentinel && slot == v) return true;
			idx = (idx + 1) & mask;
		}
	}

	bool erase(const glm::ivec3 &v) {
		if (_capacity == 0) return false;
		const size_t mask = _capacity - 1;
		size_t idx = (size_t)hashKey(v) & mask;
		while (true) {
			glm::ivec3 &slot = _data[idx];
			if (slot.x == kEmptySentinel) return false;
			if (slot.x != kTombSentinel && slot == v) {
				slot = glm::ivec3(kTombSentinel, 0, 0);
				--_size;
				return true;
			}
			idx = (idx + 1) & mask;
		}
	}

	size_t size() const { return _size; }
};

// One resolution level.
// Solid lookup: sparse CellHash (cells that contain >= 1 solid voxel).
// Exterior/interior: flat uint8_t state array indexed by grid position.
//   Size = dimX * dimY * dimZ bytes. For level-16 grid this is ~12 MB vs
//   ~450 MB for an unordered_set -- 37x smaller, fits in L3 cache.
struct Level {
	glm::ivec3 gridLower{0};
	// cellSize, dim* stay int: cellSize is small (2..128), dim* per-axis is small
	// (~2000 max at cs=2). Their PRODUCT uses size_t (state.size()). int is
	// required here because glm::ivec3 holds int and `glm::ivec3 * cellSize` in
	// toCell() needs a matching scalar type, and inBounds() needs signed
	// semantics for the >= 0 check.
	int cellSize = 0;
	int dimX = 0, dimY = 0, dimZ = 0;

	CellHash solid;             // sparse solid map: snapped cell → node index
	std::vector<uint8_t> state; // flat: kStateEmpty/Exterior/Interior per cell

	void initGrid(const glm::ivec3 &lo, const glm::ivec3 &hi) {
		gridLower = lo;
		dimX = (hi.x - lo.x) / cellSize + 1;
		dimY = (hi.y - lo.y) / cellSize + 1;
		dimZ = (hi.z - lo.z) / cellSize + 1;
		state.assign((size_t)dimX * (size_t)dimY * (size_t)dimZ, kStateEmpty);
	}

	bool inBounds(const glm::ivec3 &cell) const {
		const int ix = (cell.x - gridLower.x) / cellSize;
		const int iy = (cell.y - gridLower.y) / cellSize;
		const int iz = (cell.z - gridLower.z) / cellSize;
		return ix >= 0 && ix < dimX && iy >= 0 && iy < dimY && iz >= 0 && iz < dimZ;
	}

	size_t toIdx(const glm::ivec3 &cell) const {
		return (size_t)((cell.x - gridLower.x) / cellSize) * (size_t)dimY * (size_t)dimZ
			 + (size_t)((cell.y - gridLower.y) / cellSize) * (size_t)dimZ
			 + (size_t)((cell.z - gridLower.z) / cellSize);
	}

	glm::ivec3 toCell(size_t idx) const {
		const int iz = (int)(idx % (size_t)dimZ);
		const int iy = (int)((idx / (size_t)dimZ) % (size_t)dimY);
		const int ix = (int)(idx / ((size_t)dimY * (size_t)dimZ));
		return gridLower + glm::ivec3(ix, iy, iz) * cellSize;
	}

	uint8_t cellState(const glm::ivec3 &cell) const {
		if (!inBounds(cell)) return kStateExterior; // outside grid = exterior by definition
		return state[toIdx(cell)];
	}
};

static double elapsedSince(uint64_t startMs) {
	return (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0;
}

static glm::ivec3 transformPoint(const glm::mat4 &m, const glm::ivec3 &p) {
	const glm::vec4 w = m * glm::vec4((float)p.x, (float)p.y, (float)p.z, 1.0f);
	// MUST be (int), not (uint64_t): world voxel coordinates can be negative
	// after applying a transform with translation. Casting through unsigned
	// would wrap negatives to huge positives.
	return glm::ivec3((int)glm::round(w.x), (int)glm::round(w.y), (int)glm::round(w.z));
}

// Signed floor division. For negative dividends this differs from C/C++
// truncation: floorDiv(-1, 4) = -1, while -1/4 = 0. The (a^b)<0 trick relies
// on signed semantics; uint64_t breaks this.
static int floorDiv(int a, int b) {
	return a / b - (a % b != 0 && (a ^ b) < 0);
}

static glm::ivec3 toCellOrigin(const glm::ivec3 &worldPos, int cellSize) {
	return glm::ivec3(floorDiv(worldPos.x, cellSize) * cellSize,
					  floorDiv(worldPos.y, cellSize) * cellSize,
					  floorDiv(worldPos.z, cellSize) * cellSize);
}

// Build solid hash (parallel per-node, then sequential merge).
// Each node collects at most (nodeSize/fineSize)^3 unique cells (512 at level 16).
static void buildSolidHash(Level &lvl, const core::DynamicArray<NodeInfo> &nodes,
							ProgressTimer *timer = nullptr) {
	// numNodes stays int: app::for_parallel takes int range. Node coordinates
	// (x/y/z below) stay int because voxel::Region returns int and node-local
	// coords can be small or negative.
	const int numNodes = (int)nodes.size();
	core::DynamicArray<CellSet> perNodeCells;
	perNodeCells.resize((size_t)numNodes);

	std::atomic<int> processed{0};
	app::for_parallel(0, numNodes, [&](int start, int end) {
		for (int i = start; i < end; ++i) {
			const NodeInfo &ni = nodes[i];
			const voxel::Region &r = ni.rv->region();
			CellSet &cs = perNodeCells[i];
			size_t solidCount = 0;
			for (int z = r.getLowerZ(); z <= r.getUpperZ(); ++z)
				for (int y = r.getLowerY(); y <= r.getUpperY(); ++y)
					for (int x = r.getLowerX(); x <= r.getUpperX(); ++x)
						if (!voxel::isAir(ni.rv->voxel(x, y, z).getMaterial()))
							++solidCount;
			cs.reserve(solidCount);
			for (int z = r.getLowerZ(); z <= r.getUpperZ(); ++z) {
				for (int y = r.getLowerY(); y <= r.getUpperY(); ++y) {
					for (int x = r.getLowerX(); x <= r.getUpperX(); ++x) {
						if (!voxel::isAir(ni.rv->voxel(x, y, z).getMaterial())) {
							const glm::ivec3 w = transformPoint(ni.worldMat, glm::ivec3(x, y, z));
							cs.insert(toCellOrigin(w, lvl.cellSize));
						}
					}
				}
			}
			if (timer) {
				timer->addVoxels((int64_t)solidCount);
				timer->tick(++processed);
			}
		}
	});

	size_t totalCells = 0;
	for (int i = 0; i < numNodes; ++i) {
		totalCells += perNodeCells[i].size();
	}
	lvl.solid.reserve(totalCells);
	for (int i = 0; i < numNodes; ++i) {
		for (const glm::ivec3 &cell : perNodeCells[i]) {
			lvl.solid.emplace(cell, i);
		}
	}
}

// Parallel frontier BFS.
// Safety: concurrent writes of the same value to a flat uint8_t array are safe on x86:
//   - All writes are idempotent (always write kStateExterior=1 or kStateInterior=2)
//   - Byte writes don't tear on x86
//   - Some cells may enter the frontier twice (harmless: second pass finds no new neighbours)
// Returns the count of cells marked with the given stateValue.
static uint64_t parallelBFS(Level &lvl, core::DynamicArray<int32_t> &frontier, uint8_t stateValue) {
	uint64_t totalMarked = 0;
	while (!frontier.empty()) {
		// sz stays int because for_parallel takes int range. Frontier is always
		// well under INT_MAX (bounded by surface area in cells). totalMarked is
		// the running sum across rounds and stays uint64_t.
		const int sz = (int)frontier.size();
		totalMarked += (uint64_t)sz;

		core::DynamicArray<core::DynamicArray<int32_t>> perRangeNext;
		const int maxSlots = 64;
		perRangeNext.resize((size_t)maxSlots);

		std::atomic<int> slotCounter{0};
		app::for_parallel(0, sz, [&](int start, int end) {
			const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % maxSlots;
			core::DynamicArray<int32_t> &local = perRangeNext[slot];
			for (int i = start; i < end; ++i) {
				const glm::ivec3 cur = lvl.toCell((size_t)frontier[i]);
				for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
					const glm::ivec3 nb = cur + off * lvl.cellSize;
					if (!lvl.inBounds(nb)) continue;
					if (lvl.solid.count(nb) > 0) continue;
					const size_t nbIdx = lvl.toIdx(nb);
					if (lvl.state[nbIdx] != kStateEmpty) continue;
					lvl.state[nbIdx] = stateValue;
					local.push_back((int32_t)nbIdx);
				}
			}
		});

		frontier.clear();
		for (int s = 0; s < maxSlots; ++s) {
			for (int32_t idx : perRangeNext[s]) {
				frontier.push_back(idx);
			}
		}
	}
	return totalMarked;
}

// Exterior flood fill (6-connectivity from all 6 grid faces) using parallel BFS.
// Detects holes: if any exterior cell's coarse parent was interior → returns hole cell size.
static uint64_t buildExterior(Level &lvl, const Level &prevLevel) {
	core::DynamicArray<int32_t> frontier;
	frontier.reserve((size_t)lvl.dimX * (size_t)lvl.dimY * 4); // rough seed count estimate

	// int64_t for loop counters: signed (gridLower can be negative), no overflow
	// at billions, same speed as int on x86-64 (often faster -- register-native).
	// Convert to int at glm::ivec3 construction since ivec3 components are int.
	auto seedCell = [&](int64_t cx, int64_t cy, int64_t cz) {
		const glm::ivec3 cell((int)cx, (int)cy, (int)cz);
		if (lvl.solid.count(cell) > 0) return;
		const size_t idx = lvl.toIdx(cell);
		if (lvl.state[idx] != kStateEmpty) return;
		lvl.state[idx] = kStateExterior;
		frontier.push_back((int32_t)idx);
	};

	const int64_t maxX = (int64_t)lvl.gridLower.x + (int64_t)(lvl.dimX - 1) * lvl.cellSize;
	const int64_t maxY = (int64_t)lvl.gridLower.y + (int64_t)(lvl.dimY - 1) * lvl.cellSize;
	const int64_t maxZ = (int64_t)lvl.gridLower.z + (int64_t)(lvl.dimZ - 1) * lvl.cellSize;

	for (int64_t cy = lvl.gridLower.y; cy <= maxY; cy += lvl.cellSize) {
		for (int64_t cz = lvl.gridLower.z; cz <= maxZ; cz += lvl.cellSize) {
			seedCell(lvl.gridLower.x, cy, cz);
			seedCell(maxX,            cy, cz);
		}
	}
	for (int64_t cx = (int64_t)lvl.gridLower.x + lvl.cellSize; cx < maxX; cx += lvl.cellSize) {
		for (int64_t cz = lvl.gridLower.z; cz <= maxZ; cz += lvl.cellSize) {
			seedCell(cx, lvl.gridLower.y, cz);
			seedCell(cx, maxY,            cz);
		}
	}
	for (int64_t cx = (int64_t)lvl.gridLower.x + lvl.cellSize; cx < maxX; cx += lvl.cellSize) {
		for (int64_t cy = (int64_t)lvl.gridLower.y + lvl.cellSize; cy < maxY; cy += lvl.cellSize) {
			seedCell(cx, cy, lvl.gridLower.z);
			seedCell(cx, cy, maxZ           );
		}
	}

	parallelBFS(lvl, frontier, kStateExterior);

	// Hole detection: scan for exterior cells whose coarse parent was interior.
	// Only do this if we have a previous level with interior.
	if (prevLevel.cellSize > 0) {
		const size_t total = lvl.state.size();
		for (size_t i = 0; i < total; ++i) {
			if (lvl.state[i] == kStateExterior) {
				const glm::ivec3 cell = lvl.toCell((uint64_t)i);
				if (prevLevel.cellState(toCellOrigin(cell, prevLevel.cellSize)) == kStateInterior) {
					return lvl.cellSize; // hole found at this scale
				}
			}
		}
	}
	return 0;
}

// Find the kStateEmpty cell nearest to the geometric center of the grid.
// Starting from center lands in the true model interior rather than a peripheral
// corridor or cavity that happens to be first in scan order.
static bool findInteriorSeedFromCenter(size_t &seedIdx, const Level &lvl) {
	const float cx = (float)lvl.dimX * 0.5f;
	const float cy = (float)lvl.dimY * 0.5f;
	const float cz = (float)lvl.dimZ * 0.5f;
	float bestDist = FLT_MAX;
	bool found = false;
	const size_t total = lvl.state.size();
	for (size_t i = 0; i < total; ++i) {
		if (lvl.state[i] != kStateEmpty) continue;
		const uint64_t ix = (uint64_t)(i / ((size_t)lvl.dimY * lvl.dimZ));
		const uint64_t iy = (uint64_t)((i / (size_t)lvl.dimZ) % (size_t)lvl.dimY);
		const uint64_t iz = (uint64_t)(i % (size_t)lvl.dimZ);
		const float dx = (float)ix - cx;
		const float dy = (float)iy - cy;
		const float dz = (float)iz - cz;
		const float dist = dx * dx + dy * dy + dz * dz;
		if (dist < bestDist) {
			bestDist = dist;
			seedIdx = i;
			found = true;
		}
	}
	return found;
}

// Interior flood fill (26-connectivity) from a single seed cell.
// Returns the number of cells marked interior.
static uint64_t buildInteriorAllSeeds(Level &lvl) {
	size_t seedIdx = 0;
	if (!findInteriorSeedFromCenter(seedIdx, lvl)) {
		return 0;
	}
	lvl.state[seedIdx] = kStateInterior;
	core::DynamicArray<int32_t> queue;
	queue.push_back((int32_t)seedIdx);
	uint64_t head = 0;
	while (head < (uint64_t)queue.size()) {
		const glm::ivec3 cur = lvl.toCell((size_t)queue[head++]);
		for (int dx = -1; dx <= 1; ++dx) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dz = -1; dz <= 1; ++dz) {
					if (dx == 0 && dy == 0 && dz == 0) continue;
					const glm::ivec3 nb = cur + glm::ivec3(dx, dy, dz) * lvl.cellSize;
					if (!lvl.inBounds(nb)) continue;
					if (lvl.solid.count(nb) > 0) continue;
					const int32_t nbIdx = (int32_t)lvl.toIdx(nb);
					if (lvl.state[(size_t)nbIdx] != kStateEmpty) continue;
					lvl.state[(size_t)nbIdx] = kStateInterior;
					queue.push_back(nbIdx);
				}
			}
		}
	}
	return (uint64_t)queue.size();
}

// Solid voxel check using level's solid hash.
static bool isSolidAt(const glm::ivec3 &worldPos, const Level &lvl,
					  const core::DynamicArray<NodeInfo> &nodes) {
	const glm::ivec3 cell = toCellOrigin(worldPos, lvl.cellSize);
	auto it = lvl.solid.find(cell);
	if (it == lvl.solid.end()) return false;
	const NodeInfo &ni = nodes[it->second];
	const glm::ivec3 local = transformPoint(ni.invWorldMat, worldPos);
	return ni.rv->region().containsPoint(local) && !voxel::isAir(ni.rv->voxel(local).getMaterial());
}

static uint64_t countCellsByState(const Level &lvl, uint8_t s) {
	uint64_t n = 0;
	for (uint8_t v : lvl.state) {
		if (v == s) ++n;
	}
	return n;
}

// Seed fine cell states from the previous (coarser) level classification.
// Cells whose coarse parent is exterior/interior get that state.
// Cells in coarse-solid parents that are not in fine.solid stay kStateEmpty (the probe zone).
// Fine solid cells must already be marked kStateSolid by the caller.
static void initFineFromPrev(Level &fine, const Level &prev) {
	// Each slot writes only to its own index -- safe to parallelize.
	// prev.state reads are concurrent but read-only -- safe on all platforms.
	// Chunked by 1B cells because app::for_parallel takes int range; at cs=2 the
	// total exceeds INT_MAX. Per-chunk parallel scan with heartbeat; outer loop
	// sequences chunks. NO sequential remainder -- everything goes through the
	// parallel path.
	const size_t total = fine.state.size();
	const int64_t totalS = (int64_t)total;
	const uint64_t startMs = core::TimeProvider::systemMillis();
	Log::info("3dprint holefill: initFineFromPrev scanning %lld cells (cs=%d -> cs=%d)",
			  (long long)totalS, prev.cellSize, fine.cellSize);
	std::atomic<int64_t> processed{0};
	std::atomic<uint64_t> lastLogMs{startMs};
	static constexpr int64_t kChunkSize = 1ll << 30; // 1B cells per chunk -- fits in int range
	for (int64_t base = 0; base < totalS; base += kChunkSize) {
		const int64_t chunkEnd = glm::min(base + kChunkSize, totalS);
		const int chunkLen = (int)(chunkEnd - base);
		app::for_parallel(0, chunkLen, [&](int start, int end) {
			int64_t since = 0;
			for (int64_t i = base + (int64_t)start; i < base + (int64_t)end; ++i) {
				if (fine.state[(size_t)i] == kStateSolid) { ++since; continue; }
				const glm::ivec3 cell = fine.toCell((size_t)i);
				const uint8_t parentState = prev.cellState(toCellOrigin(cell, prev.cellSize));
				if (parentState == kStateExterior) {
					fine.state[(size_t)i] = kStateExterior;
				} else if (parentState == kStateInterior) {
					fine.state[(size_t)i] = kStateInterior;
				}
				if ((++since & 0x3FFFFF) == 0) {
					const int64_t globalDone = processed.fetch_add(since, std::memory_order_relaxed) + since;
					since = 0;
					const uint64_t now = core::TimeProvider::systemMillis();
					uint64_t prevMs = lastLogMs.load(std::memory_order_relaxed);
					if (now - prevMs >= 3000u && lastLogMs.compare_exchange_strong(prevMs, now)) {
						fprintf(stderr, "[INIT] initFineFromPrev %lld/%lld (%.1f%%) elapsed=%.1fs [RSS=%.1f GB]\n",
								(long long)globalDone, (long long)totalS,
								100.0 * (double)globalDone / (double)totalS,
								(double)(now - startMs) / 1000.0,
								rssGB());
						fflush(stderr);
						checkRSSCap("initFineFromPrev heartbeat");
					}
					if (g_rssCapTripped.load(std::memory_order_relaxed)) return;
				}
			}
			processed.fetch_add(since, std::memory_order_relaxed);
		});
		if (g_rssCapTripped.load(std::memory_order_relaxed)) return;
	}
}

// Scan the grid and build exterior and interior frontiers:
// cells that are exterior/interior AND have at least one neighbour that's a "wall".
// What counts as a wall depends on wrapSolid:
//   wrapSolid=false: only kStateEmpty (probe-zone) neighbours count -- used by
//      runHoleMap/runHoleFill where the BFS is between coarse-derived classification
//      and unclassified probe zones.
//   wrapSolid=true:  kStateEmpty OR kStateSolid neighbours count -- used by
//      single-level debugfrontier where the wrap should hug the actual fine solid
//      voxels too. Without this, an ext/int cell whose only "into the wall"
//      neighbour is a solid voxel is dropped, which leaves visible holes where a
//      thin solid skin runs along the boundary.
static void buildFrontiers(const Level &lvl,
							core::DynamicArray<size_t> &extFrontier,
							core::DynamicArray<size_t> &intFrontier,
							bool wrapSolid = false) {
	const size_t total = lvl.state.size();
	const int64_t totalS = (int64_t)total;
	const int nSlots = 64;
	core::DynamicArray<core::DynamicArray<size_t>> perExt, perInt;
	perExt.resize((size_t)nSlots);
	perInt.resize((size_t)nSlots);

	// Do NOT reserve based on grid size -- frontier is proportional to surface area, not volume.

	const uint64_t startMs = core::TimeProvider::systemMillis();
	Log::info("3dprint holefill: buildFrontiers scanning %lld cells (cs=%d, wrapSolid=%d)",
			  (long long)totalS, lvl.cellSize, wrapSolid ? 1 : 0);
	std::atomic<int> slotCounter{0};
	std::atomic<int64_t> processed{0};
	std::atomic<uint64_t> lastLogMs{startMs};
	// Chunked by 1B cells because app::for_parallel takes int range; at cs=2 the
	// total exceeds INT_MAX. NO sequential remainder; previously cells beyond
	// INT_MAX were silently dropped (correctness bug at cs=2).
	static constexpr int64_t kChunkSize = 1ll << 30;
	for (int64_t base = 0; base < totalS; base += kChunkSize) {
		const int64_t chunkEnd = glm::min(base + kChunkSize, totalS);
		const int chunkLen = (int)(chunkEnd - base);
		app::for_parallel(0, chunkLen, [&](int start, int end) {
			const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % nSlots;
			core::DynamicArray<size_t> &locExt = perExt[(size_t)slot];
			core::DynamicArray<size_t> &locInt = perInt[(size_t)slot];
			int64_t since = 0;
			for (int64_t i = base + (int64_t)start; i < base + (int64_t)end; ++i) {
				const uint8_t s = lvl.state[(size_t)i];
				if (s == kStateExterior || s == kStateInterior) {
					const glm::ivec3 cur = lvl.toCell((size_t)i);
					for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
						const glm::ivec3 nb = cur + off * lvl.cellSize;
						if (!lvl.inBounds(nb)) continue;
						const uint8_t nbState = lvl.state[lvl.toIdx(nb)];
						const bool isWall = nbState == kStateEmpty
							|| (wrapSolid && (nbState == kStateSolid || nbState == kStateBlocked));
						if (isWall) {
							if (s == kStateExterior) locExt.push_back((size_t)i);
							else locInt.push_back((size_t)i);
							break;
						}
					}
				}
				if ((++since & 0x3FFFFF) == 0) {
					const int64_t globalDone = processed.fetch_add(since, std::memory_order_relaxed) + since;
					since = 0;
					const uint64_t now = core::TimeProvider::systemMillis();
					uint64_t prevMs = lastLogMs.load(std::memory_order_relaxed);
					if (now - prevMs >= 3000u && lastLogMs.compare_exchange_strong(prevMs, now)) {
						fprintf(stderr, "[FRONT] buildFrontiers %lld/%lld (%.1f%%) elapsed=%.1fs [RSS=%.1f GB]\n",
								(long long)globalDone, (long long)totalS,
								100.0 * (double)globalDone / (double)totalS,
								(double)(now - startMs) / 1000.0,
								rssGB());
						fflush(stderr);
						checkRSSCap("buildFrontiers heartbeat");
					}
					if (g_rssCapTripped.load(std::memory_order_relaxed)) return;
				}
			}
			processed.fetch_add(since, std::memory_order_relaxed);
		});
		if (g_rssCapTripped.load(std::memory_order_relaxed)) return;
	}

	size_t extTotal = 0, intTotal = 0;
	for (uint64_t s = 0; s < nSlots; ++s) {
		extTotal += perExt[(size_t)s].size();
		intTotal += perInt[(size_t)s].size();
	}
	extFrontier.reserve(extTotal);
	intFrontier.reserve(intTotal);
	for (uint64_t s = 0; s < nSlots; ++s) {
		for (size_t idx : perExt[(size_t)s]) extFrontier.push_back(idx);
		for (size_t idx : perInt[(size_t)s]) intFrontier.push_back(idx);
	}
}


static core::DynamicArray<glm::ivec3> runBidirectionalBFS(Level &lvl) {
	core::DynamicArray<size_t> extFrontier, intFrontier;
	buildFrontiers(lvl, extFrontier, intFrontier);
	Log::info("holefill BFS: ext frontier=%zu int frontier=%zu", extFrontier.size(), intFrontier.size());

	core::DynamicArray<glm::ivec3> holeCells;

	uint64_t bfsRound = 0;
	uint64_t bfsStartMs = core::TimeProvider::systemMillis();
	uint64_t lastLogMs = bfsStartMs;
	while (!extFrontier.empty() || !intFrontier.empty()) {
		++bfsRound;
		const uint64_t nowMs = core::TimeProvider::systemMillis();
		if (nowMs - lastLogMs >= 5000u) {
			lastLogMs = nowMs;
			fprintf(stderr, "[BFS] round %d, ext=%zu int=%zu holes=%zu elapsed=%.1fs [RSS=%.1f GB]\n",
					(int)bfsRound, extFrontier.size(), intFrontier.size(),
					holeCells.size(), (double)(nowMs - bfsStartMs) / 1000.0,
					rssGB());
			fflush(stderr);
			if (checkRSSCap("BFS round heartbeat")) return holeCells;
		}
		core::DynamicArray<size_t> nextExt;
		// BFS expansion can produce up to 6 neighbours per frontier cell.
		// Reserve to that ceiling so push_back never reallocs mid-loop.
		nextExt.reserve(extFrontier.size() * 6);
		for (size_t idx : extFrontier) {
			const glm::ivec3 cur = lvl.toCell(idx);
			for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nb = cur + off * lvl.cellSize;
				if (!lvl.inBounds(nb)) continue;
				const size_t nbIdx = lvl.toIdx(nb);
				const uint8_t nbState = lvl.state[nbIdx];
				if (nbState == kStateSolid || nbState == kStateExterior || nbState == kStateBlocked) continue;
				if (nbState == kStateInterior) {
					// Ext-meets-int: this is a hole. Recording must be SYMMETRIC
					// with the int-meets-ext branch below. Without recording here,
					// any hole reached by ext FIRST gets silently absorbed (int
					// neighbour becomes kStateBlocked, BFS stops on both sides,
					// no hole emitted). That was the cause of multiple 4-voxel
					// holes being undetected at cs=2 even though the frontiers
					// physically met there.
					lvl.state[nbIdx] = kStateBlocked;
					holeCells.push_back(nb);
				} else {
					lvl.state[nbIdx] = kStateExterior;
					nextExt.push_back(nbIdx);
				}
			}
		}
		extFrontier = core::move(nextExt);

		core::DynamicArray<size_t> nextInt;
		nextInt.reserve(intFrontier.size() * 6);
		for (size_t idx : intFrontier) {
			const glm::ivec3 cur = lvl.toCell(idx);
			for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nb = cur + off * lvl.cellSize;
				if (!lvl.inBounds(nb)) continue;
				const size_t nbIdx = lvl.toIdx(nb);
				const uint8_t nbState = lvl.state[nbIdx];
				if (nbState == kStateSolid || nbState == kStateInterior || nbState == kStateBlocked) continue;
				if (nbState == kStateExterior) {
					lvl.state[nbIdx] = kStateBlocked;
					holeCells.push_back(nb);
				} else {
					lvl.state[nbIdx] = kStateInterior;
					nextInt.push_back(nbIdx);
				}
			}
		}
		intFrontier = core::move(nextInt);
	}

	return holeCells;
}

// Sparse voxel-level BFS that runs after the dense progressive chain in runHoleFill
// terminates (either reached minCellSize or hit budget). It bridges the remaining
// gap at voxel resolution without ever allocating a dense state grid -- storage
// is a hash set per shell, bounded by O(model surface area in voxels).
//
// Inputs:
//   prev      : the deepest dense Level produced by runHoleFill (full state).
//   nodes     : nodes for actual voxel-level solid lookups via isSolidAt.
//   outHoleVoxels : append-only list of single-voxel hole positions.
//
// Algorithm per round:
//   1. Ext expansion first (preserves model surface).
//   2. Int expansion second.
//   3. Either ext-meets-uint64_t OR uint64_t-meets-ext converts the contacted voxel to
//      solid (recorded as a hole) and removes it from the opposite shell.
//   4. Solidified voxels never seed expansion in subsequent rounds.
//   5. Propagation cannot enter dense cells of the OPPOSITE classification
//      (ext can't enter dense uint64_t; uint64_t can't enter dense ext) -- this keeps
//      the BFS confined to a thin band along the wall.
static void runSparseVoxelBFS(const Level &prev,
                               const core::DynamicArray<NodeInfo> &nodes,
                               core::DynamicArray<glm::ivec3> &outHoleVoxels) {
	const int cs = prev.cellSize;
	const uint64_t startMs = core::TimeProvider::systemMillis();

	// Unconditional entry log so we can prove the sparse pass started.
	Log::info("3dprint holefill sparse: ENTRY cellSize=%d, dense state cells=%zu, solid cells=%zu, nodes=%zu",
			  cs, prev.state.size(), prev.solid.size(), nodes.size());

	// Seeding: parallel scan of dense state. For each ext/uint64_t cell, check each of
	// the 6 face neighbours; if a neighbour is non-classified (solid/empty/blocked),
	// the corresponding cs*cs voxel face of this cell goes into the seed set.
	const size_t total = prev.state.size();
	constexpr uint64_t kSlots = 64;
	core::DynamicArray<core::DynamicArray<glm::ivec3>> perExtSeed, perIntSeed;
	perExtSeed.resize((size_t)kSlots);
	perIntSeed.resize((size_t)kSlots);
	// Per-slot reserves: prevent per-shell-cell push_backs from triggering
	// realloc-storms under 23 concurrent threads. Estimate from solid-cell count
	// (a rough proxy for surface area). With 1 voxel per wall-facing face after
	// the seeding reduction, 6 faces max per shell cell, divided across slots.
	// Generous floor so small models don't pay setup cost.
	const size_t perSlotReserve = (size_t)glm::max((size_t)4096, prev.solid.size() * 6 / (size_t)kSlots);
	for (uint64_t s = 0; s < kSlots; ++s) {
		perExtSeed[(size_t)s].reserve(perSlotReserve);
		perIntSeed[(size_t)s].reserve(perSlotReserve);
	}
	std::atomic<uint64_t> slotCounter{0};
	Log::info("3dprint holefill sparse: dispatching parallel seed scan over %zu cells (%.1fs since entry)",
			  total, (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0);
	// Periodic-progress plumbing -- every ~5s one thread emits a "still alive"
	// line with rate + ETA so multi-minute runs don't look like a deadlock.
	std::atomic<int64_t> seedProcessed{0};
	std::atomic<uint64_t> seedLastLogMs{startMs};
	const int64_t seedTotal = (int64_t)total;
	// Wall-driven seeding. Most cells in the grid are dense ext or dense uint64_t
	// after the bidirectional BFS at the deepest dense level. Only a tiny
	// fraction (~5M of ~950M for a 3000^3 model at cs=4) are actual walls
	// (kStateSolid or kStateBlocked). We only need shell voxels on the ext/int
	// SIDE of those walls -- so iterate walls, not the whole grid.
	// Chunked by 1B cells: at cs=2 total exceeds INT_MAX which app::for_parallel
	// can't span. NO sequential remainder; previously cells past INT_MAX were
	// silently dropped from the seed scan.
	static constexpr int64_t kSparseChunk = 1ll << 30;
	for (int64_t base = 0; base < seedTotal; base += kSparseChunk) {
		const int64_t chunkEnd = glm::min(base + kSparseChunk, seedTotal);
		const int chunkLen = (int)(chunkEnd - base);
		app::for_parallel(0, chunkLen, [&](int start, int end) {
			const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % kSlots;
			core::DynamicArray<glm::ivec3> &locExt = perExtSeed[(size_t)slot];
			core::DynamicArray<glm::ivec3> &locInt = perIntSeed[(size_t)slot];
			uint64_t lastHeartbeat = core::TimeProvider::systemMillis();
			const int mid = cs / 2;
			for (int64_t i = base + (int64_t)start; i < base + (int64_t)end; ++i) {
				if ((i & 0x1FFFFFF) == 0) {
					const uint64_t now = core::TimeProvider::systemMillis();
					if (now - lastHeartbeat >= 3000u) {
						lastHeartbeat = now;
						Log::info("3dprint holefill sparse: seed scan i=%lld/%lld (%.1f%%) [RSS=%.1f GB]",
								  (long long)i, (long long)seedTotal,
								  100.0 * (double)i / (double)seedTotal,
								  rssGB());
					}
				}
				const uint8_t s = prev.state[(size_t)i];
				if (s != kStateSolid && s != kStateBlocked) continue;
				const glm::ivec3 wallCell = prev.toCell((size_t)i);
				for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
					const glm::ivec3 nb = wallCell + off * cs;
					if (!prev.inBounds(nb)) continue;
					const uint8_t nbState = prev.state[prev.toIdx(nb)];
					if (nbState != kStateExterior && nbState != kStateInterior) continue;
					glm::ivec3 vp;
					if (off.x != 0)      vp = nb + glm::ivec3(off.x > 0 ? 0 : cs - 1, mid, mid);
					else if (off.y != 0) vp = nb + glm::ivec3(mid, off.y > 0 ? 0 : cs - 1, mid);
					else                 vp = nb + glm::ivec3(mid, mid, off.z > 0 ? 0 : cs - 1);
					if (nbState == kStateExterior) locExt.push_back(vp);
					else                           locInt.push_back(vp);
				}
			}
			seedProcessed.fetch_add((int64_t)(end - start), std::memory_order_relaxed);
		});
	}
	fprintf(stderr, "[SPARSE-stderr] for_parallel returned, all chunks complete\n");
	fflush(stderr);
	Log::info("3dprint holefill sparse: parallel seed scan returned (%.1fs since entry)",
			  (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0);

	FlatVoxelSet extVox, intVox, solidified;
	core::DynamicArray<glm::ivec3> extFrontier, intFrontier;
	size_t extSeedCount = 0, intSeedCount = 0;
	for (uint64_t s = 0; s < kSlots; ++s) {
		extSeedCount += perExtSeed[(size_t)s].size();
		intSeedCount += perIntSeed[(size_t)s].size();
	}
	// Up-front reserves sized to FRONTIER GROWTH CEILING, not initial seed count.
	// Per project_holefill_sparse_revision.md the broken-by-design BFS grows the
	// frontier 2-5x per round. Observed in prior cs=2 run: int frontier hit 99M
	// from a 13.8M seed (~7x). Sizing to seed*2 forces FlatVoxelSet to grow and
	// rehash mid-merge, stalling for tens of seconds on 8-byte pointer chasing.
	//
	// Ceiling derivation: the band of air voxels reachable by the BFS without
	// crossing solid is bounded by O(solid_surface_voxel_count). Empirically
	// 16x solid_cell_count covers all observed runs with headroom. solidified
	// is bounded by the meeting surface (where ext meets int) and is at most
	// the smaller of the two shells.
	const size_t solidCells = prev.solid.size();
	const size_t shellCap = solidCells * 16;
	const size_t solidifiedCap = solidCells * 4;
	Log::info("3dprint holefill sparse: pre-allocating shells extCap=%zu intCap=%zu solidifiedCap=%zu (solidCells=%zu, ~%.1f GB)",
			  shellCap, shellCap, solidifiedCap, solidCells,
			  (double)(shellCap * 2 + solidifiedCap) * 24.0 / (1024.0 * 1024.0 * 1024.0));
	extVox.reserve(shellCap);
	intVox.reserve(shellCap);
	solidified.reserve(solidifiedCap);
	extFrontier.reserve(shellCap);
	intFrontier.reserve(shellCap);
	for (uint64_t s = 0; s < kSlots; ++s) {
		for (const glm::ivec3 &p : perExtSeed[(size_t)s]) {
			if (extVox.insert(p)) extFrontier.push_back(p);
		}
		for (const glm::ivec3 &p : perIntSeed[(size_t)s]) {
			if (intVox.insert(p)) intFrontier.push_back(p);
		}
	}
	Log::info("3dprint holefill sparse: seeded ext=%zu int=%zu (cellSize=%d, %.1fs)",
			  extVox.size(), intVox.size(), cs,
			  (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0);

	// Parallel BFS rounds. Each round is split into:
	//   1. PARALLEL scatter: read-only over extVox/intVox/solidified; per-thread
	//      buffers collect "candidates to add" and "candidates to solidify".
	//   2. SEQUENTIAL merge: solidify-before-add ordering enforces the user's
	//      rule that a freshly-solidified voxel can't seed next-round expansion
	//      (it's filtered by the `solidified` lookup before insert into
	//      ext/intVox or push into nextExt/Int).
	// Hash sets aren't safe under concurrent insert/erase, so writes are serial.
	// Reads in the scatter phase are safe because std::unordered_set permits
	// concurrent reads on a stable container.
	constexpr uint64_t kBFSSlots = 64;
	core::DynamicArray<core::DynamicArray<glm::ivec3>> perNextExt, perExtSolidify;
	core::DynamicArray<core::DynamicArray<glm::ivec3>> perNextInt, perIntSolidify;
	perNextExt.resize((size_t)kBFSSlots);
	perExtSolidify.resize((size_t)kBFSSlots);
	perNextInt.resize((size_t)kBFSSlots);
	perIntSolidify.resize((size_t)kBFSSlots);

	uint64_t round = 0;
	while (!extFrontier.empty() || !intFrontier.empty()) {
		++round;
		const uint64_t roundStart = core::TimeProvider::systemMillis();
		const size_t holesBefore = outHoleVoxels.size();
		core::DynamicArray<glm::ivec3> nextExt, nextInt;
		// BFS expansion is up to 6x current frontier (each voxel emits 6
		// neighbours). Reserving at current size forces realloc mid-merge.
		// Use shellCap as the per-round ceiling -- same bound as extVox/intVox.
		nextExt.reserve(shellCap);
		nextInt.reserve(shellCap);
		fprintf(stderr, "[SPARSE-stderr] round %d START extFront=%zu intFront=%zu solidified=%zu\n",
				round, extFrontier.size(), intFrontier.size(), solidified.size());
		fflush(stderr);

		// Reuse per-slot buffers across rounds: clear() keeps capacity, so after
		// the first round each slot's vector grows to its peak and stops
		// reallocating. Reserve a generous initial guess on the first round
		// (frontier_size / kBFSSlots * 6 face neighbours, doubled) so growth is
		// minimal even round one.
		const uint64_t extReservePerSlot = (uint64_t)(extFrontier.size() * 12 / kBFSSlots) + 16;
		const uint64_t intReservePerSlot = (uint64_t)(intFrontier.size() * 12 / kBFSSlots) + 16;
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			perNextExt[(size_t)s].clear();
			perExtSolidify[(size_t)s].clear();
			perNextInt[(size_t)s].clear();
			perIntSolidify[(size_t)s].clear();
			if (perNextExt[(size_t)s].capacity() < (size_t)extReservePerSlot)
				perNextExt[(size_t)s].reserve((size_t)extReservePerSlot);
			if (perExtSolidify[(size_t)s].capacity() < (size_t)(extReservePerSlot / 8 + 4))
				perExtSolidify[(size_t)s].reserve((size_t)(extReservePerSlot / 8 + 4));
			if (perNextInt[(size_t)s].capacity() < (size_t)intReservePerSlot)
				perNextInt[(size_t)s].reserve((size_t)intReservePerSlot);
			if (perIntSolidify[(size_t)s].capacity() < (size_t)(intReservePerSlot / 8 + 4))
				perIntSolidify[(size_t)s].reserve((size_t)(intReservePerSlot / 8 + 4));
		}

		// Step 1a: PARALLEL ext scatter. Ext expansion runs first so surface
		// detail is preserved against int over-eating. Intra-chunk progress
		// emit so chunks of millions of voxels don't go silent.
		if (!extFrontier.empty()) {
			fprintf(stderr, "[SPARSE-stderr] round %d EXT scatter dispatching (%d voxels)\n",
					round, (uint64_t)extFrontier.size());
			fflush(stderr);
			std::atomic<int> slotCounterE{0};
			std::atomic<uint64_t> extProcessed{0};
			std::atomic<uint64_t> extLastLogMs{roundStart};
			const int extTotalThis = (int)extFrontier.size();
			app::for_parallel(0, extTotalThis, [&](int start, int end) {
				const int slot = slotCounterE.fetch_add(1, std::memory_order_relaxed) % kBFSSlots;
				core::DynamicArray<glm::ivec3> &locNext = perNextExt[(size_t)slot];
				core::DynamicArray<glm::ivec3> &locSol = perExtSolidify[(size_t)slot];
				fprintf(stderr, "[SPARSE-stderr] round %d EXT chunk %d..%d slot=%d entered\n",
						round, start, end, slot);
				fflush(stderr);
				uint64_t lastHeartbeat = core::TimeProvider::systemMillis();
				constexpr uint64_t kProgressBatch = 50000;
				uint64_t batchProcessed = 0;
				for (int i = start; i < end; ++i) {
					if ((i & 0x3FFFF) == 0) {
						const uint64_t now = core::TimeProvider::systemMillis();
						if (now - lastHeartbeat >= 3000u) {
							lastHeartbeat = now;
							fprintf(stderr, "[SPARSE-stderr] round %d EXT slot=%d at i=%d of %d (%.1f%%)\n",
									round, slot, i, end,
									100.0 * (double)(i - start) / (double)(end - start));
							fflush(stderr);
						}
					}
					const glm::ivec3 &v = extFrontier[(size_t)i];
					if (!solidified.contains(v)) {
						const glm::ivec3 srcCell = toCellOrigin(v, cs);
						for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
							const glm::ivec3 n = v + off;
							if (isSolidAt(n, prev, nodes)) continue;
							if (solidified.contains(n)) continue;
							if (extVox.contains(n)) continue;
							if (intVox.contains(n)) {
								locSol.push_back(n);
								continue;
							}
							// Stay-in-band constraint. Within the source's own dense
							// cell, anything goes. Across a cell boundary, only step
							// into non-classified territory (kStateSolid/Empty/Blocked).
							// This prevents the BFS from invading dense-ext interior
							// (its own classification) which has no leaks to find and
							// would explode the frontier exponentially.
							const glm::ivec3 dc = toCellOrigin(n, cs);
							if (dc != srcCell) {
								const uint8_t nbDense = prev.cellState(dc);
								if (nbDense == kStateExterior || nbDense == kStateInterior) continue;
							}
							locNext.push_back(n);
						}
					}
					if (++batchProcessed >= kProgressBatch) {
						const uint64_t doneNow = extProcessed.fetch_add(batchProcessed,
																   std::memory_order_relaxed) + batchProcessed;
						batchProcessed = 0;
						const uint64_t now = core::TimeProvider::systemMillis();
						uint64_t prevTs = extLastLogMs.load(std::memory_order_relaxed);
						if (now - prevTs >= 5000u && extLastLogMs.compare_exchange_strong(prevTs, now)) {
							const double secs = (double)(now - roundStart) / 1000.0;
							const double rate = secs > 0.001 ? (double)doneNow / secs : 0.0;
							const double etaSec = rate > 0.0 ? (double)(extTotalThis - doneNow) / rate : 0.0;
							Log::info("3dprint holefill sparse: round %d ext scatter %d/%d (%.0f vox/s, eta %.0fs)",
									  round, doneNow, extTotalThis, rate, etaSec);
						}
					}
				}
				if (batchProcessed > 0) {
					extProcessed.fetch_add(batchProcessed, std::memory_order_relaxed);
				}
				fprintf(stderr, "[SPARSE-stderr] round %d EXT chunk %d..%d slot=%d done\n",
						round, start, end, slot);
				fflush(stderr);
			});
			fprintf(stderr, "[SPARSE-stderr] round %d EXT scatter all chunks done\n", round);
			fflush(stderr);
		}
		// Step 1b: SEQUENTIAL merge. Solidify first (so an int voxel killed by
		// ext-met-int is gone from intVox before int's own scatter runs), then
		// promote candidates to extVox + nextExt with dedup.
		const uint64_t extMergeStart = core::TimeProvider::systemMillis();
		size_t extSolidCount = 0, extNextCount = 0;
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			extSolidCount += perExtSolidify[(size_t)s].size();
			extNextCount += perNextExt[(size_t)s].size();
		}
		fprintf(stderr, "[SPARSE-stderr] round %d EXT merge START solidify=%zu next=%zu\n",
				round, extSolidCount, extNextCount);
		fflush(stderr);
		Log::info("3dprint holefill sparse: round %d ext merge starting (solidify=%zu next=%zu)",
				  round, extSolidCount, extNextCount);
		// Heartbeat for the sequential merge: 52M+ FlatVoxelSet inserts (with
		// possible rehashes) can take 10-30s silently. Log every ~3s.
		uint64_t mergeLastLog = core::TimeProvider::systemMillis();
		size_t mergeDone = 0;
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			for (const glm::ivec3 &n : perExtSolidify[(size_t)s]) {
				if (solidified.insert(n)) {
					intVox.erase(n);
					outHoleVoxels.push_back(n);
				}
				if ((++mergeDone & 0xFFFFF) == 0) {
					const uint64_t now = core::TimeProvider::systemMillis();
					if (now - mergeLastLog >= 3000u) {
						mergeLastLog = now;
						Log::info("3dprint holefill sparse: round %d ext merge solidify %zu/%zu [RSS=%.1f GB]",
								  round, mergeDone, extSolidCount, rssGB());
					}
				}
			}
		}
		mergeDone = 0;
		mergeLastLog = core::TimeProvider::systemMillis();
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			for (const glm::ivec3 &n : perNextExt[(size_t)s]) {
				if (solidified.contains(n)) continue;
				if (extVox.insert(n)) {
					nextExt.push_back(n);
				}
				if ((++mergeDone & 0xFFFFF) == 0) {
					const uint64_t now = core::TimeProvider::systemMillis();
					if (now - mergeLastLog >= 3000u) {
						mergeLastLog = now;
						Log::info("3dprint holefill sparse: round %d ext merge next %zu/%zu (extVox=%zu) [RSS=%.1f GB]",
								  round, mergeDone, extNextCount, extVox.size(), rssGB());
					}
				}
			}
		}
		Log::info("3dprint holefill sparse: round %d ext merge done in %.1fs",
				  round, (double)(core::TimeProvider::systemMillis() - extMergeStart) / 1000.0);
		fprintf(stderr, "[SPARSE-stderr] round %d EXT merge DONE in %.1fs\n",
				round, (double)(core::TimeProvider::systemMillis() - extMergeStart) / 1000.0);
		fflush(stderr);

		// Step 2a: PARALLEL int scatter. Sees the ext-step's merged solidified
		// + extVox state because the ext merge ran before this point.
		if (!intFrontier.empty()) {
			fprintf(stderr, "[SPARSE-stderr] round %d INT scatter dispatching (%d voxels)\n",
					round, (uint64_t)intFrontier.size());
			fflush(stderr);
			std::atomic<int> slotCounterI{0};
			std::atomic<uint64_t> intProcessed{0};
			const uint64_t intStart = core::TimeProvider::systemMillis();
			std::atomic<uint64_t> intLastLogMs{intStart};
			const int intTotalThis = (int)intFrontier.size();
			app::for_parallel(0, intTotalThis, [&](int start, int end) {
				const int slot = slotCounterI.fetch_add(1, std::memory_order_relaxed) % kBFSSlots;
				core::DynamicArray<glm::ivec3> &locNext = perNextInt[(size_t)slot];
				core::DynamicArray<glm::ivec3> &locSol = perIntSolidify[(size_t)slot];
				fprintf(stderr, "[SPARSE-stderr] round %d INT chunk %d..%d slot=%d entered\n",
						round, start, end, slot);
				fflush(stderr);
				uint64_t lastHeartbeat = core::TimeProvider::systemMillis();
				constexpr uint64_t kProgressBatch = 50000;
				uint64_t batchProcessed = 0;
				for (int i = start; i < end; ++i) {
					if ((i & 0x3FFFF) == 0) {
						const uint64_t now = core::TimeProvider::systemMillis();
						if (now - lastHeartbeat >= 3000u) {
							lastHeartbeat = now;
							fprintf(stderr, "[SPARSE-stderr] round %d INT slot=%d at i=%d of %d (%.1f%%)\n",
									round, slot, i, end,
									100.0 * (double)(i - start) / (double)(end - start));
							fflush(stderr);
						}
					}
					const glm::ivec3 &v = intFrontier[(size_t)i];
					if (!solidified.contains(v)) {
						const glm::ivec3 srcCell = toCellOrigin(v, cs);
						for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
							const glm::ivec3 n = v + off;
							if (isSolidAt(n, prev, nodes)) continue;
							if (solidified.contains(n)) continue;
							if (intVox.contains(n)) continue;
							if (extVox.contains(n)) {
								locSol.push_back(n);
								continue;
							}
							// Stay-in-band constraint: same as ext expansion, mirror
							// for int. Boundary crossings only enter non-classified
							// territory.
							const glm::ivec3 dc = toCellOrigin(n, cs);
							if (dc != srcCell) {
								const uint8_t nbDense = prev.cellState(dc);
								if (nbDense == kStateExterior || nbDense == kStateInterior) continue;
							}
							locNext.push_back(n);
						}
					}
					if (++batchProcessed >= kProgressBatch) {
						const uint64_t doneNow = intProcessed.fetch_add(batchProcessed,
																   std::memory_order_relaxed) + batchProcessed;
						batchProcessed = 0;
						const uint64_t now = core::TimeProvider::systemMillis();
						uint64_t prevTs = intLastLogMs.load(std::memory_order_relaxed);
						if (now - prevTs >= 5000u && intLastLogMs.compare_exchange_strong(prevTs, now)) {
							const double secs = (double)(now - intStart) / 1000.0;
							const double rate = secs > 0.001 ? (double)doneNow / secs : 0.0;
							const double etaSec = rate > 0.0 ? (double)(intTotalThis - doneNow) / rate : 0.0;
							Log::info("3dprint holefill sparse: round %d int scatter %d/%d (%.0f vox/s, eta %.0fs)",
									  round, doneNow, intTotalThis, rate, etaSec);
						}
					}
				}
				if (batchProcessed > 0) {
					intProcessed.fetch_add(batchProcessed, std::memory_order_relaxed);
				}
				fprintf(stderr, "[SPARSE-stderr] round %d INT chunk %d..%d slot=%d done\n",
						round, start, end, slot);
				fflush(stderr);
			});
			fprintf(stderr, "[SPARSE-stderr] round %d INT scatter all chunks done\n", round);
			fflush(stderr);
		}
		// Step 2b: SEQUENTIAL merge int results. An ext voxel solidified here
		// is removed from extVox; nextExt was already filtered against
		// solidified above, but we need to drop entries that just became
		// solid in this int merge -- they'd otherwise expand next round.
		const uint64_t intMergeStart = core::TimeProvider::systemMillis();
		size_t intSolidCount = 0, intNextCount = 0;
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			intSolidCount += perIntSolidify[(size_t)s].size();
			intNextCount += perNextInt[(size_t)s].size();
		}
		fprintf(stderr, "[SPARSE-stderr] round %d INT merge START solidify=%zu next=%zu\n",
				round, intSolidCount, intNextCount);
		fflush(stderr);
		Log::info("3dprint holefill sparse: round %d int merge starting (solidify=%zu next=%zu)",
				  round, intSolidCount, intNextCount);
		uint64_t intMergeLastLog = core::TimeProvider::systemMillis();
		size_t intMergeDone = 0;
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			for (const glm::ivec3 &n : perIntSolidify[(size_t)s]) {
				if (solidified.insert(n)) {
					extVox.erase(n);
					outHoleVoxels.push_back(n);
				}
				if ((++intMergeDone & 0xFFFFF) == 0) {
					const uint64_t now = core::TimeProvider::systemMillis();
					if (now - intMergeLastLog >= 3000u) {
						intMergeLastLog = now;
						Log::info("3dprint holefill sparse: round %d int merge solidify %zu/%zu [RSS=%.1f GB]",
								  round, intMergeDone, intSolidCount, rssGB());
					}
				}
			}
		}
		// Filter just-solidified out of nextExt (rare but possible: ext step
		// added n; int step then solidified n).
		size_t writeIdx = 0;
		for (size_t readIdx = 0; readIdx < nextExt.size(); ++readIdx) {
			if (!solidified.contains(nextExt[readIdx])) {
				nextExt[writeIdx++] = nextExt[readIdx];
			}
		}
		nextExt.resize(writeIdx);
		intMergeDone = 0;
		intMergeLastLog = core::TimeProvider::systemMillis();
		for (uint64_t s = 0; s < kBFSSlots; ++s) {
			for (const glm::ivec3 &n : perNextInt[(size_t)s]) {
				if (solidified.contains(n)) continue;
				if (intVox.insert(n)) {
					nextInt.push_back(n);
				}
				if ((++intMergeDone & 0xFFFFF) == 0) {
					const uint64_t now = core::TimeProvider::systemMillis();
					if (now - intMergeLastLog >= 3000u) {
						intMergeLastLog = now;
						Log::info("3dprint holefill sparse: round %d int merge next %zu/%zu (intVox=%zu) [RSS=%.1f GB]",
								  round, intMergeDone, intNextCount, intVox.size(), rssGB());
					}
				}
			}
		}
		Log::info("3dprint holefill sparse: round %d int merge done in %.1fs",
				  round, (double)(core::TimeProvider::systemMillis() - intMergeStart) / 1000.0);

		Log::info("3dprint holefill sparse: round %d ext+%zu int+%zu newHoles=%zu (round %.1fs / total %.1fs)",
				  round, nextExt.size(), nextInt.size(),
				  outHoleVoxels.size() - holesBefore,
				  (double)(core::TimeProvider::systemMillis() - roundStart) / 1000.0,
				  (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0);
		fprintf(stderr, "[SPARSE-stderr] round %d END nextExt=%zu nextInt=%zu newHoles=%zu round=%.1fs total=%.1fs\n",
				round, nextExt.size(), nextInt.size(), outHoleVoxels.size() - holesBefore,
				(double)(core::TimeProvider::systemMillis() - roundStart) / 1000.0,
				(double)(core::TimeProvider::systemMillis() - startMs) / 1000.0);
		fflush(stderr);

		extFrontier = core::move(nextExt);
		intFrontier = core::move(nextInt);
	}

	const double totalSecs = (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0;
	Log::info("3dprint holefill sparse: %zu hole voxel(s) found in %d round(s) (total %.1fs)",
			  outHoleVoxels.size(), round, totalSecs);
	fprintf(stderr, "[SPARSE-stderr] DONE %zu hole voxels in %d rounds total=%.1fs\n",
			outHoleVoxels.size(), round, totalSecs);
	fflush(stderr);
}

// Voxel-level plug pass.
//
// Catches sub-cell leaks the cs=N BFS can't see (1-voxel slits, narrow gaps
// inside `kStateSolid` cells). For each ext-frontier cs=N cell, we step into
// an adjacent solid cell at voxel granularity and BFS through air voxels.
//
// BFS rules:
//   - Walks only voxels that are air AND whose cs=N cell is `kStateSolid`,
//     `kStateEmpty`, or `kStateBlocked`. These are the "between the shells"
//     cells where leaks can live.
//   - When the walker sees a face neighbour voxel in a `kStateExterior` cell,
//     it sets `reachedExt = true` and does not enter. Same for interior.
//   - Capped at Manhattan distance kRadius from the start.
//
// Plug rule (after BFS finishes):
//   If both reachedExt and reachedInt are true, this is a leak. We then plug
//   only visited voxels whose face-neighbour cell is `kStateExterior` or
//   `kStateInterior` -- the shell-surface voxels. Voxels deep inside wall
//   thickness or in inter-shell gaps are NOT plugged: they preserve intentional
//   architecture (cavity walls etc.) while the actual missing-wall voxels at
//   the shell surfaces close the leak.
//
// Output positions are appended to outPlugVoxels in world voxel coordinates.
// Same fill machinery as `allHoleVoxels` handles them, including orphan-node
// creation when no existing node owns the position.
// Try to write a plug voxel live to the model during plug detection.
//   - Looks up the owning node via `lookup.solid` (closest adjacent solid cell).
//   - If found and the world position is inside that node's region: writes the
//     fill voxel directly via setVoxelUnsafe (lock-free; aligned 4-byte writes
//     are atomic on x86, and all threads write the SAME fill voxel value so
//     concurrent writes are idempotent), atomic-marks the node dirty, returns true.
//   - If not found, or position is outside node's region: returns false. Caller
//     adds the voxel to a per-thread orphan list for end-of-pass orphan-node
//     creation (which has to be done sequentially because scene-graph mutations
//     aren't thread-safe).
//
// Race tolerance: another thread may have already written this voxel; the second
// write is a no-op observable. `isAir` check skips the write when the voxel is
// already plugged, which removes the cost of repeated writes.
static bool tryWritePlugVoxelLive(const glm::ivec3 &worldVoxel,
                                    const Level &lookup,
                                    core::DynamicArray<NodeInfo> &nodes,
                                    std::atomic<uint8_t> *nodeDirty,
                                    int finalCellSize) {
	const glm::ivec3 holeCell = toCellOrigin(worldVoxel, finalCellSize);
	int bestNodeIdx = -1;
	int bestDistSq = INT_MAX;
	auto trySolidCell = [&](const glm::ivec3 &c) {
		auto it = lookup.solid.find(c);
		if (it == lookup.solid.end()) return;
		const int ni = (int)it->second;
		const glm::ivec3 local = transformPoint(nodes[(size_t)ni].invWorldMat, worldVoxel);
		const glm::ivec3 lo = nodes[(size_t)ni].rv->region().getLowerCorner();
		const glm::ivec3 hi = nodes[(size_t)ni].rv->region().getUpperCorner();
		const glm::ivec3 clamped = glm::clamp(local, lo, hi);
		const glm::ivec3 d = local - clamped;
		const int distSq = d.x * d.x + d.y * d.y + d.z * d.z;
		if (distSq < bestDistSq) { bestDistSq = distSq; bestNodeIdx = ni; }
	};
	trySolidCell(holeCell);
	for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
		trySolidCell(holeCell + off * finalCellSize);
	}
	if (bestNodeIdx < 0) return false;

	NodeInfo &ni = nodes[(size_t)bestNodeIdx];
	const glm::ivec3 localPos = transformPoint(ni.invWorldMat, worldVoxel);
	if (!ni.rv->region().containsPoint(localPos)) return false;

	// Skip if already solid (avoids redundant atomic writes).
	if (!voxel::isAir(ni.rv->voxel(localPos).getMaterial())) {
		// Treat as "already plugged" - count as success so caller doesn't add
		// to orphan list.
		return true;
	}

	const voxel::Voxel fillVoxel = voxel::createVoxel(voxel::VoxelType::Generic, ni.fillColorIdx);
	ni.rv->setVoxelUnsafe(localPos, fillVoxel);
	nodeDirty[bestNodeIdx].store(1, std::memory_order_relaxed);
	return true;
}

// Two-sided multi-source BFS plug pass.
//
// For each ext-frontier cs=2 cell:
//  1. Position a kCubeSize^3 cube centered on a start voxel near the wall.
//  2. Populate cube types per voxel:
//       EXT  = world voxel's enclosing cs=2 cell is kStateExterior
//       INT  = enclosing cell is kStateInterior
//       WALL = isSolidAt(world voxel) returns true
//       AIR  = anything else (sub-cell air in solid cell, gap voxel, etc.)
//  3. BFS-A multi-source from every EXT voxel, expanding only through AIR.
//     Records ext-side BFS depth in each visited AIR voxel.
//  4. BFS-B same from every INT voxel.
//  5. For each AIR voxel reached by both, totalDepth = extDepth + intDepth.
//     The minimum totalDepth across the cube = the bottleneck path length
//     (i.e. the shortest path from any ext voxel to any int voxel through
//     the air component the BFSes share).
//  6. Plug every AIR voxel that achieves the minimum totalDepth.
//
// Why this kills the splat: wall-parallel crawl voxels are reachable by
// BFS-A (they're on the ext side) but NOT by BFS-B (the wall blocks BFS-B
// from reaching them) -- their intDepth stays infinite, totalDepth is
// infinite, they're not in the minimum, they don't get plugged. Only
// genuine wall-piercing path voxels appear at the minimum.
static void runVoxelPlug(const Level &prev,
                          core::DynamicArray<NodeInfo> &nodes,
                          std::atomic<uint8_t> *nodeDirty,
                          const core::DynamicArray<size_t> &extFrontier,
                          core::DynamicArray<glm::ivec3> &outPlugVoxels,
                          int kRadius) {
	const int cs = prev.cellSize;
	const uint64_t startMs = core::TimeProvider::systemMillis();
	const int kCubeSize = 2 * kRadius + 1;
	const int kCubeVol = kCubeSize * kCubeSize * kCubeSize;
	fprintf(stderr, "[PLUG] starting two-sided BFS (cs=%d, radius=%d, cube=%d^3=%d voxels, frontier=%zu)\n",
			cs, kRadius, kCubeSize, kCubeVol, extFrontier.size());
	fflush(stderr);
	Log::info("3dprint plug: two-sided BFS (cs=%d, radius=%d, cube=%d^3, frontier=%zu)",
			  cs, kRadius, kCubeSize, extFrontier.size());

	// Cube type encoding (uint8_t):
	enum : uint8_t {
		kCubeAir  = 0,
		kCubeWall = 1,
		kCubeExt  = 2,
		kCubeInt  = 3,
	};
	static constexpr uint8_t kInfDepth = 255;

	constexpr int kSlots = 64;
	core::DynamicArray<core::DynamicArray<glm::ivec3>> perSlots;
	perSlots.resize(kSlots);
	// Reserve per-slot output to bounded growth ceiling. Each ext-frontier cell
	// produces at most ~kCubeVol plug voxels in the worst case (which would mean
	// the whole cube is on the bottleneck -- never realistic). Empirically a
	// real leak produces 1-15 plug voxels. Reserve frontier*4/kSlots gives
	// generous headroom without over-allocating.
	const size_t perSlotReserve = (size_t)glm::max(8192,
		(int)((int64_t)extFrontier.size() * 4 / kSlots));
	for (int s = 0; s < kSlots; ++s) {
		perSlots[(size_t)s].reserve(perSlotReserve);
	}
	Log::info("3dprint plug: pre-allocated %zu plug-voxel capacity per slot (%d slots, ~%.1f MB total)",
			  perSlotReserve, kSlots,
			  (double)(perSlotReserve * kSlots * sizeof(glm::ivec3)) / (1024.0 * 1024.0));

	std::atomic<int> slotCounter{0};
	std::atomic<int> leaksFound{0};
	std::atomic<int64_t> processed{0};
	std::atomic<uint64_t> lastLogMs{startMs};

	const int totalFrontier = (int)extFrontier.size();

	app::for_parallel(0, totalFrontier, [&](int start, int end) {
		const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % kSlots;
		core::DynamicArray<glm::ivec3> &localPlugs = perSlots[(size_t)slot];

		// Per-thread cube + depth arrays. Pre-allocated, reused per ext-frontier
		// cell. memset between cells is the only per-cell clear cost.
		core::DynamicArray<uint8_t> cubeType;
		cubeType.resize((size_t)kCubeVol);
		core::DynamicArray<uint8_t> extDepth;
		extDepth.resize((size_t)kCubeVol);
		core::DynamicArray<uint8_t> intDepth;
		intDepth.resize((size_t)kCubeVol);

		core::DynamicArray<int> bfsQueueA;
		bfsQueueA.reserve((size_t)kCubeVol);
		core::DynamicArray<int> bfsQueueB;
		bfsQueueB.reserve((size_t)kCubeVol);

		// Cube index helpers. Cube spans [-kRadius..+kRadius] relative to start.
		// Index = (z+R)*D*D + (y+R)*D + (x+R).
		auto cubeIdx = [&](int xRel, int yRel, int zRel) -> int {
			return (zRel + kRadius) * kCubeSize * kCubeSize
				 + (yRel + kRadius) * kCubeSize
				 + (xRel + kRadius);
		};
		// Inverse: given flat index, compute relative offset.
		auto cubeRel = [&](int idx) -> glm::ivec3 {
			const int x = (idx % kCubeSize) - kRadius;
			const int y = ((idx / kCubeSize) % kCubeSize) - kRadius;
			const int z = (idx / (kCubeSize * kCubeSize)) - kRadius;
			return glm::ivec3(x, y, z);
		};

		int64_t since = 0;
		for (int i = start; i < end; ++i) {
			if (g_rssCapTripped.load(std::memory_order_relaxed)) return;

			const size_t cellIdx = extFrontier[(size_t)i];
			const glm::ivec3 extCellOrigin = prev.toCell(cellIdx);

			// Pick a start voxel near the wall. Iterate face directions, find
			// first solid-or-blocked cell adjacent to this ext cell, place
			// start voxel on the shared face from the solid side.
			glm::ivec3 startVoxel(0);
			bool haveStart = false;
			for (const glm::ivec3 &dirOff : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nbCell = extCellOrigin + dirOff * cs;
				if (!prev.inBounds(nbCell)) continue;
				const uint8_t nbState = prev.state[prev.toIdx(nbCell)];
				if (nbState != kStateSolid && nbState != kStateBlocked) continue;
				glm::ivec3 sv = nbCell;
				sv.x += (dirOff.x > 0) ? 0 : (dirOff.x < 0 ? cs - 1 : cs / 2);
				sv.y += (dirOff.y > 0) ? 0 : (dirOff.y < 0 ? cs - 1 : cs / 2);
				sv.z += (dirOff.z > 0) ? 0 : (dirOff.z < 0 ? cs - 1 : cs / 2);
				if (isSolidAt(sv, prev, nodes)) continue;
				startVoxel = sv;
				haveStart = true;
				break;
			}
			if (!haveStart) {
				if ((++since & 0x3FF) == 0) {
					processed.fetch_add(since, std::memory_order_relaxed);
					since = 0;
				}
				continue;
			}

			// Populate cube. For each cube voxel, compute world position, look
			// up its cs=N cell's state, and decide type.
			memset(cubeType.data(), 0, (size_t)kCubeVol);
			memset(extDepth.data(), kInfDepth, (size_t)kCubeVol);
			memset(intDepth.data(), kInfDepth, (size_t)kCubeVol);
			bfsQueueA.clear();
			bfsQueueB.clear();

			for (int z = -kRadius; z <= kRadius; ++z) {
				for (int y = -kRadius; y <= kRadius; ++y) {
					for (int x = -kRadius; x <= kRadius; ++x) {
						const glm::ivec3 worldPos = startVoxel + glm::ivec3(x, y, z);
						const glm::ivec3 worldCell = toCellOrigin(worldPos, cs);
						const int idx = cubeIdx(x, y, z);
						uint8_t type;
						if (!prev.inBounds(worldCell)) {
							type = kCubeAir;
						} else {
							const uint8_t cellState = prev.state[prev.toIdx(worldCell)];
							if (cellState == kStateExterior) {
								type = kCubeExt;
								extDepth[(size_t)idx] = 0;
								bfsQueueA.push_back(idx);
							} else if (cellState == kStateInterior) {
								type = kCubeInt;
								intDepth[(size_t)idx] = 0;
								bfsQueueB.push_back(idx);
							} else {
								// kStateSolid / kStateEmpty / kStateBlocked.
								// Sub-cell solidity check via isSolidAt.
								// Optimization: only call isSolidAt for kStateSolid
								// cells (kStateEmpty has 0 solid voxels by definition).
								if (cellState == kStateSolid && isSolidAt(worldPos, prev, nodes)) {
									type = kCubeWall;
								} else {
									type = kCubeAir;
								}
							}
						}
						cubeType[(size_t)idx] = type;
					}
				}
			}

			// If no EXT or no INT in cube, can't have a leak. Skip BFS.
			if (bfsQueueA.empty() || bfsQueueB.empty()) {
				if ((++since & 0x3FF) == 0) {
					processed.fetch_add(since, std::memory_order_relaxed);
					since = 0;
				}
				continue;
			}

			// BFS-A from EXT voxels through AIR. Don't enter WALL or INT.
			// (BFS-A's purpose is to find air voxels reachable from ext.)
			{
				size_t qHead = 0;
				while (qHead < bfsQueueA.size()) {
					const int curIdx = bfsQueueA[qHead++];
					const glm::ivec3 rel = cubeRel(curIdx);
					const uint8_t curDepth = extDepth[(size_t)curIdx];
					if (curDepth == kInfDepth) continue;
					const uint8_t nextDepth = (curDepth < kInfDepth - 1) ? (uint8_t)(curDepth + 1) : kInfDepth;
					for (const glm::ivec3 &dir : voxel::arrayPathfinderFaces) {
						const glm::ivec3 nbRel = rel + dir;
						if (abs(nbRel.x) > kRadius || abs(nbRel.y) > kRadius || abs(nbRel.z) > kRadius) continue;
						const int nbIdx = cubeIdx(nbRel.x, nbRel.y, nbRel.z);
						const uint8_t nbType = cubeType[(size_t)nbIdx];
						if (nbType != kCubeAir) continue;  // walls / int / ext: don't enter
						if (extDepth[(size_t)nbIdx] <= nextDepth) continue;
						extDepth[(size_t)nbIdx] = nextDepth;
						bfsQueueA.push_back(nbIdx);
					}
				}
			}

			// BFS-B from INT voxels through AIR. Symmetric.
			{
				size_t qHead = 0;
				while (qHead < bfsQueueB.size()) {
					const int curIdx = bfsQueueB[qHead++];
					const glm::ivec3 rel = cubeRel(curIdx);
					const uint8_t curDepth = intDepth[(size_t)curIdx];
					if (curDepth == kInfDepth) continue;
					const uint8_t nextDepth = (curDepth < kInfDepth - 1) ? (uint8_t)(curDepth + 1) : kInfDepth;
					for (const glm::ivec3 &dir : voxel::arrayPathfinderFaces) {
						const glm::ivec3 nbRel = rel + dir;
						if (abs(nbRel.x) > kRadius || abs(nbRel.y) > kRadius || abs(nbRel.z) > kRadius) continue;
						const int nbIdx = cubeIdx(nbRel.x, nbRel.y, nbRel.z);
						const uint8_t nbType = cubeType[(size_t)nbIdx];
						if (nbType != kCubeAir) continue;
						if (intDepth[(size_t)nbIdx] <= nextDepth) continue;
						intDepth[(size_t)nbIdx] = nextDepth;
						bfsQueueB.push_back(nbIdx);
					}
				}
			}

			// Plug every AIR voxel reached by both BFSes. Splat-free because
			// wall-parallel crawl voxels are reached by BFS-A only (wall blocks
			// BFS-B), so they're filtered out automatically. Catches multiple
			// leak paths of any length (1-voxel slit, 5-voxel slit, cavity
			// walls, etc.) -- the previous "min totalDepth" rule missed longer
			// slits when shorter ones existed in the same cube.
			bool anyReached = false;
			int plugCountThisCube = 0;
			for (int idx = 0; idx < kCubeVol; ++idx) {
				if (cubeType[(size_t)idx] != kCubeAir) continue;
				const uint8_t e = extDepth[(size_t)idx];
				const uint8_t in = intDepth[(size_t)idx];
				if (e == kInfDepth || in == kInfDepth) continue;
				anyReached = true;
				const glm::ivec3 rel = cubeRel(idx);
				const glm::ivec3 worldVoxel = startVoxel + rel;
				// Shell-adjacency filter: only plug voxels with at least one
				// face-neighbour in a kStateExterior or kStateInterior cell.
				// These are voxels at the actual shell surface where the leak
				// crosses from air-space into wall-material -- the ones that
				// need solidifying to seal. Middle-of-wall and middle-of-gap
				// voxels (no shell face-neighbours) become sealed pockets, which
				// is fine for watertightness. Eliminates cavity-wall gap splat
				// and middle-wall splat without missing the entry/exit points
				// of any leak path.
				bool hasShellFaceNb = false;
				for (const glm::ivec3 &dir : voxel::arrayPathfinderFaces) {
					const glm::ivec3 nb = worldVoxel + dir;
					const glm::ivec3 nbCell = toCellOrigin(nb, cs);
					if (!prev.inBounds(nbCell)) continue;
					const uint8_t nbState = prev.state[prev.toIdx(nbCell)];
					if (nbState == kStateExterior || nbState == kStateInterior) {
						hasShellFaceNb = true;
						break;
					}
				}
				if (!hasShellFaceNb) continue;
				// Live write: try to write the plug voxel directly into the model.
				// If successful, subsequent threads' isSolidAt sees the new wall
				// and BFS will skip it. If no owning node exists, fall back to
				// the orphan list (handled by applyHoleFills after detection).
				if (!tryWritePlugVoxelLive(worldVoxel, prev, nodes, nodeDirty, cs)) {
					localPlugs.push_back(worldVoxel);
				}
				++plugCountThisCube;
			}
			if (!anyReached) {
				// No AIR voxel reachable by both BFSes -> no leak.
				if ((++since & 0x3FF) == 0) {
					processed.fetch_add(since, std::memory_order_relaxed);
					since = 0;
				}
				continue;
			}
			leaksFound.fetch_add(1, std::memory_order_relaxed);
			(void)plugCountThisCube;  // future heartbeat enrichment

			// Heartbeat every 1024 outer iterations.
			if ((++since & 0x3FF) == 0) {
				const int64_t globalDone = processed.fetch_add(since, std::memory_order_relaxed) + since;
				since = 0;
				const uint64_t now = core::TimeProvider::systemMillis();
				uint64_t prevMs = lastLogMs.load(std::memory_order_relaxed);
				if (now - prevMs >= 3000u && lastLogMs.compare_exchange_strong(prevMs, now)) {
					fprintf(stderr, "[PLUG] %lld/%d ext cells (%.1f%%) leaks=%d elapsed=%.1fs [RSS=%.1f GB]\n",
							(long long)globalDone, totalFrontier,
							100.0 * (double)globalDone / (double)totalFrontier,
							leaksFound.load(std::memory_order_relaxed),
							(double)(now - startMs) / 1000.0, rssGB());
					fflush(stderr);
					checkRSSCap("plug heartbeat");
				}
			}
		}
		processed.fetch_add(since, std::memory_order_relaxed);
	});

	// Merge per-slot plug lists into outPlugVoxels with dedup.
	std::unordered_set<glm::ivec3, glm::hash<glm::ivec3>> uniquePlugs;
	size_t totalRaw = 0;
	for (int s = 0; s < kSlots; ++s) totalRaw += perSlots[(size_t)s].size();
	uniquePlugs.reserve(totalRaw);
	for (int s = 0; s < kSlots; ++s) {
		for (const glm::ivec3 &v : perSlots[(size_t)s]) {
			if (uniquePlugs.insert(v).second) {
				outPlugVoxels.push_back(v);
			}
		}
	}

	const double totalSecs = (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0;
	Log::info("3dprint plug: done in %.1fs (%d leaks, %zu raw -> %zu unique plug voxels)",
			  totalSecs, leaksFound.load(), totalRaw, outPlugVoxels.size());
}

// Single-source BFS plug pass with parent tracking (option C, drop-in alternative
// to runVoxelPlug above). Same signature, simpler algorithm:
//
//   For each ext-frontier cs=2 cell:
//     1. Pick start voxel near wall.
//     2. Single BFS through air voxels, recording parent direction per visited
//        voxel (1 byte: which direction to walk back to reach the parent).
//        BFS stops at first kStateInterior contact.
//     3. Walk parents from contact voxel back to source -- gives the shortest
//        path from THIS source to the FIRST int contact.
//     4. Plug every voxel on that path (1-voxel-thick line, no width).
//
// Trade-offs vs the two-sided BFS version:
//   - Cheaper per cell (one BFS, no cube populate).
//   - Only finds optimal path FROM ITS SPECIFIC SOURCE -- multiple
//     ext-frontier cells reach the same hole via different paths, so the total
//     plug volume is line_length * num_cells_seeing_this_hole. Lines stack.
//   - Two-sided version finds the global bottleneck and plugs only that.
static void runVoxelPlugSinglePath(const Level &prev,
                                     core::DynamicArray<NodeInfo> &nodes,
                                     std::atomic<uint8_t> *nodeDirty,
                                     const core::DynamicArray<size_t> &extFrontier,
                                     core::DynamicArray<glm::ivec3> &outPlugVoxels,
                                     int kRadius) {
	const int cs = prev.cellSize;
	const uint64_t startMs = core::TimeProvider::systemMillis();
	fprintf(stderr, "[PLUG-C] starting single-source path tracking (cs=%d, radius=%d, frontier=%zu)\n",
			cs, kRadius, extFrontier.size());
	fflush(stderr);
	Log::info("3dprint plug: single-source path (cs=%d, radius=%d, frontier=%zu)",
			  cs, kRadius, extFrontier.size());

	// Direction encoding: 0 = no parent (source), 1-6 = direction to walk back to parent.
	// Index into kDirs.
	static const glm::ivec3 kDirs[7] = {
		glm::ivec3(0,0,0),
		glm::ivec3(-1,0,0), glm::ivec3(1,0,0),
		glm::ivec3(0,-1,0), glm::ivec3(0,1,0),
		glm::ivec3(0,0,-1), glm::ivec3(0,0,1),
	};
	// For an outward direction d (1-6), the back-direction code:
	// 1 (-X step) -> 2 (parent is at +X): no, let me think again.
	// If we step from cur to nb via direction d (nb = cur + kDirs[d]),
	// then to go back from nb to cur we need to walk in -d direction.
	// Encoding: parentDir[nb] = the code for -d. For d=1 (-X), -d = +X = code 2.
	static const uint8_t kReverseDir[7] = {0, 2, 1, 4, 3, 6, 5};

	constexpr int kSlots = 64;
	core::DynamicArray<core::DynamicArray<glm::ivec3>> perSlots;
	perSlots.resize(kSlots);
	const size_t perSlotReserve = (size_t)glm::max(8192,
		(int)((int64_t)extFrontier.size() * 4 / kSlots));
	for (int s = 0; s < kSlots; ++s) {
		perSlots[(size_t)s].reserve(perSlotReserve);
	}
	Log::info("3dprint plug: pre-allocated %zu plug-voxel capacity per slot (%d slots, ~%.1f MB total)",
			  perSlotReserve, kSlots,
			  (double)(perSlotReserve * kSlots * sizeof(glm::ivec3)) / (1024.0 * 1024.0));

	std::atomic<int> slotCounter{0};
	std::atomic<int> leaksFound{0};
	std::atomic<int64_t> processed{0};
	std::atomic<uint64_t> lastLogMs{startMs};

	const int totalFrontier = (int)extFrontier.size();

	app::for_parallel(0, totalFrontier, [&](int start, int end) {
		const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % kSlots;
		core::DynamicArray<glm::ivec3> &localPlugs = perSlots[(size_t)slot];

		const int kDiam = 2 * kRadius + 1;
		const int kFlatSize = kDiam * kDiam * kDiam;
		core::DynamicArray<uint8_t> visitedFlat;
		visitedFlat.resize((size_t)kFlatSize);
		core::DynamicArray<uint8_t> parentDir;
		parentDir.resize((size_t)kFlatSize);
		core::DynamicArray<glm::ivec3> bfsQueue;
		bfsQueue.reserve((size_t)kFlatSize);

		auto flatIdx = [&](const glm::ivec3 &rel) -> int {
			return (rel.z + kRadius) * kDiam * kDiam
				 + (rel.y + kRadius) * kDiam
				 + (rel.x + kRadius);
		};

		int64_t since = 0;
		for (int i = start; i < end; ++i) {
			if (g_rssCapTripped.load(std::memory_order_relaxed)) return;

			const size_t cellIdx = extFrontier[(size_t)i];
			const glm::ivec3 extCellOrigin = prev.toCell(cellIdx);

			bool foundLeak = false;
			for (const glm::ivec3 &dirOff : voxel::arrayPathfinderFaces) {
				if (foundLeak) break;
				const glm::ivec3 nbCell = extCellOrigin + dirOff * cs;
				if (!prev.inBounds(nbCell)) continue;
				const uint8_t nbState = prev.state[prev.toIdx(nbCell)];
				if (nbState != kStateSolid && nbState != kStateBlocked) continue;

				glm::ivec3 startVoxel = nbCell;
				startVoxel.x += (dirOff.x > 0) ? 0 : (dirOff.x < 0 ? cs - 1 : cs / 2);
				startVoxel.y += (dirOff.y > 0) ? 0 : (dirOff.y < 0 ? cs - 1 : cs / 2);
				startVoxel.z += (dirOff.z > 0) ? 0 : (dirOff.z < 0 ? cs - 1 : cs / 2);
				if (isSolidAt(startVoxel, prev, nodes)) continue;

				memset(visitedFlat.data(), 0, (size_t)kFlatSize);
				memset(parentDir.data(), 0, (size_t)kFlatSize);
				bfsQueue.clear();
				visitedFlat[(size_t)flatIdx(glm::ivec3(0))] = 1;
				bfsQueue.push_back(startVoxel);

				glm::ivec3 contactVoxel(0);
				bool foundContact = false;
				size_t qHead = 0;
				while (qHead < bfsQueue.size() && !foundContact) {
					const glm::ivec3 cur = bfsQueue[qHead++];
					const glm::ivec3 rel = cur - startVoxel;
					const int dist = abs(rel.x) + abs(rel.y) + abs(rel.z);
					if (dist >= kRadius) continue;

					for (int d = 1; d <= 6 && !foundContact; ++d) {
						const glm::ivec3 nb = cur + kDirs[d];
						const glm::ivec3 nbCellOrigin = toCellOrigin(nb, cs);
						if (!prev.inBounds(nbCellOrigin)) continue;
						const uint8_t nbCellState = prev.state[prev.toIdx(nbCellOrigin)];

						if (nbCellState == kStateInterior) {
							// BFS reached int. Path = source -> ... -> cur.
							contactVoxel = cur;
							foundContact = true;
							break;
						}
						if (nbCellState == kStateExterior) continue;

						const glm::ivec3 nbRel = nb - startVoxel;
						if (abs(nbRel.x) > kRadius || abs(nbRel.y) > kRadius || abs(nbRel.z) > kRadius) continue;
						const int fi = flatIdx(nbRel);
						if (visitedFlat[(size_t)fi] != 0) continue;
						if (isSolidAt(nb, prev, nodes)) continue;

						visitedFlat[(size_t)fi] = 1;
						parentDir[(size_t)fi] = kReverseDir[d];
						bfsQueue.push_back(nb);
					}
				}

				if (foundContact) {
					leaksFound.fetch_add(1, std::memory_order_relaxed);
					// Walk parents from contact back to source. Plug each path voxel.
					glm::ivec3 cur = contactVoxel;
					int safetyCounter = kFlatSize;
					while (--safetyCounter > 0) {
						// Shell-adjacency filter (same as option E): only plug path
						// voxels at the actual shell surface (face-neighbour cell
						// classified ext or int). Skips middle-of-wall and middle-
						// of-gap path voxels.
						bool hasShellFaceNb = false;
						for (const glm::ivec3 &dir2 : voxel::arrayPathfinderFaces) {
							const glm::ivec3 nb2 = cur + dir2;
							const glm::ivec3 nbCell = toCellOrigin(nb2, cs);
							if (!prev.inBounds(nbCell)) continue;
							const uint8_t nbState = prev.state[prev.toIdx(nbCell)];
							if (nbState == kStateExterior || nbState == kStateInterior) {
								hasShellFaceNb = true;
								break;
							}
						}
						if (hasShellFaceNb) {
							// Live write to model (lock-free); fall back to orphan list
							// if no owning node exists for this voxel.
							if (!tryWritePlugVoxelLive(cur, prev, nodes, nodeDirty, cs)) {
								localPlugs.push_back(cur);
							}
						}
						const glm::ivec3 rel = cur - startVoxel;
						if (rel.x == 0 && rel.y == 0 && rel.z == 0) break;
						const int fi = flatIdx(rel);
						const uint8_t d = parentDir[(size_t)fi];
						if (d == 0) break;
						cur = cur + kDirs[d];
					}
					foundLeak = true;
				}
			}

			if ((++since & 0x3FF) == 0) {
				const int64_t globalDone = processed.fetch_add(since, std::memory_order_relaxed) + since;
				since = 0;
				const uint64_t now = core::TimeProvider::systemMillis();
				uint64_t prevMs = lastLogMs.load(std::memory_order_relaxed);
				if (now - prevMs >= 3000u && lastLogMs.compare_exchange_strong(prevMs, now)) {
					fprintf(stderr, "[PLUG-C] %lld/%d ext cells (%.1f%%) leaks=%d elapsed=%.1fs [RSS=%.1f GB]\n",
							(long long)globalDone, totalFrontier,
							100.0 * (double)globalDone / (double)totalFrontier,
							leaksFound.load(std::memory_order_relaxed),
							(double)(now - startMs) / 1000.0, rssGB());
					fflush(stderr);
					checkRSSCap("plug heartbeat");
				}
			}
		}
		processed.fetch_add(since, std::memory_order_relaxed);
	});

	std::unordered_set<glm::ivec3, glm::hash<glm::ivec3>> uniquePlugs;
	size_t totalRaw = 0;
	for (int s = 0; s < kSlots; ++s) totalRaw += perSlots[(size_t)s].size();
	uniquePlugs.reserve(totalRaw);
	for (int s = 0; s < kSlots; ++s) {
		for (const glm::ivec3 &v : perSlots[(size_t)s]) {
			if (uniquePlugs.insert(v).second) {
				outPlugVoxels.push_back(v);
			}
		}
	}

	const double totalSecs = (double)(core::TimeProvider::systemMillis() - startMs) / 1000.0;
	Log::info("3dprint plug: done in %.1fs (%d leaks, %zu raw -> %zu unique plug voxels)",
			  totalSecs, leaksFound.load(), totalRaw, outPlugVoxels.size());
}

} // namespace

void runFaceClassify(SceneManager *sceneMgr, int minCellSize) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	core::DynamicArray<NodeInfo> nodes;
	nodes.reserve((size_t)graph.size());
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) continue;
		NodeInfo info;
		info.nodeId = node.id();
		info.rv = rv;
		info.worldMat = graph.worldMatrix(node, 0);
		info.invWorldMat = glm::inverse(info.worldMat);
		info.cellOrigin = glm::ivec3(0);
		nodes.push_back(info);
	}
	if (nodes.empty()) {
		Log::info("3dprint faceclassify: no model nodes");
		return;
	}

	const int totalNodes = (int)nodes.size();
	int64_t legendOuter = 0, legendInner = 0, legendThin = 0, legendBuried = 0;
	{
	ProgressTimer timer("faceclassify", totalNodes);

	// -----------------------------------------------------------------------
	// Step 1: Infer coarse cell size, world bbox, coarse solid hash + grid.
	// -----------------------------------------------------------------------
	int coarseCellSize = 0;
	{
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes) {
			widthCount[ni.rv->region().getWidthInVoxels()]++;
		}
		int bestCount = 0;
		for (const auto &kv : widthCount) {
			if (kv.second > bestCount) { bestCount = kv.second; coarseCellSize = kv.first; }
		}
	}
	if (coarseCellSize <= 0) {
		Log::error("3dprint faceclassify: could not determine cell size -- run 3dprint regrid first");
		return;
	}

	// Validate and snap minCellSize
	if (minCellSize > 0) {
		if (minCellSize >= coarseCellSize) {
			Log::warn("3dprint faceclassify: minCellSize %d >= coarse %d -- refinement disabled",
					  minCellSize, coarseCellSize);
			minCellSize = 0;
		} else {
			uint64_t s = coarseCellSize;
			bool valid = false;
			while (s > 0) { if (s == minCellSize) { valid = true; break; } s /= 2; }
			if (!valid) {
				s = coarseCellSize;
				while (s / 2 >= minCellSize) s /= 2;
				minCellSize = s;
				Log::info("3dprint faceclassify: snapped minCellSize to %d", minCellSize);
			}
		}
	}

	voxel::Region worldBbox = voxel::Region::InvalidRegion;
	Level coarse;
	coarse.cellSize = coarseCellSize;
	coarse.solid.reserve((size_t)totalNodes);

	for (int i = 0; i < totalNodes; ++i) {
		NodeInfo &ni = nodes[i];
		const voxel::Region &r = ni.rv->region();
		const glm::ivec3 corners[8] = {
			{r.getLowerX(), r.getLowerY(), r.getLowerZ()}, {r.getUpperX(), r.getLowerY(), r.getLowerZ()},
			{r.getLowerX(), r.getUpperY(), r.getLowerZ()}, {r.getUpperX(), r.getUpperY(), r.getLowerZ()},
			{r.getLowerX(), r.getLowerY(), r.getUpperZ()}, {r.getUpperX(), r.getLowerY(), r.getUpperZ()},
			{r.getLowerX(), r.getUpperY(), r.getUpperZ()}, {r.getUpperX(), r.getUpperY(), r.getUpperZ()},
		};
		for (const glm::ivec3 &c : corners) {
			const glm::ivec3 wc = transformPoint(ni.worldMat, c);
			if (worldBbox.isValid()) worldBbox.accumulate(wc); else worldBbox = voxel::Region(wc, wc);
		}
		ni.cellOrigin = toCellOrigin(transformPoint(ni.worldMat, r.getLowerCorner()), coarseCellSize);
		coarse.solid.emplace(ni.cellOrigin, i);
	}
	{
		const uint64_t col = (uint64_t)nodes.size() - (uint64_t)coarse.solid.size();
		if (col > 0) Log::warn("3dprint faceclassify: %d node(s) share a snapped cell origin", col);
	}

	glm::ivec3 gridLower(INT_MAX, INT_MAX, INT_MAX);
	glm::ivec3 gridUpper(INT_MIN, INT_MIN, INT_MIN);
	for (const NodeInfo &ni : nodes) {
		gridLower = glm::min(gridLower, ni.cellOrigin);
		gridUpper = glm::max(gridUpper, ni.cellOrigin);
	}
	// Pad by one cell on each side so boundary seeding never touches interior air cells
	gridLower -= glm::ivec3(coarseCellSize);
	gridUpper += glm::ivec3(coarseCellSize);
	coarse.initGrid(gridLower, gridUpper);
	// Mark solid cells in the state array
	for (const auto &kv : coarse.solid) {
		if (coarse.inBounds(kv.first)) {
			coarse.state[(size_t)coarse.toIdx(kv.first)] = 255; // occupied slot (not exterior/interior)
		}
	}

	Log::info("3dprint faceclassify: cell size %d, bbox %d x %d x %d, grid %d x %d x %d = %d cells, solid=%d",
			  coarseCellSize, worldBbox.getWidthInVoxels(), worldBbox.getHeightInVoxels(), worldBbox.getDepthInVoxels(),
			  coarse.dimX, coarse.dimY, coarse.dimZ, coarse.dimX * coarse.dimY * coarse.dimZ,
			  (uint64_t)coarse.solid.size());

	// -----------------------------------------------------------------------
	// Step 2: Coarse exterior + interior.
	// -----------------------------------------------------------------------
	{
		// Reset state (solid cells get a placeholder != kStateEmpty so BFS skips them)
		coarse.state.assign((size_t)coarse.dimX * coarse.dimY * coarse.dimZ, kStateEmpty);
		for (const auto &kv : coarse.solid) {
			if (coarse.inBounds(kv.first)) coarse.state[(size_t)coarse.toIdx(kv.first)] = kStateSolid;
		}

		uint64_t t = core::TimeProvider::systemMillis();
		Level emptyPrev; // no previous level at coarse
		const uint64_t hole = buildExterior(coarse, emptyPrev);
		(void)hole;
		const uint64_t extCount = countCellsByState(coarse, kStateExterior);
		Log::info("3dprint faceclassify: coarse exterior: %d cells (%.2fs)", extCount, elapsedSince(t));

		t = core::TimeProvider::systemMillis();
		{
			const uint64_t intCount = buildInteriorAllSeeds(coarse);
			if (intCount == 0) {
				Log::warn("3dprint faceclassify: no enclosed interior at cell size %d -- model may need sealing",
						  coarseCellSize);
			} else {
				Log::info("3dprint faceclassify: coarse interior: %d cells (%.2fs)", intCount, elapsedSince(t));
			}
		}
	}

	// -----------------------------------------------------------------------
	// Step 3: Refinement levels.
	// -----------------------------------------------------------------------
	core::DynamicArray<Level> fineLevels;

	if (minCellSize > 0 && minCellSize < coarseCellSize) {
		int fineSize = coarseCellSize / 2;
		Level *prevLevel = &coarse;

		while (fineSize >= minCellSize) {
			fineLevels.push_back(Level{});
			Level &lvl = fineLevels[fineLevels.size() - 1];
			lvl.cellSize = fineSize;
			lvl.initGrid(gridLower, gridUpper);

			Log::info("3dprint faceclassify: refinement level %d (grid %d x %d x %d = %d cells) ...",
					  fineSize, lvl.dimX, lvl.dimY, lvl.dimZ, lvl.dimX * lvl.dimY * lvl.dimZ);

			uint64_t t = core::TimeProvider::systemMillis();
			buildSolidHash(lvl, nodes);
			Log::info("3dprint faceclassify:   level %d solid cells: %d (%.2fs)",
					  fineSize, (uint64_t)lvl.solid.size(), elapsedSince(t));

			// Mark solid in state array so BFS skips them
			for (const auto &kv : lvl.solid) {
				if (lvl.inBounds(kv.first)) lvl.state[(size_t)lvl.toIdx(kv.first)] = kStateSolid;
			}

			t = core::TimeProvider::systemMillis();
			const uint64_t hole = buildExterior(lvl, *prevLevel);
			const uint64_t extCount = countCellsByState(lvl, kStateExterior);
			Log::info("3dprint faceclassify:   level %d exterior: %d cells (%.2fs)",
					  fineSize, extCount, elapsedSince(t));
			if (hole > 0) {
				Log::warn("3dprint faceclassify:   HOLE DETECTED at cell size %d -- largest hole >= %d voxels. "
						  "Stopping refinement (using level %d as finest).",
						  fineSize, fineSize, prevLevel->cellSize);
				fineLevels.resize(fineLevels.size() - 1); // discard broken level -- exterior is unreliable
				break; // skip interior BFS on broken level, use previous level as finest
			}

			t = core::TimeProvider::systemMillis();
			{
				const uint64_t intCount = buildInteriorAllSeeds(lvl);
				if (intCount == 0) {
					Log::warn("3dprint faceclassify:   level %d: no enclosed interior (%.2fs)",
							  fineSize, elapsedSince(t));
				} else {
					Log::info("3dprint faceclassify:   level %d interior: %d cells (%.2fs)",
							  fineSize, intCount, elapsedSince(t));
				}
			}

			prevLevel = &lvl;
			fineSize /= 2;
		}
	}

	// Build level stack: finest first, coarse last (for ray cast checks)
	core::DynamicArray<Level *> levelStack;
	// Reverse loop MUST be int (signed): unsigned `i >= 0` is always true,
	// after i decrements past 0 it wraps to SIZE_MAX -> infinite loop.
	for (int i = (int)fineLevels.size() - 1; i >= 0; --i) levelStack.push_back(&fineLevels[i]);
	levelStack.push_back(&coarse);
	const Level &finestLevel = *levelStack[0];

	// -----------------------------------------------------------------------
	// Step 4: Parallel ray cast classification per node.
	// -----------------------------------------------------------------------
	core::DynamicArray<core::DynamicArray<ClassifyResult>> results;
	results.resize((size_t)totalNodes);
	std::atomic<uint64_t> processed{0};

	app::for_parallel(0, totalNodes, [&](int start, int end) {
		for (int i = start; i < end; ++i) {
			const NodeInfo &ni = nodes[i];
			const voxel::Region &region = ni.rv->region();
			core::DynamicArray<ClassifyResult> &nodeResults = results[i];

			for (int z = region.getLowerZ(); z <= region.getUpperZ(); ++z) {
				for (int y = region.getLowerY(); y <= region.getUpperY(); ++y) {
					for (int x = region.getLowerX(); x <= region.getUpperX(); ++x) {
						if (voxel::isAir(ni.rv->voxel(x, y, z).getMaterial())) continue;

						const glm::ivec3 worldPos = transformPoint(ni.worldMat, glm::ivec3(x, y, z));
						bool facesOuter = false;
						bool facesInner = false;

						for (const glm::ivec3 &faceOff : voxel::arrayPathfinderFaces) {
							glm::ivec3 rayPos = worldPos + faceOff;
							while (true) {
								// cellState returns kStateExterior for out-of-bounds positions,
								// so no explicit bbox check needed -- the ray self-terminates.
								bool classified = false;
								for (Level *lvl : levelStack) {
									const uint8_t s = lvl->cellState(toCellOrigin(rayPos, lvl->cellSize));
									if (s == kStateExterior) { facesOuter = true; classified = true; break; }
									if (s == kStateInterior) { facesInner = true; classified = true; break; }
								}
								if (classified) break;

								// Solid check at finest available level
								if (isSolidAt(rayPos, finestLevel, nodes)) break;

								rayPos += faceOff;
							}
							if (facesOuter && facesInner) break;
						}

						uint8_t tag;
						if      (facesOuter && facesInner) tag = kTagThin;
						else if (facesOuter)               tag = kTagOuter;
						else if (facesInner)               tag = kTagInner;
						else                               tag = kTagBuried;
						nodeResults.push_back({glm::ivec3(x, y, z), tag});
					}
				}
			}
			timer.addVoxels((int64_t)nodeResults.size());
			timer.tick(++processed);
		}
	});

	// -----------------------------------------------------------------------
	// Step 5: Apply recoloring (sequential).
	// -----------------------------------------------------------------------
	static constexpr color::RGBA kOuterColor(255, 140,   0, 255);
	static constexpr color::RGBA kInnerColor( 30, 120, 255, 255);
	static constexpr color::RGBA kThinColor (220,   0, 220, 255);
	static constexpr color::RGBA kBuriedColor(255, 230,  0, 255);

	uint64_t totalClassified = 0, nodesTouched = 0;
	for (int i = 0; i < totalNodes; ++i) {
		const core::DynamicArray<ClassifyResult> &nodeResults = results[i];
		if (nodeResults.empty()) continue;
		scenegraph::SceneGraphNode &node = graph.node(nodes[i].nodeId);
		palette::Palette &pal = node.palette();
		uint8_t outerIdx = 0, innerIdx = 0, thinIdx = 0, buriedIdx = 0;
		pal.tryAdd(kOuterColor,  true, &outerIdx,  true);
		pal.tryAdd(kInnerColor,  true, &innerIdx,  true);
		pal.tryAdd(kThinColor,   true, &thinIdx,   true);
		pal.tryAdd(kBuriedColor, true, &buriedIdx, true);
		voxel::RawVolumeWrapper wrapper(nodes[i].rv);
		for (const ClassifyResult &r : nodeResults) {
			const voxel::Voxel &orig = nodes[i].rv->voxel(r.localPos);
			uint8_t colorIdx;
			switch (r.tag) {
			case kTagOuter:  colorIdx = outerIdx;  ++legendOuter;  break;
			case kTagInner:  colorIdx = innerIdx;  ++legendInner;  break;
			case kTagThin:   colorIdx = thinIdx;   ++legendThin;   break;
			default:         colorIdx = buriedIdx; ++legendBuried; break;
			}
			wrapper.setVoxel(r.localPos, voxel::createVoxel(orig.getMaterial(), colorIdx,
															  orig.getNormal(), orig.getFlags(), orig.getBoneIdx()));
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(nodes[i].nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
		totalClassified += (uint64_t)nodeResults.size();
	}
	} // ProgressTimer destructs here

	Log::info("3dprint faceclassify: color legend and voxel counts:");
	Log::info("  ORANGE  (255,140,  0) = outer surface -- face sees exterior space       : %ld voxels", (long)legendOuter);
	Log::info("  BLUE    ( 30,120,255) = inner surface -- face sees interior space only  : %ld voxels", (long)legendInner);
	Log::info("  MAGENTA (220,  0,220) = thin wall     -- face sees both ext and interior: %ld voxels", (long)legendThin);
	Log::info("  YELLOW  (255,230,  0) = buried solid  -- no air face at all             : %ld voxels", (long)legendBuried);
}

void runHoleMap(SceneManager *sceneMgr, int minCellSize) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	core::DynamicArray<NodeInfo> nodes;
	nodes.reserve((size_t)graph.size());
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) continue;
		NodeInfo info;
		info.nodeId = node.id();
		info.rv = rv;
		info.worldMat = graph.worldMatrix(node, 0);
		info.invWorldMat = glm::inverse(info.worldMat);
		info.cellOrigin = glm::ivec3(0);
		nodes.push_back(info);
	}
	if (nodes.empty()) {
		Log::info("3dprint holemap: no model nodes");
		return;
	}
	const int totalNodes = (int)nodes.size();

	// Infer coarse cell size from modal node width
	int coarseCellSize = 0;
	{
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes)
			widthCount[ni.rv->region().getWidthInVoxels()]++;
		int bestCount = 0;
		for (const auto &kv : widthCount)
			if (kv.second > bestCount) { bestCount = kv.second; coarseCellSize = kv.first; }
	}
	if (coarseCellSize <= 0) {
		Log::error("3dprint holemap: could not determine cell size -- run 3dprint regrid first");
		return;
	}
	if (minCellSize <= 0 || minCellSize >= coarseCellSize) {
		minCellSize = coarseCellSize / 2;
	}
	if (minCellSize <= 0) {
		Log::error("3dprint holemap: coarse cell size %d too small to subdivide", coarseCellSize);
		return;
	}
	Log::info("3dprint holemap: coarse=%d, refining down to minCellSize=%d, nodes=%d",
			  coarseCellSize, minCellSize, totalNodes);

	// World bbox and coarse solid hash (one cell per node at coarse level)
	voxel::Region worldBbox = voxel::Region::InvalidRegion;
	Level coarse;
	coarse.cellSize = coarseCellSize;
	coarse.solid.reserve((size_t)totalNodes);
	glm::ivec3 gridLower(INT_MAX, INT_MAX, INT_MAX);
	glm::ivec3 gridUpper(INT_MIN, INT_MIN, INT_MIN);

	for (int i = 0; i < totalNodes; ++i) {
		NodeInfo &ni = nodes[i];
		const voxel::Region &r = ni.rv->region();
		const glm::ivec3 corners[8] = {
			{r.getLowerX(), r.getLowerY(), r.getLowerZ()}, {r.getUpperX(), r.getLowerY(), r.getLowerZ()},
			{r.getLowerX(), r.getUpperY(), r.getLowerZ()}, {r.getUpperX(), r.getUpperY(), r.getLowerZ()},
			{r.getLowerX(), r.getLowerY(), r.getUpperZ()}, {r.getUpperX(), r.getLowerY(), r.getUpperZ()},
			{r.getLowerX(), r.getUpperY(), r.getUpperZ()}, {r.getUpperX(), r.getUpperY(), r.getUpperZ()},
		};
		for (const glm::ivec3 &c : corners) {
			const glm::ivec3 wc = transformPoint(ni.worldMat, c);
			if (worldBbox.isValid()) worldBbox.accumulate(wc); else worldBbox = voxel::Region(wc, wc);
		}
		ni.cellOrigin = toCellOrigin(transformPoint(ni.worldMat, r.getLowerCorner()), coarseCellSize);
		gridLower = glm::min(gridLower, ni.cellOrigin);
		gridUpper = glm::max(gridUpper, ni.cellOrigin);
		// Only add to coarse solid hash if node has at least one solid voxel.
		// Fully empty nodes left out so they can seed interior rooms after exterior BFS.
		bool hasSolid = false;
		for (int z = r.getLowerZ(); z <= r.getUpperZ() && !hasSolid; ++z)
			for (int y = r.getLowerY(); y <= r.getUpperY() && !hasSolid; ++y)
				for (int x = r.getLowerX(); x <= r.getUpperX() && !hasSolid; ++x)
					if (!voxel::isAir(ni.rv->voxel(x, y, z).getMaterial()))
						hasSolid = true;
		if (hasSolid) {
			coarse.solid.emplace(ni.cellOrigin, i);
		}
	}

	gridLower -= glm::ivec3(coarseCellSize);
	gridUpper += glm::ivec3(coarseCellSize);
	coarse.initGrid(gridLower, gridUpper);
	coarse.state.assign((size_t)coarse.dimX * coarse.dimY * coarse.dimZ, kStateEmpty);
	for (const auto &kv : coarse.solid)
		if (coarse.inBounds(kv.first)) coarse.state[(size_t)coarse.toIdx(kv.first)] = kStateSolid;

	// Coarse exterior + interior BFS (full flood, works at node scale)
	{
		Level emptyPrev;
		buildExterior(coarse, emptyPrev);
	}
	{
		const uint64_t intCount = buildInteriorAllSeeds(coarse);
		if (intCount == 0) {
			Log::warn("3dprint holemap: no enclosed interior at coarse scale -- model may not be sealed");
			return;
		}
		Log::info("3dprint holemap: coarse grid %dx%dx%d -- exterior=%d interior=%d solid=%d",
				  coarse.dimX, coarse.dimY, coarse.dimZ,
				  countCellsByState(coarse, kStateExterior),
				  intCount,
				  (uint64_t)coarse.solid.size());
	}

	uint64_t numLevels = 0;
	for (int s = coarseCellSize / 2; s >= minCellSize; s /= 2) ++numLevels;
	ProgressTimer timer("holemap", numLevels);
	uint64_t levelsProcessed = 0;

	core::DynamicArray<glm::ivec3> allHoleCells;
	allHoleCells.reserve(totalNodes);
	Level prev = coarse;

	static constexpr size_t kMaxStateCellsHM = 1024ull * 1024ull * 1024ull;

	for (int fineSize = coarseCellSize / 2; fineSize >= minCellSize; fineSize /= 2) {
		Level fine;
		fine.cellSize = fineSize;
		fine.initGrid(gridLower, gridUpper);

		const size_t stateCells = (size_t)fine.dimX * (size_t)fine.dimY * (size_t)fine.dimZ;
		if (stateCells > kMaxStateCellsHM) {
			Log::warn("3dprint holemap: level %d grid %dx%dx%d = %zu cells (%.1f GB) exceeds budget -- stopping",
					  fineSize, fine.dimX, fine.dimY, fine.dimZ, stateCells,
					  (double)stateCells / (1024.0 * 1024.0 * 1024.0));
			break;
		}

		{
			ProgressTimer levelTimer("holemap solidHash", totalNodes);
			buildSolidHash(fine, nodes, &levelTimer);
		}
		for (const auto &kv : fine.solid)
			if (fine.inBounds(kv.first)) fine.state[(size_t)fine.toIdx(kv.first)] = kStateSolid;
		Log::info("3dprint holemap: level %d solidHash done -- grid %dx%dx%d solid=%d",
				  fineSize, fine.dimX, fine.dimY, fine.dimZ, (uint64_t)fine.solid.size());

		initFineFromPrev(fine, prev);
		Log::info("3dprint holemap: level %d initFromPrev done", fineSize);

		const core::DynamicArray<glm::ivec3> holeCells = runBidirectionalBFS(fine);
		timer.addVoxels((int64_t)holeCells.size());
		timer.tick(++levelsProcessed);

		Log::info("3dprint holemap: level %d BFS done -> %d hole cell(s)", fineSize, (uint64_t)holeCells.size());

		for (const glm::ivec3 &h : holeCells) {
			allHoleCells.push_back(h);
		}

		prev = core::move(fine);
	}

	if (allHoleCells.empty()) {
		Log::info("3dprint holemap: model appears watertight down to cell size %d", minCellSize);
		return;
	}
	Log::info("3dprint holemap: %d total hole cell(s) across all levels", (uint64_t)allHoleCells.size());

	// Collect unique solid cells adjacent to each hole cell and recolor them red.
	// No new solid voxels are placed -- only existing wall voxels are recolored.
	static constexpr color::RGBA kHoleColor(255, 0, 0, 255);
	const int finalCellSize = prev.cellSize;

	CellSet adjacentSolids;
	adjacentSolids.reserve(allHoleCells.size() * 6);
	for (const glm::ivec3 &holeCell : allHoleCells) {
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			const glm::ivec3 adj = holeCell + off * finalCellSize;
			if (prev.solid.count(adj) > 0)
				adjacentSolids.insert(adj);
		}
	}

	uint64_t totalColored = 0;
	uint64_t nodesTouched = 0;
	for (const glm::ivec3 &solidCell : adjacentSolids) {
		auto it = prev.solid.find(solidCell);
		if (it == prev.solid.end()) continue;
		const NodeInfo &ni = nodes[it->second];
		scenegraph::SceneGraphNode &node = graph.node(ni.nodeId);
		palette::Palette &pal = node.palette();

		uint64_t skipColorIdx = palette::PaletteColorNotFound;
		for (uint64_t dz = 0; dz < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dz)
			for (uint64_t dy = 0; dy < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dy)
				for (uint64_t dx = 0; dx < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dx) {
					const glm::ivec3 localPos = transformPoint(ni.invWorldMat,
					                                           solidCell + glm::ivec3(dx, dy, dz));
					if (!ni.rv->region().containsPoint(localPos)) continue;
					const voxel::Voxel &v = ni.rv->voxel(localPos);
					if (!voxel::isAir(v.getMaterial()))
						skipColorIdx = (uint64_t)v.getColor();
				}

		uint8_t holeColorIdx = 0;
		const bool paletteChanged = pal.tryAdd(kHoleColor, true, &holeColorIdx, true, skipColorIdx);

		voxel::RawVolumeWrapper wrapper(ni.rv);
		for (uint64_t dz = 0; dz < finalCellSize; ++dz)
			for (uint64_t dy = 0; dy < finalCellSize; ++dy)
				for (uint64_t dx = 0; dx < finalCellSize; ++dx) {
					const glm::ivec3 localPos = transformPoint(ni.invWorldMat,
					                                           solidCell + glm::ivec3(dx, dy, dz));
					if (!ni.rv->region().containsPoint(localPos)) continue;
					const voxel::Voxel &v = ni.rv->voxel(localPos);
					if (voxel::isAir(v.getMaterial())) continue;
					wrapper.setVoxel(localPos, voxel::createVoxel(v.getMaterial(), holeColorIdx,
					                                               v.getNormal(), v.getFlags(), v.getBoneIdx()));
					++totalColored;
				}

		if (wrapper.dirtyRegion().isValid() || paletteChanged) {
			const voxel::Region dirtyRgn = wrapper.dirtyRegion().isValid()
				? wrapper.dirtyRegion() : ni.rv->region();
			sceneMgr->modified(ni.nodeId, dirtyRgn);
			++nodesTouched;
		}
	}

	Log::info("3dprint holemap: colored %d voxel(s) across %d node(s) red (wall voxels adjacent to holes)",
			  totalColored, nodesTouched);
}

// Apply hole fills to the model. Cell-level holes are expanded to finalCellSize^3
// voxels per cell; voxel-level holes (from voxel-plug pass) are placed as single
// voxels. Each hole position is mapped to the nearest existing node via
// `lookup.solid`; positions with no adjacent existing-node owner go into orphan
// bins, which become new model nodes. Modifies the scene graph (`sceneMgr->modified`,
// `nodeResize`, `moveNodeToSceneGraph`).
struct FillStats {
	uint64_t totalFilled = 0;
	uint64_t orphanFilled = 0;
	int orphanNodes = 0;
	uint64_t nodesResized = 0;
	uint64_t nodesTouched = 0;
};

static FillStats applyHoleFills(
		const core::DynamicArray<glm::ivec3> &holeCells,
		const core::DynamicArray<glm::ivec3> &holeVoxels,
		const Level &lookup,
		core::DynamicArray<NodeInfo> &nodes,
		SceneManager *sceneMgr,
		int finalCellSize) {

	FillStats stats;
	if (holeCells.empty() && holeVoxels.empty()) return stats;

	static constexpr color::RGBA kFillColor(0, 220, 0, 255);
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	struct FillPos {
		int nodeIdx;
		glm::ivec3 localPos;
	};
	const size_t fillUpperBound = holeCells.size() * (size_t)(finalCellSize * finalCellSize * finalCellSize)
								  + holeVoxels.size();
	core::DynamicArray<FillPos> fills;
	fills.reserve(fillUpperBound);

	// Worst case: every fill is an orphan. Reserve to same upper bound so push_back
	// during the per-cell/per-voxel loop never reallocs.
	core::DynamicArray<glm::ivec3> orphanVoxels;
	orphanVoxels.reserve(fillUpperBound);

	for (const glm::ivec3 &holeCell : holeCells) {
		int bestNodeIdx = -1;
		int bestDistSq = INT_MAX;
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			const glm::ivec3 adjCell = holeCell + off * finalCellSize;
			auto it = lookup.solid.find(adjCell);
			if (it == lookup.solid.end()) continue;
			const int ni = it->second;
			const glm::ivec3 local = transformPoint(nodes[(size_t)ni].invWorldMat, holeCell);
			const glm::ivec3 lo = nodes[(size_t)ni].rv->region().getLowerCorner();
			const glm::ivec3 hi = nodes[(size_t)ni].rv->region().getUpperCorner();
			const glm::ivec3 clamped = glm::clamp(local, lo, hi);
			const glm::ivec3 d = local - clamped;
			const int distSq = d.x * d.x + d.y * d.y + d.z * d.z;
			if (distSq < bestDistSq) { bestDistSq = distSq; bestNodeIdx = ni; }
		}
		if (bestNodeIdx < 0) {
			for (int dz = 0; dz < finalCellSize; ++dz)
				for (int dy = 0; dy < finalCellSize; ++dy)
					for (int dx = 0; dx < finalCellSize; ++dx)
						orphanVoxels.push_back(holeCell + glm::ivec3(dx, dy, dz));
			continue;
		}
		for (int dz = 0; dz < finalCellSize; ++dz) {
			for (int dy = 0; dy < finalCellSize; ++dy) {
				for (int dx = 0; dx < finalCellSize; ++dx) {
					const glm::ivec3 worldPos = holeCell + glm::ivec3(dx, dy, dz);
					fills.push_back({bestNodeIdx,
									 transformPoint(nodes[(size_t)bestNodeIdx].invWorldMat, worldPos)});
				}
			}
		}
	}

	for (const glm::ivec3 &holeVoxel : holeVoxels) {
		const glm::ivec3 holeCell = toCellOrigin(holeVoxel, finalCellSize);
		int bestNodeIdx = -1;
		int bestDistSq = INT_MAX;
		auto trySolidCell = [&](const glm::ivec3 &c) {
			auto it = lookup.solid.find(c);
			if (it == lookup.solid.end()) return;
			const int ni = it->second;
			const glm::ivec3 local = transformPoint(nodes[(size_t)ni].invWorldMat, holeVoxel);
			const glm::ivec3 lo = nodes[(size_t)ni].rv->region().getLowerCorner();
			const glm::ivec3 hi = nodes[(size_t)ni].rv->region().getUpperCorner();
			const glm::ivec3 clamped = glm::clamp(local, lo, hi);
			const glm::ivec3 d = local - clamped;
			const int distSq = d.x * d.x + d.y * d.y + d.z * d.z;
			if (distSq < bestDistSq) { bestDistSq = distSq; bestNodeIdx = ni; }
		};
		trySolidCell(holeCell);
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			trySolidCell(holeCell + off * finalCellSize);
		}
		if (bestNodeIdx < 0) {
			orphanVoxels.push_back(holeVoxel);
			continue;
		}
		fills.push_back({bestNodeIdx,
						 transformPoint(nodes[(size_t)bestNodeIdx].invWorldMat, holeVoxel)});
	}

	fills.sort([](const FillPos &a, const FillPos &b) { return a.nodeIdx < b.nodeIdx; });

	int fillIdx = 0;
	while (fillIdx < (int)fills.size()) {
		const int curNode = fills[(size_t)fillIdx].nodeIdx;
		const int nodeId = nodes[(size_t)curNode].nodeId;

		voxel::Region fillRegion = nodes[(size_t)curNode].rv->region();
		for (int fi = fillIdx; fi < (int)fills.size() && fills[(size_t)fi].nodeIdx == curNode; ++fi) {
			fillRegion.accumulate(fills[(size_t)fi].localPos);
		}
		if (fillRegion != nodes[(size_t)curNode].rv->region()) {
			sceneMgr->nodeResize(nodeId, fillRegion);
			nodes[(size_t)curNode].rv = sceneMgr->volume(nodeId);
			++stats.nodesResized;
		}

		scenegraph::SceneGraphNode &node = graph.node(nodeId);
		palette::Palette &pal = node.palette();
		uint8_t fillColorIdx = 0;
		pal.tryAdd(kFillColor, true, &fillColorIdx, true);
		voxel::RawVolumeWrapper wrapper(nodes[(size_t)curNode].rv);
		while (fillIdx < (int)fills.size() && fills[(size_t)fillIdx].nodeIdx == curNode) {
			const glm::ivec3 &localPos = fills[(size_t)fillIdx].localPos;
			if (voxel::isAir(nodes[(size_t)curNode].rv->voxel(localPos).getMaterial())) {
				wrapper.setVoxel(localPos, voxel::createVoxel(voxel::VoxelType::Generic, fillColorIdx));
				++stats.totalFilled;
			}
			++fillIdx;
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(nodeId, wrapper.dirtyRegion());
			++stats.nodesTouched;
		}
	}

	if (!orphanVoxels.empty()) {
		static constexpr int kBinSize = 64;
		using Bin = core::DynamicArray<glm::ivec3>;
		std::unordered_map<glm::ivec3, Bin, glm::hash<glm::ivec3>> bins;
		// Estimate: in the worst case every orphan goes to its own bin.
		// In practice orphans cluster, so this is over-reserve but bounded.
		bins.reserve(orphanVoxels.size());
		for (const glm::ivec3 &p : orphanVoxels) {
			Bin &b = bins[toCellOrigin(p, kBinSize)];
			// First-time creation: reserve 1024 entries so push_back doesn't grow
			// from default 0/1 capacity. Tight bin: ~few voxels. Worst-case bin:
			// 64^3 = 262144 voxels, but that'd require a fully-orphan cs=64 cell
			// which is extraordinary.
			if (b.capacity() == 0) b.reserve(1024);
			b.push_back(p);
		}
		for (auto &kv : bins) {
			const Bin &voxels = kv.second;
			glm::ivec3 mn = voxels[0];
			glm::ivec3 mx = voxels[0];
			for (const glm::ivec3 &p : voxels) {
				mn = glm::min(mn, p);
				mx = glm::max(mx, p);
			}
			const voxel::Region region(glm::ivec3(0), mx - mn);
			voxel::RawVolume *vol = new voxel::RawVolume(region);
			palette::Palette pal;
			uint8_t fillColorIdx = 0;
			pal.tryAdd(kFillColor, true, &fillColorIdx, true);
			const voxel::Voxel fillVoxel = voxel::createVoxel(voxel::VoxelType::Generic, fillColorIdx);
			for (const glm::ivec3 &p : voxels) {
				vol->setVoxelUnsafe(p - mn, fillVoxel);
				++stats.orphanFilled;
			}
			scenegraph::SceneGraphNode newNode(scenegraph::SceneGraphNodeType::Model);
			newNode.setVolume(vol);
			newNode.setName("holefill_orphan");
			newNode.setPalette(pal);
			scenegraph::SceneGraphTransform transform;
			transform.setWorldTranslation(glm::vec3(mn));
			newNode.setTransform(0, transform);
			sceneMgr->moveNodeToSceneGraph(newNode, 0);
			++stats.orphanNodes;
		}
	}

	return stats;
}

void runHoleFill(SceneManager *sceneMgr, int minCellSize) {
	g_rssCapTripped.store(false);
	logRSS("entry to runHoleFill");
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	core::DynamicArray<NodeInfo> nodes;
	nodes.reserve((size_t)graph.size());
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) continue;
		NodeInfo info;
		info.nodeId = node.id();
		info.rv = rv;
		info.worldMat = graph.worldMatrix(node, 0);
		info.invWorldMat = glm::inverse(info.worldMat);
		info.cellOrigin = glm::ivec3(0);
		nodes.push_back(info);
	}
	if (nodes.empty()) {
		Log::info("3dprint holefill: no model nodes");
		return;
	}
	const int totalNodes = (int)nodes.size();

	int coarseCellSize = 0;
	{
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes)
			widthCount[ni.rv->region().getWidthInVoxels()]++;
		int bestCount = 0;
		for (const auto &kv : widthCount)
			if (kv.second > bestCount) { bestCount = kv.second; coarseCellSize = kv.first; }
	}
	if (coarseCellSize <= 0) {
		Log::error("3dprint holefill: could not determine cell size -- run 3dprint regrid first");
		return;
	}
	if (minCellSize <= 0 || minCellSize >= coarseCellSize) {
		minCellSize = coarseCellSize / 2;
	}
	if (minCellSize <= 0) {
		Log::error("3dprint holefill: coarse cell size %d too small to subdivide", coarseCellSize);
		return;
	}

	voxel::Region worldBbox = voxel::Region::InvalidRegion;
	Level coarse;
	coarse.cellSize = coarseCellSize;
	coarse.solid.reserve((size_t)totalNodes);
	glm::ivec3 gridLower(INT_MAX, INT_MAX, INT_MAX);
	glm::ivec3 gridUpper(INT_MIN, INT_MIN, INT_MIN);

	for (int i = 0; i < totalNodes; ++i) {
		NodeInfo &ni = nodes[i];
		const voxel::Region &r = ni.rv->region();
		const glm::ivec3 corners[8] = {
			{r.getLowerX(), r.getLowerY(), r.getLowerZ()}, {r.getUpperX(), r.getLowerY(), r.getLowerZ()},
			{r.getLowerX(), r.getUpperY(), r.getLowerZ()}, {r.getUpperX(), r.getUpperY(), r.getLowerZ()},
			{r.getLowerX(), r.getLowerY(), r.getUpperZ()}, {r.getUpperX(), r.getLowerY(), r.getUpperZ()},
			{r.getLowerX(), r.getUpperY(), r.getUpperZ()}, {r.getUpperX(), r.getUpperY(), r.getUpperZ()},
		};
		for (const glm::ivec3 &c : corners) {
			const glm::ivec3 wc = transformPoint(ni.worldMat, c);
			if (worldBbox.isValid()) worldBbox.accumulate(wc); else worldBbox = voxel::Region(wc, wc);
		}
		ni.cellOrigin = toCellOrigin(transformPoint(ni.worldMat, r.getLowerCorner()), coarseCellSize);
		gridLower = glm::min(gridLower, ni.cellOrigin);
		gridUpper = glm::max(gridUpper, ni.cellOrigin);
		bool hasSolid = false;
		for (int z = r.getLowerZ(); z <= r.getUpperZ() && !hasSolid; ++z)
			for (int y = r.getLowerY(); y <= r.getUpperY() && !hasSolid; ++y)
				for (int x = r.getLowerX(); x <= r.getUpperX() && !hasSolid; ++x)
					if (!voxel::isAir(ni.rv->voxel(x, y, z).getMaterial()))
						hasSolid = true;
		if (hasSolid) {
			coarse.solid.emplace(ni.cellOrigin, i);
		}
	}

	gridLower -= glm::ivec3(coarseCellSize);
	gridUpper += glm::ivec3(coarseCellSize);
	coarse.initGrid(gridLower, gridUpper);
	coarse.state.assign((size_t)coarse.dimX * coarse.dimY * coarse.dimZ, kStateEmpty);
	for (const auto &kv : coarse.solid)
		if (coarse.inBounds(kv.first)) coarse.state[(size_t)coarse.toIdx(kv.first)] = kStateSolid;
	logRSS("after coarse grid init");

	{
		Level emptyPrev;
		buildExterior(coarse, emptyPrev);
	}
	logRSS("after coarse buildExterior");
	{
		const uint64_t intCount = buildInteriorAllSeeds(coarse);
		if (intCount == 0) {
			Log::warn("3dprint holefill: no enclosed interior at coarse scale -- model may not be sealed");
			return;
		}
		Log::info("3dprint holefill: coarse interior: %d cell(s)", intCount);
	}
	logRSS("after coarse buildInterior");

	uint64_t numLevels = 0;
	for (int s = coarseCellSize / 2; s >= minCellSize; s /= 2) ++numLevels;
	ProgressTimer timer("holefill", numLevels + 1); // +1 for fill phase
	uint64_t levelsProcessed = 0;

	core::DynamicArray<glm::ivec3> allHoleCells;
	allHoleCells.reserve(totalNodes);
	Level prev = coarse;

	// 16 GB state cap: admits cs=2 on the big model (~7.6 GB) while still blocking
	// cs=1 (~60 GB). Keep the next-level scan time in mind -- BFS at cs=2 walks an
	// 8x larger grid than cs=4.
	static constexpr size_t kMaxStateCells = 16ull * 1024ull * 1024ull * 1024ull;

	for (int fineSize = coarseCellSize / 2; fineSize >= minCellSize; fineSize /= 2) {
		Level fine;
		fine.cellSize = fineSize;
		fine.initGrid(gridLower, gridUpper);

		const size_t stateCells = (size_t)fine.dimX * (size_t)fine.dimY * (size_t)fine.dimZ;
		if (stateCells > kMaxStateCells) {
			Log::warn("3dprint holefill: level %d grid %dx%dx%d = %zu cells (%.1f GB state) exceeds "
					  "16 GB budget -- stopping refinement here",
					  fineSize, fine.dimX, fine.dimY, fine.dimZ, stateCells,
					  (double)stateCells / (1024.0 * 1024.0 * 1024.0));
			break;
		}

		// Per-level convergence: build state from current model (which may have
		// fills from prior levels or prior sub-iterations), run BFS, fill any
		// holes, repeat until a sub-iteration produces 0 new holes. No iter cap.
		//
		// Optimization: only sub-iter 1 builds state from scratch (solidHash +
		// initFineFromPrev). Subsequent sub-iters reuse the state from the prior
		// BFS+fill: prev-level classification didn't change (so initFineFromPrev
		// would produce identical output), and the cells just filled act as
		// walls for BFS via their kStateBlocked classification (which BFS
		// skips, equivalent to kStateSolid for expansion purposes). This saves
		// ~15 s per sub-iter at cs=2 (5 s init + 10 s solidHash).
		const uint64_t levelStartMs = core::TimeProvider::systemMillis();
		int subIter = 0;
		uint64_t totalLevelFilled = 0;
		while (true) {
			++subIter;
			const uint64_t subStartMs = core::TimeProvider::systemMillis();

			// Sub-iter 2+: state was preserved from sub-iter 1 (we skip
			// solidHash + initFineFromPrev). Sub-iter 1's BFS ran to completion
			// (frontier exhausted at end). Since fills only add walls (which
			// BFS already skips via kStateBlocked) and don't create new ext/int
			// sources, sub-iter 2's BFS is guaranteed to find an empty frontier
			// and 0 holes. Skip the buildFrontiers re-scan (~10 s at cs=2).
			if (subIter > 1) {
				Log::info("3dprint holefill: cs=%d sub-iter %d skipped -- shells unchanged since previous BFS, convergence implicit",
						  fineSize, subIter);
				break;
			}

			if (subIter == 1) {
				fine.solid.clear();
				fine.state.assign(stateCells, kStateEmpty);

				const uint64_t hashStartMs = core::TimeProvider::systemMillis();
				{
					ProgressTimer levelTimer("holefill solidHash", totalNodes);
					buildSolidHash(fine, nodes, &levelTimer);
				}
				for (const auto &kv : fine.solid)
					if (fine.inBounds(kv.first)) fine.state[(size_t)fine.toIdx(kv.first)] = kStateSolid;
				const double hashSecs = (double)(core::TimeProvider::systemMillis() - hashStartMs) / 1000.0;
				Log::info("3dprint holefill: cs=%d sub-iter %d solidHash done in %.1fs -- solid=%zu",
						  fineSize, subIter, hashSecs, fine.solid.size());

				const uint64_t initStartMs = core::TimeProvider::systemMillis();
				initFineFromPrev(fine, prev);
				const double initSecs = (double)(core::TimeProvider::systemMillis() - initStartMs) / 1000.0;
				Log::info("3dprint holefill: cs=%d sub-iter %d initFromPrev done in %.1fs",
						  fineSize, subIter, initSecs);
				if (checkRSSCap("after initFineFromPrev")) break;
			} else {
				Log::info("3dprint holefill: cs=%d sub-iter %d reusing state from prior BFS (skipped solidHash + initFineFromPrev)",
						  fineSize, subIter);
			}

			const uint64_t bfsStartMs = core::TimeProvider::systemMillis();
			const core::DynamicArray<glm::ivec3> holeCells = runBidirectionalBFS(fine);
			const double bfsSecs = (double)(core::TimeProvider::systemMillis() - bfsStartMs) / 1000.0;
			Log::info("3dprint holefill: cs=%d sub-iter %d BFS done in %.1fs -> %zu hole cell(s)",
					  fineSize, subIter, bfsSecs, holeCells.size());
			if (checkRSSCap("after runBidirectionalBFS")) break;

			if (holeCells.empty()) {
				const double subSecs = (double)(core::TimeProvider::systemMillis() - subStartMs) / 1000.0;
				Log::info("3dprint holefill: cs=%d sub-iter %d converged (no holes) in %.1fs",
						  fineSize, subIter, subSecs);
				break;
			}

			// Deferred fill: only fill at the deepest level (minCellSize). At
			// intermediate levels the BFS classification still propagates to
			// finer levels via initFineFromPrev (the kStateBlocked markers
			// don't propagate, but the surrounding ext/int expansion does), so
			// finer levels rediscover the holes at higher resolution. Filling
			// only at the deepest level produces the tightest plug volumes:
			// at cs=128 a fill is 128^3 = 2M voxels per hole; at cs=2 it's 8.
			if (fineSize > minCellSize) {
				const double subSecs = (double)(core::TimeProvider::systemMillis() - subStartMs) / 1000.0;
				Log::info("3dprint holefill: cs=%d sub-iter %d found %zu holes -- fill DEFERRED to cs=%d (sub-iter total %.1fs)",
						  fineSize, subIter, holeCells.size(), minCellSize, subSecs);
				// No fill -> no model change -> re-iterating at this cs would
				// find the same holes again. Break to advance to cs/2.
				break;
			}

			const uint64_t fillStartMs = core::TimeProvider::systemMillis();
			const core::DynamicArray<glm::ivec3> emptyVoxels;
			FillStats st = applyHoleFills(holeCells, emptyVoxels, fine, nodes, sceneMgr, fineSize);
			const double fillSecs = (double)(core::TimeProvider::systemMillis() - fillStartMs) / 1000.0;
			totalLevelFilled += st.totalFilled + st.orphanFilled;
			const double subSecs = (double)(core::TimeProvider::systemMillis() - subStartMs) / 1000.0;
			Log::info("3dprint holefill: cs=%d sub-iter %d filled %lu voxel(s) (+%lu orphan in %d new node(s)) in %.1fs (sub-iter total %.1fs)",
					  fineSize, subIter,
					  (unsigned long)st.totalFilled,
					  (unsigned long)st.orphanFilled, st.orphanNodes,
					  fillSecs, subSecs);

			// Break if BFS reported holes but applyHoleFills couldn't actually
			// add any solid voxels (every "hole" is already solid in the model).
			// Without this the loop is infinite -- BFS keeps re-detecting the
			// same already-filled cells because state is reset each sub-iter.
			if (st.totalFilled == 0 && st.orphanFilled == 0) {
				Log::info("3dprint holefill: cs=%d sub-iter %d found %zu holes but 0 actual fills -- treating as converged",
						  fineSize, subIter, holeCells.size());
				break;
			}
		}
		const double levelSecs = (double)(core::TimeProvider::systemMillis() - levelStartMs) / 1000.0;
		Log::info("3dprint holefill: cs=%d CONVERGED after %d sub-iter(s) in %.1fs (%lu total voxels filled at this level)",
				  fineSize, subIter, levelSecs, (unsigned long)totalLevelFilled);
		timer.addVoxels((int64_t)0);
		timer.tick(++levelsProcessed);

		prev = core::move(fine);
	}

	// All cell-level holes have been filled per-level by the loop above.
	//
	// Voxel-plug passes with progressive radius. Smaller radius first to
	// minimize splat size from BFS crawling along walls (the algorithm has
	// a known flaw where wall-parallel air voxels get plugged together).
	// Each radius pass applies fills before the next runs, so previously-
	// detected leaks are sealed and won't be re-detected at larger radius.
	//
	// Build the ext frontier ONCE here, reuse across all radii. Fills don't
	// create new ext-classified cells (only mark existing solid cells as
	// "more solid"), so the frontier list is identical for every radius.
	const int prevCellSize = prev.cellSize;
	core::DynamicArray<size_t> plugExtFrontier, plugIntFrontier;
	if (!g_rssCapTripped.load(std::memory_order_relaxed)) {
		const uint64_t frontierStartMs = core::TimeProvider::systemMillis();
		buildFrontiers(prev, plugExtFrontier, plugIntFrontier, /*wrapSolid=*/true);
		const double frontierSecs = (double)(core::TimeProvider::systemMillis() - frontierStartMs) / 1000.0;
		fprintf(stderr, "[PLUG] frontier built once: ext=%zu (in %.1fs, reused across all radii)\n",
				plugExtFrontier.size(), frontierSecs);
		fflush(stderr);
	}

	// Plug-pass algorithm choice. Two implementations available:
	//   true  = runVoxelPlug          (option E: two-sided multi-source BFS in 33^3
	//                                  cube, plug min-totalDepth voxels = global
	//                                  bottleneck. Tighter plugs, slower per cell.)
	//   false = runVoxelPlugSinglePath (option C: single-source BFS with parent
	//                                   tracking, plug shortest-path voxels back
	//                                   from first int contact. Faster per cell,
	//                                   plugs stack across overlapping sources.)
	// Flip and rebuild to compare.
	static constexpr bool kUseTwoSidedPlug = false;
	if (!g_rssCapTripped.load(std::memory_order_relaxed)) {
		// kPlugRadius=8 gives a 17^3 cube (4913 voxels) -- 8x fewer voxels
		// per cube than radius=16 (33^3 = 35937), cube populate is ~8x faster.
		// Sufficient for walls up to 16 voxels thick (radius 8 each side
		// converges in the middle). For thicker walls bump back to 16.
		const int kPlugRadius = 8;
		core::DynamicArray<glm::ivec3> plugVoxels;

		// Pre-compute fillColorIdx per node (single-threaded). After this,
		// parallel detection threads read fillColorIdx lock-free instead of
		// modifying the palette concurrently. Also seeds the palette so the
		// fill colour exists before live writes start.
		static constexpr color::RGBA kFillColor(0, 220, 0, 255);
		for (NodeInfo &ni : nodes) {
			scenegraph::SceneGraphNode &node = graph.node(ni.nodeId);
			palette::Palette &pal = node.palette();
			uint8_t idx = 0;
			pal.tryAdd(kFillColor, true, &idx, true);
			ni.fillColorIdx = idx;
		}

		// Atomic dirty flag per node. Set during live writes; read after
		// detection to know which nodes need sceneMgr->modified() notifications.
		const size_t numNodes = nodes.size();
		std::atomic<uint8_t> *nodeDirty = new std::atomic<uint8_t>[numNodes];
		for (size_t i = 0; i < numNodes; ++i) nodeDirty[i].store(0, std::memory_order_relaxed);

		if (kUseTwoSidedPlug) {
			runVoxelPlug(prev, nodes, nodeDirty, plugExtFrontier, plugVoxels, kPlugRadius);
		} else {
			runVoxelPlugSinglePath(prev, nodes, nodeDirty, plugExtFrontier, plugVoxels, kPlugRadius);
		}

		// Live-written nodes need explicit sceneMgr->modified() so the renderer
		// picks up the new voxels. setVoxelUnsafe writes to the RawVolume but
		// doesn't touch dirty-region/scene-graph bookkeeping.
		uint64_t liveNodesNotified = 0;
		for (size_t i = 0; i < numNodes; ++i) {
			if (nodeDirty[i].load(std::memory_order_relaxed) != 0) {
				sceneMgr->modified((int)nodes[i].nodeId, nodes[i].rv->region());
				++liveNodesNotified;
			}
		}
		delete[] nodeDirty;
		Log::info("3dprint holefill: plug live-write notified %lu dirty node(s)",
				  (unsigned long)liveNodesNotified);

		if (!plugVoxels.empty()) {
			// applyHoleFills now only sees orphan plug voxels (those with no
			// owning existing node -- live writes already handled the rest).
			const core::DynamicArray<glm::ivec3> emptyCells;
			FillStats st = applyHoleFills(emptyCells, plugVoxels, prev, nodes, sceneMgr, prevCellSize);
			Log::info("3dprint holefill: plug pass filled %lu voxel(s) (+%lu orphan in %d new node(s))",
					  (unsigned long)st.totalFilled, (unsigned long)st.orphanFilled, st.orphanNodes);
			fprintf(stderr, "[PLUG] %lu plugged, %lu orphans in %d new nodes\n",
					(unsigned long)st.totalFilled, (unsigned long)st.orphanFilled, st.orphanNodes);
			fflush(stderr);
		} else {
			Log::info("3dprint holefill: plug pass found 0 leaks");
		}
	}
	logRSS("after voxel plug pass");

	timer.tick(++levelsProcessed);
}

void runDebugFrontier(SceneManager *sceneMgr, int cellSize) {
	const uint64_t fnStart = core::TimeProvider::systemMillis();
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	core::DynamicArray<NodeInfo> nodes;
	nodes.reserve((size_t)graph.size());
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) continue;
		NodeInfo info;
		info.nodeId      = node.id();
		info.rv          = rv;
		info.worldMat    = graph.worldMatrix(node, 0);
		info.invWorldMat = glm::inverse(info.worldMat);
		info.cellOrigin  = glm::ivec3(0);
		nodes.push_back(info);
	}
	if (nodes.empty()) { Log::info("3dprint debugfrontier: no model nodes"); return; }
	const int totalNodes = (int)nodes.size();

	// Default cellSize = modal regridded width (typically 128 after `3dprint regrid`).
	if (cellSize <= 0) {
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes) widthCount[ni.rv->region().getWidthInVoxels()]++;
		int bestCount = 0;
		for (const auto &kv : widthCount)
			if (kv.second > bestCount) { bestCount = kv.second; cellSize = kv.first; }
	}
	if (cellSize <= 0) {
		Log::error("3dprint debugfrontier: invalid cellSize");
		return;
	}
	// Determine the natural coarse cell size from regridded model nodes (modal width).
	int coarseCellSize = 0;
	{
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes) widthCount[ni.rv->region().getWidthInVoxels()]++;
		int bestCount = 0;
		for (const auto &kv : widthCount)
			if (kv.second > bestCount) { bestCount = kv.second; coarseCellSize = kv.first; }
	}
	if (coarseCellSize <= 0) {
		Log::error("3dprint debugfrontier: could not determine coarse cell size -- run 3dprint regrid first");
		return;
	}
	// User's requested cellSize is the deepest level we'll refine to. Clamp:
	//  cellSize <= 0          : default to coarse
	//  cellSize >= coarse     : just run coarse, don't refine
	//  0 < cellSize < coarse  : refine via halving until we cross targetCellSize
	int targetCellSize = cellSize;
	if (targetCellSize <= 0) targetCellSize = coarseCellSize;
	if (targetCellSize > coarseCellSize) targetCellSize = coarseCellSize;
	Log::info("3dprint debugfrontier: coarse=%d target=%d nodes=%d",
			  coarseCellSize, targetCellSize, totalNodes);

	// World grid bbox from snapped node lower-corners. Expand by 1 coarse cell on
	// each side so exterior BFS has air around the model to seed from.
	glm::ivec3 gridLower(INT_MAX, INT_MAX, INT_MAX), gridUpper(INT_MIN, INT_MIN, INT_MIN);
	Level coarse;
	coarse.cellSize = coarseCellSize;
	coarse.solid.reserve((size_t)totalNodes);
	for (int i = 0; i < totalNodes; ++i) {
		NodeInfo &ni = nodes[i];
		const voxel::Region &r = ni.rv->region();
		ni.cellOrigin = toCellOrigin(transformPoint(ni.worldMat, r.getLowerCorner()), coarseCellSize);
		gridLower = glm::min(gridLower, ni.cellOrigin);
		gridUpper = glm::max(gridUpper, ni.cellOrigin);
		bool hasSolid = false;
		for (int z = r.getLowerZ(); z <= r.getUpperZ() && !hasSolid; ++z)
			for (int y = r.getLowerY(); y <= r.getUpperY() && !hasSolid; ++y)
				for (int x = r.getLowerX(); x <= r.getUpperX() && !hasSolid; ++x)
					if (!voxel::isAir(ni.rv->voxel(x, y, z).getMaterial())) hasSolid = true;
		if (hasSolid) coarse.solid.emplace(ni.cellOrigin, i);
	}
	gridLower -= glm::ivec3(coarseCellSize);
	gridUpper += glm::ivec3(coarseCellSize);

	// Coarse pass: state grid + ext/uint64_t classification. Same as runHoleFill.
	coarse.initGrid(gridLower, gridUpper);
	for (const auto &kv : coarse.solid)
		if (coarse.inBounds(kv.first)) coarse.state[(size_t)coarse.toIdx(kv.first)] = kStateSolid;
	{ Level emptyPrev; buildExterior(coarse, emptyPrev); }
	const uint64_t coarseInt = buildInteriorAllSeeds(coarse);
	Log::info("3dprint debugfrontier: coarse %dx%dx%d ext+uint64_t classified (uint64_t=%d, elapsed=%.1fs)",
			  coarse.dimX, coarse.dimY, coarse.dimZ, coarseInt, elapsedSince(fnStart));

	// Progressive refinement chain (mirrors runHoleFill exactly so the visualised
	// frontier matches what holefill would actually expand from). State at each
	// fine level inherits from prev via initFineFromPrev, then runBidirectionalBFS
	// advances the frontier and detects holes. We iterate down to targetCellSize.
	static constexpr int64_t kStateBudget = 4ll * 1024ll * 1024ll * 1024ll;
	Level prev = core::move(coarse);
	for (int fineSize = coarseCellSize / 2; fineSize >= targetCellSize; fineSize /= 2) {
		Level fine;
		fine.cellSize = fineSize;
		fine.initGrid(gridLower, gridUpper);
		const int64_t fineCells = (int64_t)fine.dimX * (int64_t)fine.dimY * (int64_t)fine.dimZ;
		if (fineCells > kStateBudget) {
			Log::error("3dprint debugfrontier: level %d grid %dx%dx%d = %lld cells (%.1f GB) exceeds state budget",
					   fineSize, fine.dimX, fine.dimY, fine.dimZ, (long long)fineCells,
					   (double)fineCells / (1024.0 * 1024.0 * 1024.0));
			return;
		}
		{
			ProgressTimer t("debugfrontier solidHash", totalNodes);
			buildSolidHash(fine, nodes, &t);
		}
		for (const auto &kv : fine.solid)
			if (fine.inBounds(kv.first)) fine.state[(size_t)fine.toIdx(kv.first)] = kStateSolid;
		initFineFromPrev(fine, prev);
		const core::DynamicArray<glm::ivec3> holeCells = runBidirectionalBFS(fine);
		Log::info("3dprint debugfrontier: level %d done (grid %dx%dx%d, holes=%d, elapsed=%.1fs)",
				  fineSize, fine.dimX, fine.dimY, fine.dimZ,
				  (uint64_t)holeCells.size(), elapsedSince(fnStart));
		prev = core::move(fine);
		if (fineSize == targetCellSize) break;
	}

	// Extract frontier from the deepest level. wrapSolid=true picks up ext/uint64_t
	// cells whose neighbour is solid OR blocked (a detected hole), not just
	// kStateEmpty -- after runBidirectionalBFS most empties are gone, so we'd
	// otherwise get a near-empty frontier.
	const Level &fine = prev;
	const uint64_t minCellSize = fine.cellSize;
	core::DynamicArray<size_t> extFrontier, intFrontier;
	buildFrontiers(fine, extFrontier, intFrontier, /*wrapSolid=*/true);
	const uint64_t extTotal = (uint64_t)extFrontier.size();
	const uint64_t intTotal = (uint64_t)intFrontier.size();
	Log::info("3dprint debugfrontier: cellSize=%d ext frontier=%d int frontier=%d (elapsed=%.1fs)",
			  minCellSize, extTotal, intTotal, elapsedSince(fnStart));
	if (extTotal == 0 && intTotal == 0) {
		Log::info("3dprint debugfrontier: no frontier cells -- nothing to visualise");
		return;
	}

	// Chunked output: a single dense RawVolume covering the world bbox would be
	// terabytes for typical models. Instead bin frontier cells into world-aligned
	// chunks; each occupied chunk becomes one scene-graph node sized to a tight
	// bbox of the cells inside it. Empty chunks cost nothing.
	// Pick chunk granularity that's a multiple of cellSize so binning lines up
	// cleanly. Floor at 256 voxels per axis so we don't shatter into thousands of
	// tiny nodes; if the cellSize is larger than 256, snap up to cellSize.
	uint64_t chunkSizeVoxels = cellSize;
	while (chunkSizeVoxels < 256) chunkSizeVoxels *= 2;

	struct ChunkCell {
		glm::ivec3 localOrigin; // relative to chunkOrigin
		voxel::Voxel voxel;
	};
	struct ChunkData {
		glm::ivec3 chunkOrigin{0};
		glm::ivec3 bboxLo{INT_MAX, INT_MAX, INT_MAX};
		glm::ivec3 bboxHi{INT_MIN, INT_MIN, INT_MIN};
		core::DynamicArray<ChunkCell> cells;
		voxel::RawVolume *volume = nullptr;
	};
	std::unordered_map<glm::ivec3, uint64_t, glm::hash<glm::ivec3>> chunkLookup;
	chunkLookup.reserve((size_t)(extTotal + intTotal) / 16 + 16);
	core::DynamicArray<ChunkData> chunks;
	chunks.reserve((size_t)(extTotal + intTotal) / 16 + 16);

	palette::Palette pal;
	uint8_t extColorIdx = 0;
	uint8_t intColorIdx = 0;
	pal.tryAdd(color::RGBA(255, 128,   0, 255), true, &extColorIdx, true); // orange = exterior frontier
	pal.tryAdd(color::RGBA(  0, 128, 255, 255), true, &intColorIdx, true); // blue   = interior frontier
	const voxel::Voxel extV = voxel::createVoxel(voxel::VoxelType::Generic, extColorIdx);
	const voxel::Voxel intV = voxel::createVoxel(voxel::VoxelType::Generic, intColorIdx);

	const uint64_t binStart = core::TimeProvider::systemMillis();
	uint64_t lastBinLogMs = binStart;
	auto binCell = [&](size_t cellIdx, const voxel::Voxel &v) {
		const glm::ivec3 origin = fine.toCell(cellIdx);
		const glm::ivec3 chunkOrigin = toCellOrigin(origin, chunkSizeVoxels);
		auto it = chunkLookup.find(chunkOrigin);
		uint64_t idx;
		if (it == chunkLookup.end()) {
			idx = (uint64_t)chunks.size();
			ChunkData cd;
			cd.chunkOrigin = chunkOrigin;
			chunks.push_back(core::move(cd));
			chunkLookup.emplace(chunkOrigin, idx);
		} else {
			idx = it->second;
		}
		ChunkData &cd = chunks[(size_t)idx];
		const glm::ivec3 local = origin - chunkOrigin;
		const glm::ivec3 localHi = local + glm::ivec3(minCellSize - 1);
		cd.bboxLo = glm::min(cd.bboxLo, local);
		cd.bboxHi = glm::max(cd.bboxHi, localHi);
		ChunkCell c;
		c.localOrigin = local;
		c.voxel = v;
		cd.cells.push_back(c);
	};
	for (uint64_t i = 0; i < extTotal; ++i) {
		binCell(extFrontier[(size_t)i], extV);
		if (((i + 1) & 65535) == 0) {
			const uint64_t now = core::TimeProvider::systemMillis();
			if (now - lastBinLogMs >= 2000u) {
				lastBinLogMs = now;
				Log::info("3dprint debugfrontier: binning ext %d/%d (%zu chunks so far) elapsed=%.1fs",
						  i + 1, extTotal, chunks.size(), (double)(now - binStart) / 1000.0);
			}
		}
	}
	for (uint64_t i = 0; i < intTotal; ++i) {
		binCell(intFrontier[(size_t)i], intV);
		if (((i + 1) & 65535) == 0) {
			const uint64_t now = core::TimeProvider::systemMillis();
			if (now - lastBinLogMs >= 2000u) {
				lastBinLogMs = now;
				Log::info("3dprint debugfrontier: binning uint64_t %d/%d (%zu chunks so far) elapsed=%.1fs",
						  i + 1, intTotal, chunks.size(), (double)(now - binStart) / 1000.0);
			}
		}
	}
	const uint64_t chunkCount = (uint64_t)chunks.size();
	Log::info("3dprint debugfrontier: %d cell(s) binned into %d chunk(s) of %d^3 voxels (elapsed=%.1fs)",
			  extTotal + intTotal, chunkCount, chunkSizeVoxels, elapsedSince(fnStart));

	// Allocate + paint chunks in parallel. Each chunk's RawVolume is independent
	// memory, and within a chunk setVoxelUnsafe writes to disjoint indices, so
	// the parallel range is fully race-free. RawVolume new[] is the up-front
	// reservation -- one allocation per chunk, sized to the tight bbox.
	const uint64_t paintStart = core::TimeProvider::systemMillis();
	std::atomic<uint64_t> chunksDone{0};
	std::atomic<int64_t> voxelsPainted{0};
	std::atomic<uint64_t> lastLogMs{paintStart};

	app::for_parallel(0, chunkCount, [&](uint64_t start, uint64_t end) {
		for (uint64_t i = start; i < end; ++i) {
			ChunkData &cd = chunks[(size_t)i];
			const voxel::Region region(cd.bboxLo, cd.bboxHi);
			cd.volume = new voxel::RawVolume(region);
			int64_t local = 0;
			for (const ChunkCell &c : cd.cells) {
				for (uint64_t dz = 0; dz < minCellSize; ++dz)
					for (uint64_t dy = 0; dy < minCellSize; ++dy)
						for (uint64_t dx = 0; dx < minCellSize; ++dx)
							cd.volume->setVoxelUnsafe(c.localOrigin + glm::ivec3(dx, dy, dz), c.voxel);
				local += (int64_t)minCellSize * minCellSize * minCellSize;
			}
			voxelsPainted.fetch_add(local, std::memory_order_relaxed);
			const uint64_t done = chunksDone.fetch_add(1, std::memory_order_relaxed) + 1;
			const uint64_t now = core::TimeProvider::systemMillis();
			uint64_t prevMs = lastLogMs.load(std::memory_order_relaxed);
			if (now - prevMs >= 2000u && lastLogMs.compare_exchange_strong(prevMs, now)) {
				Log::info("3dprint debugfrontier: painting chunk %d/%d (%lld voxels) elapsed=%.1fs",
						  done, chunkCount,
						  (long long)voxelsPainted.load(std::memory_order_relaxed),
						  (double)(now - paintStart) / 1000.0);
			}
		}
	});
	Log::info("3dprint debugfrontier: paint done in %.1fs (%d chunks, %lld voxels)",
			  (double)(core::TimeProvider::systemMillis() - paintStart) / 1000.0,
			  chunkCount, (long long)voxelsPainted.load());

	// Sequential scene-graph insertion. moveNodeToSceneGraph touches shared state
	// (memento history, child-id maps, modified() bookkeeping), so this stays on
	// the main thread. One node per chunk -- not per cell -- keeps node count
	// bounded by surface area / chunkSize^2 instead of total surface cell count.
	//
	// Use SceneManager::moveNodeToSceneGraph so each insertion runs onNewNodeAdded:
	// that propagates the dirty world translation to a real local matrix via
	// updateTransforms() and registers the node with the renderer. Skipping that
	// (calling scenegraph::moveNodeToSceneGraph directly) leaves every node sitting
	// at world (0,0,0) because the world-translation getters set DIRTY_WORLDVALUES
	// only -- the local matrix stays identity until update() runs.
	const uint64_t insertStart = core::TimeProvider::systemMillis();
	uint64_t lastInsertLogMs = insertStart;
	for (uint64_t i = 0; i < chunkCount; ++i) {
		ChunkData &cd = chunks[(size_t)i];
		scenegraph::SceneGraphNode newNode(scenegraph::SceneGraphNodeType::Model);
		newNode.setVolume(cd.volume);
		cd.volume = nullptr;
		newNode.setName("debugfrontier");
		newNode.setPalette(pal);
		scenegraph::SceneGraphTransform transform;
		transform.setWorldTranslation(glm::vec3(cd.chunkOrigin));
		newNode.setTransform(0, transform);
		sceneMgr->moveNodeToSceneGraph(newNode, 0);
		if (((i + 1) & 31) == 0) {
			const uint64_t now = core::TimeProvider::systemMillis();
			if (now - lastInsertLogMs >= 2000u) {
				lastInsertLogMs = now;
				Log::info("3dprint debugfrontier: inserting node %d/%d elapsed=%.1fs",
						  i + 1, chunkCount, (double)(now - insertStart) / 1000.0);
			}
		}
	}
	Log::info("3dprint debugfrontier: %d node(s) inserted -- total elapsed=%.1fs",
			  chunkCount, elapsedSince(fnStart));
}

} // namespace printing
} // namespace voxedit
