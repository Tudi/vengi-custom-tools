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
#include "voxedit-util/SceneManager.h"
#include "voxel/Connectivity.h"
#include "voxel/RawVolume.h"
#include "voxel/RawVolumeWrapper.h"
#include "voxel/Region.h"
#include "voxel/Voxel.h"

#include <atomic>
#include <glm/matrix.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace voxedit {
namespace printing {

namespace {

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
	int nodeId;
	voxel::RawVolume *rv;
	glm::mat4 worldMat;
	glm::mat4 invWorldMat;
	glm::ivec3 cellOrigin; // snapped to coarse cell grid
};

using CellHash = std::unordered_map<glm::ivec3, int, glm::hash<glm::ivec3>>;
using CellSet  = std::unordered_set<glm::ivec3, glm::hash<glm::ivec3>>;

// One resolution level.
// Solid lookup: sparse CellHash (cells that contain >= 1 solid voxel).
// Exterior/interior: flat uint8_t state array indexed by grid position.
//   Size = dimX * dimY * dimZ bytes. For level-16 grid this is ~12 MB vs
//   ~450 MB for an unordered_set -- 37x smaller, fits in L3 cache.
struct Level {
	int cellSize = 0;
	glm::ivec3 gridLower{0};
	int dimX = 0, dimY = 0, dimZ = 0;

	CellHash solid;             // sparse solid map: snapped cell → node index
	std::vector<uint8_t> state; // flat: kStateEmpty/Exterior/Interior per cell

	void initGrid(const glm::ivec3 &lo, const glm::ivec3 &hi) {
		gridLower = lo;
		dimX = (hi.x - lo.x) / cellSize + 1;
		dimY = (hi.y - lo.y) / cellSize + 1;
		dimZ = (hi.z - lo.z) / cellSize + 1;
		state.assign((size_t)dimX * dimY * dimZ, kStateEmpty);
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
	return glm::ivec3((int)glm::round(w.x), (int)glm::round(w.y), (int)glm::round(w.z));
}

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
static int parallelBFS(Level &lvl, core::DynamicArray<int32_t> &frontier, uint8_t stateValue) {
	int totalMarked = 0;
	while (!frontier.empty()) {
		const int sz = (int)frontier.size();
		totalMarked += sz;

		// Parallel: each thread processes its range of the frontier and writes
		// new neighbours to a per-thread local buffer.
		core::DynamicArray<core::DynamicArray<int32_t>> perRangeNext;
		// app::for_parallel divides [0,sz) into contiguous ranges.
		// We use one buffer per "slot" -- over-allocate and trim after.
		const int maxSlots = 64;
		perRangeNext.resize((size_t)maxSlots);

		std::atomic<int> slotCounter{0};
		app::for_parallel(0, sz, [&](int start, int end) {
			const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % maxSlots;
			core::DynamicArray<int32_t> &local = perRangeNext[slot];
			for (int i = start; i < end; ++i) {
				const glm::ivec3 cur = lvl.toCell(frontier[i]);
				for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
					const glm::ivec3 nb = cur + off * lvl.cellSize;
					if (!lvl.inBounds(nb)) continue;
					if (lvl.solid.count(nb) > 0) continue;
					const int nbIdx = lvl.toIdx(nb);
					if (lvl.state[nbIdx] != kStateEmpty) continue;
					// Idempotent write: multiple threads may write the same value -- safe
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
static int buildExterior(Level &lvl, const Level &prevLevel) {
	core::DynamicArray<int32_t> frontier;
	frontier.reserve(lvl.dimX * lvl.dimY * 4); // rough seed count estimate

	// Seed from all 6 faces of the grid
	auto seedCell = [&](int cx, int cy, int cz) {
		const glm::ivec3 cell(cx, cy, cz);
		if (lvl.solid.count(cell) > 0) return;
		const int idx = lvl.toIdx(cell);
		if (lvl.state[idx] != kStateEmpty) return;
		lvl.state[idx] = kStateExterior;
		frontier.push_back((int32_t)idx);
	};

	const int maxX = lvl.gridLower.x + (lvl.dimX - 1) * lvl.cellSize;
	const int maxY = lvl.gridLower.y + (lvl.dimY - 1) * lvl.cellSize;
	const int maxZ = lvl.gridLower.z + (lvl.dimZ - 1) * lvl.cellSize;

	for (int cy = lvl.gridLower.y; cy <= maxY; cy += lvl.cellSize) {
		for (int cz = lvl.gridLower.z; cz <= maxZ; cz += lvl.cellSize) {
			seedCell(lvl.gridLower.x, cy, cz);
			seedCell(maxX,            cy, cz);
		}
	}
	for (int cx = lvl.gridLower.x + lvl.cellSize; cx < maxX; cx += lvl.cellSize) {
		for (int cz = lvl.gridLower.z; cz <= maxZ; cz += lvl.cellSize) {
			seedCell(cx, lvl.gridLower.y, cz);
			seedCell(cx, maxY,            cz);
		}
	}
	for (int cx = lvl.gridLower.x + lvl.cellSize; cx < maxX; cx += lvl.cellSize) {
		for (int cy = lvl.gridLower.y + lvl.cellSize; cy < maxY; cy += lvl.cellSize) {
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
				const glm::ivec3 cell = lvl.toCell((int)i);
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
		const int ix = (int)(i / ((size_t)lvl.dimY * lvl.dimZ));
		const int iy = (int)((i / (size_t)lvl.dimZ) % (size_t)lvl.dimY);
		const int iz = (int)(i % (size_t)lvl.dimZ);
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
static int buildInteriorAllSeeds(Level &lvl) {
	size_t seedIdx = 0;
	if (!findInteriorSeedFromCenter(seedIdx, lvl)) {
		return 0;
	}
	lvl.state[seedIdx] = kStateInterior;
	core::DynamicArray<int32_t> queue;
	queue.push_back((int32_t)seedIdx);
	int head = 0;
	while (head < (int)queue.size()) {
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
	return (int)queue.size();
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

static int countCellsByState(const Level &lvl, uint8_t s) {
	int n = 0;
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
	const size_t total = fine.state.size();
	const int64_t totalS = (int64_t)total;
	app::for_parallel(0, (int)glm::min(totalS, (int64_t)INT_MAX), [&](int start, int end) {
		for (int64_t i = (int64_t)start; i < (int64_t)end; ++i) {
			if (fine.state[(size_t)i] == kStateSolid) continue;
			const glm::ivec3 cell = fine.toCell((size_t)i);
			const uint8_t parentState = prev.cellState(toCellOrigin(cell, prev.cellSize));
			if (parentState == kStateExterior) {
				fine.state[(size_t)i] = kStateExterior;
			} else if (parentState == kStateInterior) {
				fine.state[(size_t)i] = kStateInterior;
			}
		}
	});
	// Handle remainder beyond INT_MAX (only on extremely large grids)
	for (size_t i = (size_t)glm::min(totalS, (int64_t)INT_MAX); i < total; ++i) {
		if (fine.state[i] == kStateSolid) continue;
		const glm::ivec3 cell = fine.toCell(i);
		const uint8_t parentState = prev.cellState(toCellOrigin(cell, prev.cellSize));
		if (parentState == kStateExterior) {
			fine.state[i] = kStateExterior;
		} else if (parentState == kStateInterior) {
			fine.state[i] = kStateInterior;
		}
	}
}

// Scan the grid and build exterior and interior frontiers:
// cells that are exterior/interior AND have at least one kStateEmpty neighbor.
static void buildFrontiers(const Level &lvl,
							core::DynamicArray<size_t> &extFrontier,
							core::DynamicArray<size_t> &intFrontier) {
	const size_t total = lvl.state.size();
	const int nSlots = 64;
	core::DynamicArray<core::DynamicArray<size_t>> perExt, perInt;
	perExt.resize((size_t)nSlots);
	perInt.resize((size_t)nSlots);

	// Do NOT reserve based on grid size -- frontier is proportional to surface area, not volume.

	std::atomic<int> slotCounter{0};
	app::for_parallel(0, (int)glm::min((size_t)INT_MAX, total), [&](int start, int end) {
		const int slot = slotCounter.fetch_add(1, std::memory_order_relaxed) % nSlots;
		core::DynamicArray<size_t> &locExt = perExt[(size_t)slot];
		core::DynamicArray<size_t> &locInt = perInt[(size_t)slot];
		for (int64_t i = (int64_t)start; i < (int64_t)end; ++i) {
			const uint8_t s = lvl.state[(size_t)i];
			if (s != kStateExterior && s != kStateInterior) continue;
			const glm::ivec3 cur = lvl.toCell((size_t)i);
			for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nb = cur + off * lvl.cellSize;
				if (!lvl.inBounds(nb)) continue;
				if (lvl.state[lvl.toIdx(nb)] == kStateEmpty) {
					if (s == kStateExterior) locExt.push_back((size_t)i);
					else locInt.push_back((size_t)i);
					break;
				}
			}
		}
	});

	size_t extTotal = 0, intTotal = 0;
	for (int s = 0; s < nSlots; ++s) {
		extTotal += perExt[(size_t)s].size();
		intTotal += perInt[(size_t)s].size();
	}
	extFrontier.reserve(extTotal);
	intFrontier.reserve(intTotal);
	for (int s = 0; s < nSlots; ++s) {
		for (size_t idx : perExt[(size_t)s]) extFrontier.push_back(idx);
		for (size_t idx : perInt[(size_t)s]) intFrontier.push_back(idx);
	}
}

// 256M entries × 8 bytes = 2 GB per frontier array. Two arrays alive at once = 4 GB max for BFS.
static constexpr size_t kMaxFrontierCells = 256u * 1024u * 1024u;

static core::DynamicArray<glm::ivec3> runBidirectionalBFS(Level &lvl) {
	core::DynamicArray<size_t> extFrontier, intFrontier;
	buildFrontiers(lvl, extFrontier, intFrontier);

	if (extFrontier.size() > kMaxFrontierCells || intFrontier.size() > kMaxFrontierCells) {
		Log::warn("holemap BFS: initial frontier too large (ext=%zu int=%zu, limit=%zu) -- "
				  "cell size %d is too fine for this model, use a larger minCellSize",
				  extFrontier.size(), intFrontier.size(), kMaxFrontierCells, lvl.cellSize);
		return {};
	}

	core::DynamicArray<glm::ivec3> holeCells;
	holeCells.reserve(256);

	int bfsRound = 0;
	uint64_t bfsStartMs = core::TimeProvider::systemMillis();
	uint64_t lastLogMs = bfsStartMs;
	while (!extFrontier.empty() || !intFrontier.empty()) {
		++bfsRound;
		const uint64_t nowMs = core::TimeProvider::systemMillis();
		if (nowMs - lastLogMs >= 5000u) {
			lastLogMs = nowMs;
			Log::info("holemap BFS: round %d, ext=%zu int=%zu holes=%d elapsed=%.1fs",
					  bfsRound, extFrontier.size(), intFrontier.size(),
					  (int)holeCells.size(), (double)(nowMs - bfsStartMs) / 1000.0);
		}
		core::DynamicArray<size_t> nextExt;
		nextExt.reserve(extFrontier.size());
		for (size_t idx : extFrontier) {
			const glm::ivec3 cur = lvl.toCell(idx);
			for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nb = cur + off * lvl.cellSize;
				if (!lvl.inBounds(nb)) continue;
				const size_t nbIdx = lvl.toIdx(nb);
				const uint8_t nbState = lvl.state[nbIdx];
				if (nbState == kStateSolid || nbState == kStateExterior || nbState == kStateBlocked) continue;
				if (nbState == kStateInterior) {
					lvl.state[nbIdx] = kStateBlocked;
				} else {
					lvl.state[nbIdx] = kStateExterior;
					nextExt.push_back(nbIdx);
				}
			}
		}
		extFrontier = core::move(nextExt);
		if (extFrontier.size() > kMaxFrontierCells) {
			Log::warn("holemap BFS: exterior frontier exceeded %zu cells at cell size %d -- aborting level",
					  kMaxFrontierCells, lvl.cellSize);
			return holeCells;
		}

		core::DynamicArray<size_t> nextInt;
		nextInt.reserve(intFrontier.size());
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
		if (intFrontier.size() > kMaxFrontierCells) {
			Log::warn("holemap BFS: interior frontier exceeded %zu cells at cell size %d -- aborting level",
					  kMaxFrontierCells, lvl.cellSize);
			return holeCells;
		}
	}

	return holeCells;
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
			int s = coarseCellSize;
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
		const int col = (int)nodes.size() - (int)coarse.solid.size();
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
			  (int)coarse.solid.size());

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
		const int hole = buildExterior(coarse, emptyPrev);
		(void)hole;
		const int extCount = countCellsByState(coarse, kStateExterior);
		Log::info("3dprint faceclassify: coarse exterior: %d cells (%.2fs)", extCount, elapsedSince(t));

		t = core::TimeProvider::systemMillis();
		{
			const int intCount = buildInteriorAllSeeds(coarse);
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
					  fineSize, (int)lvl.solid.size(), elapsedSince(t));

			// Mark solid in state array so BFS skips them
			for (const auto &kv : lvl.solid) {
				if (lvl.inBounds(kv.first)) lvl.state[(size_t)lvl.toIdx(kv.first)] = kStateSolid;
			}

			t = core::TimeProvider::systemMillis();
			const int hole = buildExterior(lvl, *prevLevel);
			const int extCount = countCellsByState(lvl, kStateExterior);
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
				const int intCount = buildInteriorAllSeeds(lvl);
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
	for (int i = (int)fineLevels.size() - 1; i >= 0; --i) levelStack.push_back(&fineLevels[i]);
	levelStack.push_back(&coarse);
	const Level &finestLevel = *levelStack[0];

	// -----------------------------------------------------------------------
	// Step 4: Parallel ray cast classification per node.
	// -----------------------------------------------------------------------
	core::DynamicArray<core::DynamicArray<ClassifyResult>> results;
	results.resize((size_t)totalNodes);
	std::atomic<int> processed{0};

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

	int totalClassified = 0, nodesTouched = 0;
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
		totalClassified += (int)nodeResults.size();
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
		const int intCount = buildInteriorAllSeeds(coarse);
		if (intCount == 0) {
			Log::warn("3dprint holemap: no enclosed interior at coarse scale -- model may not be sealed");
			return;
		}
		Log::info("3dprint holemap: coarse grid %dx%dx%d -- exterior=%d interior=%d solid=%d",
				  coarse.dimX, coarse.dimY, coarse.dimZ,
				  countCellsByState(coarse, kStateExterior),
				  intCount,
				  (int)coarse.solid.size());
	}

	int numLevels = 0;
	for (int s = coarseCellSize / 2; s >= minCellSize; s /= 2) ++numLevels;
	ProgressTimer timer("holemap", numLevels);
	int levelsProcessed = 0;

	core::DynamicArray<glm::ivec3> allHoleCells;
	allHoleCells.reserve(4096);
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
				  fineSize, fine.dimX, fine.dimY, fine.dimZ, (int)fine.solid.size());

		initFineFromPrev(fine, prev);
		Log::info("3dprint holemap: level %d initFromPrev done", fineSize);

		const core::DynamicArray<glm::ivec3> holeCells = runBidirectionalBFS(fine);
		timer.addVoxels((int64_t)holeCells.size());
		timer.tick(++levelsProcessed);

		Log::info("3dprint holemap: level %d BFS done -> %d hole cell(s)", fineSize, (int)holeCells.size());

		for (const glm::ivec3 &h : holeCells) {
			allHoleCells.push_back(h);
		}

		prev = core::move(fine);
	}

	if (allHoleCells.empty()) {
		Log::info("3dprint holemap: model appears watertight down to cell size %d", minCellSize);
		return;
	}
	Log::info("3dprint holemap: %d total hole cell(s) across all levels", (int)allHoleCells.size());

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

	int totalColored = 0;
	int nodesTouched = 0;
	for (const glm::ivec3 &solidCell : adjacentSolids) {
		auto it = prev.solid.find(solidCell);
		if (it == prev.solid.end()) continue;
		const NodeInfo &ni = nodes[it->second];
		scenegraph::SceneGraphNode &node = graph.node(ni.nodeId);
		palette::Palette &pal = node.palette();

		int skipColorIdx = palette::PaletteColorNotFound;
		for (int dz = 0; dz < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dz)
			for (int dy = 0; dy < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dy)
				for (int dx = 0; dx < finalCellSize && skipColorIdx == palette::PaletteColorNotFound; ++dx) {
					const glm::ivec3 localPos = transformPoint(ni.invWorldMat,
					                                           solidCell + glm::ivec3(dx, dy, dz));
					if (!ni.rv->region().containsPoint(localPos)) continue;
					const voxel::Voxel &v = ni.rv->voxel(localPos);
					if (!voxel::isAir(v.getMaterial()))
						skipColorIdx = (int)v.getColor();
				}

		uint8_t holeColorIdx = 0;
		const bool paletteChanged = pal.tryAdd(kHoleColor, true, &holeColorIdx, true, skipColorIdx);

		voxel::RawVolumeWrapper wrapper(ni.rv);
		for (int dz = 0; dz < finalCellSize; ++dz)
			for (int dy = 0; dy < finalCellSize; ++dy)
				for (int dx = 0; dx < finalCellSize; ++dx) {
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

void runHoleFill(SceneManager *sceneMgr, int minCellSize) {
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

	{
		Level emptyPrev;
		buildExterior(coarse, emptyPrev);
	}
	{
		const int intCount = buildInteriorAllSeeds(coarse);
		if (intCount == 0) {
			Log::warn("3dprint holefill: no enclosed interior at coarse scale -- model may not be sealed");
			return;
		}
		Log::info("3dprint holefill: coarse interior: %d cell(s)", intCount);
	}

	int numLevels = 0;
	for (int s = coarseCellSize / 2; s >= minCellSize; s /= 2) ++numLevels;
	ProgressTimer timer("holefill", numLevels + 1); // +1 for fill phase
	int levelsProcessed = 0;

	core::DynamicArray<glm::ivec3> allHoleCells;
	allHoleCells.reserve(4096);
	Level prev = coarse;

	// 1 GB state cap: beyond this the initFineFromPrev scan and BFS become impractical.
	static constexpr size_t kMaxStateCells = 1024ull * 1024ull * 1024ull;

	for (int fineSize = coarseCellSize / 2; fineSize >= minCellSize; fineSize /= 2) {
		Level fine;
		fine.cellSize = fineSize;
		fine.initGrid(gridLower, gridUpper);

		const size_t stateCells = (size_t)fine.dimX * (size_t)fine.dimY * (size_t)fine.dimZ;
		if (stateCells > kMaxStateCells) {
			Log::warn("3dprint holefill: level %d grid %dx%dx%d = %zu cells (%.1f GB state) exceeds "
					  "1 GB budget -- stopping refinement here",
					  fineSize, fine.dimX, fine.dimY, fine.dimZ, stateCells,
					  (double)stateCells / (1024.0 * 1024.0 * 1024.0));
			break;
		}

		{
			ProgressTimer levelTimer("holefill solidHash", totalNodes);
			buildSolidHash(fine, nodes, &levelTimer);
		}
		for (const auto &kv : fine.solid)
			if (fine.inBounds(kv.first)) fine.state[(size_t)fine.toIdx(kv.first)] = kStateSolid;
		Log::info("3dprint holefill: level %d solidHash done -- grid %dx%dx%d solid=%d",
				  fineSize, fine.dimX, fine.dimY, fine.dimZ, (int)fine.solid.size());

		initFineFromPrev(fine, prev);
		Log::info("3dprint holefill: level %d initFromPrev done", fineSize);

		const core::DynamicArray<glm::ivec3> holeCells = runBidirectionalBFS(fine);
		timer.addVoxels((int64_t)holeCells.size());
		timer.tick(++levelsProcessed);

		Log::info("3dprint holefill: level %d BFS done -> %d hole cell(s)", fineSize, (int)holeCells.size());
		for (const glm::ivec3 &h : holeCells)
			allHoleCells.push_back(h);

		prev = core::move(fine);
	}

	if (allHoleCells.empty()) {
		Log::info("3dprint holefill: model appears watertight down to cell size %d", minCellSize);
		return;
	}

	const int finalCellSize = prev.cellSize;
	Log::info("3dprint holefill: %d hole cell(s) at cell size %d -- filling...",
			  (int)allHoleCells.size(), finalCellSize);

	static constexpr color::RGBA kFillColor(0, 220, 0, 255);

	struct FillPos {
		int nodeIdx;
		glm::ivec3 localPos;
	};
	core::DynamicArray<FillPos> fills;
	fills.reserve(allHoleCells.size() * (size_t)(finalCellSize * finalCellSize * finalCellSize));

	int totalSkipped = 0;
	for (const glm::ivec3 &holeCell : allHoleCells) {
		// Find the nearest adjacent solid node (O(6) via solid hash, not O(all nodes)).
		// The hole cell borders a solid cell; that solid cell knows its node index.
		int bestNodeIdx = -1;
		int bestDistSq = INT_MAX;
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			const glm::ivec3 adjCell = holeCell + off * finalCellSize;
			auto it = prev.solid.find(adjCell);
			if (it == prev.solid.end()) continue;
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
			totalSkipped += finalCellSize * finalCellSize * finalCellSize;
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

	fills.sort([](const FillPos &a, const FillPos &b) { return a.nodeIdx < b.nodeIdx; });

	// Apply fills one node at a time.
	// If any fill position falls outside the current node region (inter-node gap),
	// resize the node to cover it before writing.
	int totalFilled = 0;
	int nodesResized = 0;
	int nodesTouched = 0;
	int fillIdx = 0;
	while (fillIdx < (int)fills.size()) {
		const int curNode = fills[(size_t)fillIdx].nodeIdx;
		const int nodeId = nodes[(size_t)curNode].nodeId;

		// Compute union region of all fills for this node
		voxel::Region fillRegion = nodes[(size_t)curNode].rv->region();
		for (int fi = fillIdx; fi < (int)fills.size() && fills[(size_t)fi].nodeIdx == curNode; ++fi) {
			fillRegion.accumulate(fills[(size_t)fi].localPos);
		}

		// Resize if any fill position extends beyond the current region
		if (fillRegion != nodes[(size_t)curNode].rv->region()) {
			sceneMgr->nodeResize(nodeId, fillRegion);
			nodes[(size_t)curNode].rv = sceneMgr->volume(nodeId);
			++nodesResized;
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
				++totalFilled;
			}
			++fillIdx;
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
	}

	timer.addVoxels((int64_t)totalFilled);
	timer.tick(++levelsProcessed);
	Log::info("3dprint holefill: filled %d voxel(s) across %d node(s) (%d resized), "
			  "skipped %d (no adjacent solid node)",
			  totalFilled, nodesTouched, nodesResized, totalSkipped);
}

} // namespace printing
} // namespace voxedit
