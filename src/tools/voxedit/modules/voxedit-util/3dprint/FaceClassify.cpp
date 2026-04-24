/**
 * @file
 */

#include "FaceClassify.h"
#include "Progress.h"

#include "app/Async.h"
#include "color/RGBA.h"
#include "core/GLM.h"
#include "core/Log.h"
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

namespace voxedit {
namespace printing {

namespace {

struct ClassifyResult {
	glm::ivec3 localPos;
	uint8_t tag;
};

static constexpr uint8_t kTagOuter = 0;
static constexpr uint8_t kTagInner = 1;
static constexpr uint8_t kTagThin = 2;
static constexpr uint8_t kTagBuried = 3;

static glm::ivec3 transformPoint(const glm::mat4 &m, const glm::ivec3 &p) {
	const glm::vec4 w = m * glm::vec4((float)p.x, (float)p.y, (float)p.z, 1.0f);
	return glm::ivec3((int)glm::round(w.x), (int)glm::round(w.y), (int)glm::round(w.z));
}

static int floorDiv(int a, int b) {
	return a / b - (a % b != 0 && (a ^ b) < 0);
}

static glm::ivec3 toCellOrigin(const glm::ivec3 &worldPos, int cellSize) {
	return glm::ivec3(floorDiv(worldPos.x, cellSize) * cellSize, floorDiv(worldPos.y, cellSize) * cellSize,
					  floorDiv(worldPos.z, cellSize) * cellSize);
}

struct NodeInfo {
	int nodeId;
	voxel::RawVolume *rv;
	glm::mat4 worldMat;
	glm::mat4 invWorldMat;
	glm::ivec3 cellOrigin;
};

using CellHash = std::unordered_map<glm::ivec3, int, glm::hash<glm::ivec3>>;
using CellSet = std::unordered_set<glm::ivec3, glm::hash<glm::ivec3>>;

static bool isSolid(const glm::ivec3 &worldPos, int cellSize, const CellHash &solidCellHash,
					const core::DynamicArray<NodeInfo> &nodes) {
	const glm::ivec3 cell = toCellOrigin(worldPos, cellSize);
	auto it = solidCellHash.find(cell);
	if (it == solidCellHash.end()) {
		return false;
	}
	const NodeInfo &ni = nodes[it->second];
	const glm::ivec3 local = transformPoint(ni.invWorldMat, worldPos);
	return ni.rv->region().containsPoint(local) && !voxel::isAir(ni.rv->voxel(local).getMaterial());
}

} // namespace

void runFaceClassify(SceneManager *sceneMgr) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	core::DynamicArray<NodeInfo> nodes;
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) {
			continue;
		}
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
	{
	ProgressTimer timer("faceclassify", totalNodes);

	// -------------------------------------------------------------------------
	// Step 1: World bbox + cell size + solid cell hash.
	// -------------------------------------------------------------------------
	voxel::Region worldBbox = voxel::Region::InvalidRegion;
	int cellSize = 0;
	{
		std::unordered_map<int, int> widthCount;
		for (const NodeInfo &ni : nodes) {
			widthCount[ni.rv->region().getWidthInVoxels()]++;
		}
		int bestCount = 0;
		for (const auto &kv : widthCount) {
			if (kv.second > bestCount) {
				bestCount = kv.second;
				cellSize = kv.first;
			}
		}
	}
	if (cellSize <= 0) {
		Log::error("3dprint faceclassify: could not determine cell size from nodes — run 3dprint regrid first");
		return;
	}
	Log::info("3dprint faceclassify: cell size %d", cellSize);

	CellHash solidCellHash;
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
			if (worldBbox.isValid()) {
				worldBbox.accumulate(wc);
			} else {
				worldBbox = voxel::Region(wc, wc);
			}
		}
		// Snap to the cell grid so BFS steps (multiples of cellSize from gridLower)
		// land on actual node origins. Raw world corner is NOT necessarily cell-aligned.
		ni.cellOrigin = toCellOrigin(transformPoint(ni.worldMat, r.getLowerCorner()), cellSize);
		solidCellHash[ni.cellOrigin] = i;
	}
	// Verify: count how many nodes landed on a non-unique snapped origin (collision = two nodes in one cell)
	{
		int collisions = (int)nodes.size() - (int)solidCellHash.size();
		if (collisions > 0) {
			Log::warn("3dprint faceclassify: %d node(s) share a snapped cell origin — overlapping nodes after regrid?", collisions);
		}
	}
	Log::info("3dprint faceclassify: world bbox %d x %d x %d, %d solid cells",
			  worldBbox.getWidthInVoxels(), worldBbox.getHeightInVoxels(), worldBbox.getDepthInVoxels(),
			  (int)solidCellHash.size());

	// -------------------------------------------------------------------------
	// Step 2: Interior detection via exterior-first flood fill.
	//
	// Two symmetric sets of boxes:
	//   exteriorCells — all cells reachable from the cell grid boundary (the "exterior house")
	//   interiorSet   — all enclosed cells not reachable from outside (the "interior house")
	//
	// The ray cast in Step 3 uses BOTH sets symmetrically:
	//   ray enters exteriorCells → OUTER
	//   ray enters interiorSet   → INNER
	// The world bbox is only a safety bound to prevent runaway rays.
	//
	// Steps:
	//   a) 6-connectivity flood fill from all 6 faces of the cell grid boundary
	//      → exteriorCells (all cells reachable from outside).
	//   b) Any empty cell NOT in exteriorCells = truly enclosed interior.
	//   c) 26-connectivity flood fill from first interior cell → interiorSet.
	//      (26-connectivity lets the interior fill navigate diagonal passages.)
	//
	// Warning: if NO interior cells exist the model has no rooms enclosed at
	// this cell scale and needs sealing first.
	// -------------------------------------------------------------------------
	CellSet exteriorCells;
	CellSet interiorSet;
	{
		// Derive grid bounds from actual node cell origins — NOT from worldBbox corners.
		// worldBbox corners are voxel-level and may land mid-cell, causing toCellOrigin()
		// to snap outward and add phantom empty cells beyond the solid region.
		// Using node origins keeps the grid tight: every cell in [gridLower, gridUpper]
		// corresponds to an actual regrid position.
		// Diagnostic: confirm snapped cell origins are multiples of cellSize
		{
			const int sample = totalNodes < 5 ? totalNodes : 5;
			for (int i = 0; i < sample; ++i) {
				const glm::ivec3 &o = nodes[i].cellOrigin;
				Log::info("3dprint faceclassify: node[%d] snappedOrigin=(%d,%d,%d) mod=(%d,%d,%d) [expect (0,0,0)]",
						  i, o.x, o.y, o.z, o.x % cellSize, o.y % cellSize, o.z % cellSize);
			}
		}

		glm::ivec3 gridLower(INT_MAX, INT_MAX, INT_MAX);
		glm::ivec3 gridUpper(INT_MIN, INT_MIN, INT_MIN);
		for (const NodeInfo &ni : nodes) {
			gridLower = glm::min(gridLower, ni.cellOrigin);
			gridUpper = glm::max(gridUpper, ni.cellOrigin);
		}
		Log::info("3dprint faceclassify: gridLower=(%d,%d,%d) gridUpper=(%d,%d,%d)",
				  gridLower.x, gridLower.y, gridLower.z, gridUpper.x, gridUpper.y, gridUpper.z);
		const int cellsX = (gridUpper.x - gridLower.x) / cellSize + 1;
		const int cellsY = (gridUpper.y - gridLower.y) / cellSize + 1;
		const int cellsZ = (gridUpper.z - gridLower.z) / cellSize + 1;
		Log::info("3dprint faceclassify: cell grid %d x %d x %d = %d cells (solid: %d, empty: %d)",
				  cellsX, cellsY, cellsZ, cellsX * cellsY * cellsZ,
				  (int)solidCellHash.size(), cellsX * cellsY * cellsZ - (int)solidCellHash.size());

		// --- 2a: Exterior flood fill (6-connectivity) from all 6 grid faces ---
		core::DynamicArray<glm::ivec3> extQueue;

		auto seedExt = [&](const glm::ivec3 &cell) {
			if (solidCellHash.count(cell) == 0 && exteriorCells.insert(cell).second) {
				extQueue.push_back(cell);
			}
		};

		// Seed from all cells on each of the 6 faces of the grid
		for (int cy = gridLower.y; cy <= gridUpper.y; cy += cellSize) {
			for (int cz = gridLower.z; cz <= gridUpper.z; cz += cellSize) {
				seedExt({gridLower.x, cy, cz});
				seedExt({gridUpper.x, cy, cz});
			}
		}
		for (int cx = gridLower.x + cellSize; cx < gridUpper.x; cx += cellSize) {
			for (int cz = gridLower.z; cz <= gridUpper.z; cz += cellSize) {
				seedExt({cx, gridLower.y, cz});
				seedExt({cx, gridUpper.y, cz});
			}
		}
		for (int cx = gridLower.x + cellSize; cx < gridUpper.x; cx += cellSize) {
			for (int cy = gridLower.y + cellSize; cy < gridUpper.y; cy += cellSize) {
				seedExt({cx, cy, gridLower.z});
				seedExt({cx, cy, gridUpper.z});
			}
		}

		int extIdx = 0;
		while (extIdx < (int)extQueue.size()) {
			const glm::ivec3 cur = extQueue[extIdx++];
			for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
				const glm::ivec3 nb = cur + off * cellSize;
				if (nb.x < gridLower.x || nb.x > gridUpper.x) continue;
				if (nb.y < gridLower.y || nb.y > gridUpper.y) continue;
				if (nb.z < gridLower.z || nb.z > gridUpper.z) continue;
				if (solidCellHash.count(nb) > 0) continue;
				if (exteriorCells.insert(nb).second) {
					extQueue.push_back(nb);
				}
			}
		}
		// Sanity: exterior + solid must not exceed total grid cells
		const int totalGridCells = cellsX * cellsY * cellsZ;
		const int extCount = (int)exteriorCells.size();
		Log::info("3dprint faceclassify: exterior cells: %d / %d total (solid=%d, remaining=%d)",
				  extCount, totalGridCells, (int)solidCellHash.size(),
				  totalGridCells - (int)solidCellHash.size() - extCount);
		if (extCount > totalGridCells) {
			Log::error("3dprint faceclassify: BUG — exterior (%d) exceeds total grid (%d). Cell alignment still broken.", extCount, totalGridCells);
		}

		// --- 2b: Find first interior seed (empty AND not exterior) ---
		glm::ivec3 interiorSeed(0);
		bool seedFound = false;
		for (int cx = gridLower.x; cx <= gridUpper.x && !seedFound; cx += cellSize) {
			for (int cy = gridLower.y; cy <= gridUpper.y && !seedFound; cy += cellSize) {
				for (int cz = gridLower.z; cz <= gridUpper.z && !seedFound; cz += cellSize) {
					const glm::ivec3 cell(cx, cy, cz);
					if (solidCellHash.count(cell) == 0 && exteriorCells.count(cell) == 0) {
						interiorSeed = cell;
						seedFound = true;
					}
				}
			}
		}

		if (!seedFound) {
			Log::warn("3dprint faceclassify: no enclosed interior cells found at cell size %d. "
					  "All empty cells connect to the exterior. "
					  "The model may need sealing (3dprint sealgaps / fillholes) before classify.",
					  cellSize);
		} else {
			Log::info("3dprint faceclassify: interior seed at (%d,%d,%d)", interiorSeed.x, interiorSeed.y,
					  interiorSeed.z);

			// --- 2c: Interior flood fill (26-connectivity, blocked by solid or exterior) ---
			core::DynamicArray<glm::ivec3> intQueue;
			interiorSet.insert(interiorSeed);
			intQueue.push_back(interiorSeed);

			int intIdx = 0;
			while (intIdx < (int)intQueue.size()) {
				const glm::ivec3 cur = intQueue[intIdx++];
				for (int dx = -1; dx <= 1; ++dx) {
					for (int dy = -1; dy <= 1; ++dy) {
						for (int dz = -1; dz <= 1; ++dz) {
							if (dx == 0 && dy == 0 && dz == 0) {
								continue;
							}
							const glm::ivec3 nb = cur + glm::ivec3(dx, dy, dz) * cellSize;
							if (nb.x < gridLower.x || nb.x > gridUpper.x) continue;
							if (nb.y < gridLower.y || nb.y > gridUpper.y) continue;
							if (nb.z < gridLower.z || nb.z > gridUpper.z) continue;
							if (solidCellHash.count(nb) > 0) continue;
							if (exteriorCells.count(nb) > 0) continue;
							if (interiorSet.insert(nb).second) {
								intQueue.push_back(nb);
							}
						}
					}
				}
			}
			Log::info("3dprint faceclassify: interior fill: %d cells", (int)interiorSet.size());
		}
	} // end Step 2 — exteriorCells and interiorSet remain alive for Step 3

	// -------------------------------------------------------------------------
	// Step 3: Per-node parallel classification via ray cast.
	// Both sets are used symmetrically:
	//   cell in exteriorCells  → OUTER (ray sees exterior space)
	//   cell in interiorSet    → INNER (ray sees interior space)
	//   cell in interiorSet          → INNER
	//   cell in solidCellHash        → check actual voxel; solid = blocked, air = continue
	//   empty non-interior cell      → continue
	// -------------------------------------------------------------------------
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
						if (voxel::isAir(ni.rv->voxel(x, y, z).getMaterial())) {
							continue;
						}
						const glm::ivec3 worldPos = transformPoint(ni.worldMat, glm::ivec3(x, y, z));
						bool facesOuter = false;
						bool facesInner = false;

						for (const glm::ivec3 &faceOff : voxel::arrayPathfinderFaces) {
							glm::ivec3 rayPos = worldPos + faceOff;
							while (true) {
								// Safety: prevent runaway ray beyond the entire world bbox
								if (!worldBbox.containsPoint(rayPos)) {
									break;
								}
								const glm::ivec3 cell = toCellOrigin(rayPos, cellSize);
								if (exteriorCells.count(cell) > 0) {
									facesOuter = true;
									break;
								}
								if (interiorSet.count(cell) > 0) {
									facesInner = true;
									break;
								}
								if (solidCellHash.count(cell) > 0) {
									if (isSolid(rayPos, cellSize, solidCellHash, nodes)) {
										break; // blocked by solid voxel
									}
									// air voxel inside a partially-solid cell — keep going
								}
								rayPos += faceOff;
							}
							if (facesOuter && facesInner) {
								break;
							}
						}

						uint8_t tag;
						if (facesOuter && facesInner) {
							tag = kTagThin;
						} else if (facesOuter) {
							tag = kTagOuter;
						} else if (facesInner) {
							tag = kTagInner;
						} else {
							tag = kTagBuried;
						}
						nodeResults.push_back({glm::ivec3(x, y, z), tag});
					}
				}
			}
			timer.addVoxels((int64_t)nodeResults.size());
			timer.tick(++processed);
		}
	});

	// -------------------------------------------------------------------------
	// Step 4: Apply recoloring (sequential).
	// -------------------------------------------------------------------------
	static constexpr color::RGBA kOuterColor(255, 140, 0, 255);  // orange
	static constexpr color::RGBA kInnerColor(30, 120, 255, 255); // blue
	static constexpr color::RGBA kThinColor(220, 0, 220, 255);   // magenta
	static constexpr color::RGBA kBuriedColor(255, 230, 0, 255); // yellow

	int totalClassified = 0;
	int nodesTouched = 0;
	for (int i = 0; i < totalNodes; ++i) {
		const core::DynamicArray<ClassifyResult> &nodeResults = results[i];
		if (nodeResults.empty()) {
			continue;
		}
		scenegraph::SceneGraphNode &node = graph.node(nodes[i].nodeId);
		palette::Palette &pal = node.palette();
		uint8_t outerIdx = 0, innerIdx = 0, thinIdx = 0, buriedIdx = 0;
		pal.tryAdd(kOuterColor, true, &outerIdx, true);
		pal.tryAdd(kInnerColor, true, &innerIdx, true);
		pal.tryAdd(kThinColor, true, &thinIdx, true);
		pal.tryAdd(kBuriedColor, true, &buriedIdx, true);

		voxel::RawVolumeWrapper wrapper(nodes[i].rv);
		for (const ClassifyResult &r : nodeResults) {
			const voxel::Voxel &orig = nodes[i].rv->voxel(r.localPos);
			uint8_t colorIdx;
			switch (r.tag) {
			case kTagOuter:
				colorIdx = outerIdx;
				break;
			case kTagInner:
				colorIdx = innerIdx;
				break;
			case kTagThin:
				colorIdx = thinIdx;
				break;
			default:
				colorIdx = buriedIdx;
				break;
			}
			wrapper.setVoxel(r.localPos, voxel::createVoxel(orig.getMaterial(), colorIdx, orig.getNormal(),
															  orig.getFlags(), orig.getBoneIdx()));
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(nodes[i].nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
		totalClassified += (int)nodeResults.size();
	}
	Log::info("3dprint faceclassify: classified %d voxels across %d node(s)", totalClassified, nodesTouched);
	} // ProgressTimer destructs here — prints TOTAL elapsed before the legend

	Log::info("3dprint faceclassify: color legend:");
	Log::info("  ORANGE  (255,140,  0) = outer surface — face sees exterior space");
	Log::info("  BLUE    ( 30,120,255) = inner surface — face sees interior space only");
	Log::info("  MAGENTA (220,  0,220) = thin wall     — face sees both exterior and interior");
	Log::info("  YELLOW  (255,230,  0) = buried solid  — no air face at all, fully enclosed by solid");
}

} // namespace printing
} // namespace voxedit
