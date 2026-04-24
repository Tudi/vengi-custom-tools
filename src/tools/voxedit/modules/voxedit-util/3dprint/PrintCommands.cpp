/**
 * @file
 */

#include "PrintCommands.h"
#include "FaceClassify.h"
#include "Progress.h"
#include "Regrid.h"

#include "app/Async.h"
#include "app/I18N.h"
#include "command/Command.h"
#include "color/RGBA.h"
#include "core/Common.h"
#include "core/Log.h"
#include "core/String.h"
#include "core/collection/DynamicArray.h"
#include "palette/Palette.h"
#include "scenegraph/SceneGraph.h"
#include "scenegraph/SceneGraphNode.h"
#include "voxedit-util/SceneManager.h"
#include "voxel/RawVolume.h"
#include "voxel/RawVolumeWrapper.h"
#include "voxel/Region.h"
#include "voxel/SparseVolume.h"
#include "voxel/Voxel.h"
#include "voxelutil/FillHoles.h"

#include <atomic>
#include <cstdio>
#include <glm/matrix.hpp>

namespace voxedit {
namespace printing {

namespace {

// If debugColor >= 0, return a voxel with the debug palette index (keeping
// the original material/flags/normal/bone). Otherwise return the source voxel unchanged.
static voxel::Voxel maybeRecolor(const voxel::Voxel &src, int debugColor) {
	if (debugColor < 0) {
		return src;
	}
	return voxel::createVoxel(src.getMaterial(), (uint8_t)debugColor, src.getNormal(), src.getFlags(),
							  src.getBoneIdx());
}

struct NodeTask {
	int nodeId;
	voxel::RawVolume *rv;
};

static core::DynamicArray<NodeTask> collectNodeTasks(scenegraph::SceneGraph &graph) {
	core::DynamicArray<NodeTask> tasks;
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		voxel::RawVolume *rv = node.volume();
		if (rv == nullptr) {
			continue;
		}
		tasks.push_back({node.id(), rv});
	}
	return tasks;
}

static uint8_t resolveMarkerColor(palette::Palette &pal, int debugColor) {
	if (debugColor >= 0 && debugColor < 256) {
		return (uint8_t)debugColor;
	}
	static constexpr color::RGBA cyan(0, 255, 255, 255);
	uint8_t idx = 0;
	// replaceSimilar=true so a full 256-entry palette evicts the most redundant slot
	// for our marker color instead of silently returning idx=0.
	pal.tryAdd(cyan, true, &idx, true);
	return idx;
}

static void runSealGaps(SceneManager *sceneMgr, int debugColor) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();
	const core::DynamicArray<NodeTask> tasks = collectNodeTasks(graph);
	const int totalNodes = (int)tasks.size();
	ProgressTimer timer("sealgaps", totalNodes);

	// Phase 1 (parallel): compute sealed voxels per node into thread-local result buffers.
	// RawVolume reads are safe concurrently; no scene-graph writes happen here.
	core::DynamicArray<core::DynamicArray<voxelutil::HoleVoxel>> results;
	results.resize(tasks.size());
	std::atomic<int> processed{0};
	app::for_parallel(0, totalNodes, [&](int start, int end) {
		for (int i = start; i < end; ++i) {
			voxel::SparseVolume sparse;
			sparse.copyFrom(*tasks[i].rv);
			// Edge-connected pairs first, then corner-connected. Corner pass sees fills
			// from the edge pass so it won't double-fill corners that already became
			// 6-connected through an elbow.
			const size_t before = results[i].size();
			voxelutil::sealDiagonalGaps(sparse, results[i]);
			voxelutil::sealCornerGaps(sparse, results[i]);
			timer.addVoxels((int64_t)(results[i].size() - before));
			timer.tick(++processed);
		}
	});

	// Phase 2 (sequential): apply fills on the main thread — RawVolumeWrapper writes,
	// modified() notifications and memento interactions must be single-threaded.
	int totalSealed = 0;
	int nodesTouched = 0;
	for (int i = 0; i < totalNodes; ++i) {
		const core::DynamicArray<voxelutil::HoleVoxel> &sealed = results[i];
		if (sealed.empty()) {
			continue;
		}
		scenegraph::SceneGraphNode &node = graph.node(tasks[i].nodeId);
		const int markerColor = (int)resolveMarkerColor(node.palette(), debugColor);
		voxel::RawVolumeWrapper wrapper(tasks[i].rv);
		for (const voxelutil::HoleVoxel &hv : sealed) {
			wrapper.setVoxel(hv.worldPos, maybeRecolor(hv.voxel, markerColor));
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(tasks[i].nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
		totalSealed += (int)sealed.size();
	}
	Log::info("3dprint sealgaps: sealed %d voxels across %d node(s)", totalSealed, nodesTouched);
}

static void runFillHoles(SceneManager *sceneMgr, int maxHoleSize, int minSolidNeighbors, int debugColor) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();
	const core::DynamicArray<NodeTask> tasks = collectNodeTasks(graph);
	const int totalNodes = (int)tasks.size();
	ProgressTimer timer("fillholes", totalNodes);

	core::DynamicArray<core::DynamicArray<voxelutil::HoleInfo>> results;
	results.resize(tasks.size());
	std::atomic<int> processed{0};
	app::for_parallel(0, totalNodes, [&](int start, int end) {
		for (int i = start; i < end; ++i) {
			voxel::SparseVolume sparse;
			sparse.copyFrom(*tasks[i].rv);
			voxelutil::detectHoles(sparse, results[i], maxHoleSize, minSolidNeighbors);
			int64_t nodeHoleVoxels = 0;
			for (const voxelutil::HoleInfo &h : results[i]) {
				nodeHoleVoxels += (int64_t)h.voxels.size();
			}
			timer.addVoxels(nodeHoleVoxels);
			timer.tick(++processed);
		}
	});

	int totalFilled = 0;
	int totalHoles = 0;
	int nodesTouched = 0;
	for (int i = 0; i < totalNodes; ++i) {
		const core::DynamicArray<voxelutil::HoleInfo> &holes = results[i];
		if (holes.empty()) {
			continue;
		}
		scenegraph::SceneGraphNode &node = graph.node(tasks[i].nodeId);
		const int markerColor = (int)resolveMarkerColor(node.palette(), debugColor);
		voxel::RawVolumeWrapper wrapper(tasks[i].rv);
		int filledInNode = 0;
		for (const voxelutil::HoleInfo &hole : holes) {
			for (const voxelutil::HoleVoxel &hv : hole.voxels) {
				if (wrapper.setVoxel(hv.worldPos, maybeRecolor(hv.voxel, markerColor))) {
					++filledInNode;
				}
			}
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(tasks[i].nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
		totalHoles += (int)holes.size();
		totalFilled += filledInNode;
	}
	Log::info("3dprint fillholes: detected %d hole(s), filled %d voxel(s) across %d node(s)", totalHoles, totalFilled,
			  nodesTouched);
}

static glm::ivec3 transformPoint(const glm::mat4 &m, const glm::ivec3 &p) {
	const glm::vec4 w = m * glm::vec4((float)p.x, (float)p.y, (float)p.z, 1.0f);
	return glm::ivec3((int)glm::round(w.x), (int)glm::round(w.y), (int)glm::round(w.z));
}

static void runSeamFill(SceneManager *sceneMgr, int maxHoleSize, int minSolidNeighbors, int debugColor) {
	scenegraph::SceneGraph &graph = sceneMgr->sceneGraph();

	struct NodeInfo {
		int nodeId;
		glm::mat4 worldMat;
		glm::mat4 invWorldMat;
	};
	core::DynamicArray<NodeInfo> nodeInfos;
	for (auto iter = graph.beginModel(); iter != graph.end(); ++iter) {
		scenegraph::SceneGraphNode &node = *iter;
		if (node.volume() == nullptr) {
			continue;
		}
		NodeInfo info;
		info.nodeId = node.id();
		info.worldMat = graph.worldMatrix(node, 0);
		info.invWorldMat = glm::inverse(info.worldMat);
		nodeInfos.push_back(info);
	}
	if (nodeInfos.empty()) {
		Log::info("3dprint seamfill: no model nodes");
		return;
	}

	ProgressTimer timer("seamfill", (int)nodeInfos.size());
	voxel::SparseVolume combined;
	struct WorldCollector {
		voxel::SparseVolume &target;
		const glm::mat4 &mat;
		void setVoxel(int x, int y, int z, const voxel::Voxel &v) {
			const glm::ivec3 w = transformPoint(mat, glm::ivec3(x, y, z));
			target.setVoxel(w, v);
		}
	};
	{
		int processed = 0;
		for (const NodeInfo &ni : nodeInfos) {
			scenegraph::SceneGraphNode &node = graph.node(ni.nodeId);
			voxel::SparseVolume local;
			local.copyFrom(*node.volume());
			WorldCollector collector{combined, ni.worldMat};
			local.copyTo(collector);
			++processed;
			timer.tick(processed);
		}
	}

	core::DynamicArray<voxelutil::HoleInfo> holes;
	voxelutil::detectHoles(combined, holes, maxHoleSize, minSolidNeighbors);
	if (holes.empty()) {
		Log::info("3dprint seamfill: no seam holes detected");
		return;
	}

	struct NodeFill {
		core::DynamicArray<voxelutil::HoleVoxel> localFills;
		voxel::Region enclosingRegion = voxel::Region::InvalidRegion;
	};
	core::DynamicArray<NodeFill> nodeFills;
	nodeFills.resize(nodeInfos.size());

	int assigned = 0;
	int skipped = 0;
	int64_t seamHoleVoxels = 0;
	for (const voxelutil::HoleInfo &h : holes) {
		seamHoleVoxels += (int64_t)h.voxels.size();
	}
	timer.addVoxels(seamHoleVoxels);
	for (const voxelutil::HoleInfo &hole : holes) {
		for (const voxelutil::HoleVoxel &hv : hole.voxels) {
			const glm::ivec3 &worldPos = hv.worldPos;
			int bestIdx = -1;
			int bestDistSq = -1;
			glm::ivec3 bestLocal(0);
			for (size_t i = 0; i < nodeInfos.size(); ++i) {
				const NodeInfo &ni = nodeInfos[i];
				scenegraph::SceneGraphNode &node = graph.node(ni.nodeId);
				const voxel::Region &rgn = node.volume()->region();
				const glm::ivec3 localI = transformPoint(ni.invWorldMat, worldPos);
				voxel::Region grown = rgn;
				grown.grow(1);
				if (!grown.containsPoint(localI)) {
					continue;
				}
				const glm::ivec3 d = localI - rgn.getCenter();
				const int distSq = d.x * d.x + d.y * d.y + d.z * d.z;
				if (bestIdx < 0 || distSq < bestDistSq) {
					bestIdx = (int)i;
					bestDistSq = distSq;
					bestLocal = localI;
				}
			}
			if (bestIdx < 0) {
				++skipped;
				continue;
			}
			++assigned;
			NodeFill &nf = nodeFills[bestIdx];
			voxelutil::HoleVoxel entry;
			entry.worldPos = bestLocal;
			entry.voxel = hv.voxel;
			nf.localFills.push_back(entry);
			if (nf.enclosingRegion.isValid()) {
				nf.enclosingRegion.accumulate(bestLocal);
			} else {
				nf.enclosingRegion = voxel::Region(bestLocal, bestLocal);
			}
		}
	}

	int nodesTouched = 0;
	int totalFilled = 0;
	for (size_t i = 0; i < nodeInfos.size(); ++i) {
		NodeFill &nf = nodeFills[i];
		if (nf.localFills.empty()) {
			continue;
		}
		const int nodeId = nodeInfos[i].nodeId;
		scenegraph::SceneGraphNode *node = &graph.node(nodeId);
		voxel::RawVolume *rv = node->volume();
		const voxel::Region currentRegion = rv->region();
		if (!currentRegion.containsRegion(nf.enclosingRegion)) {
			const glm::ivec3 newMins = glm::min(currentRegion.getLowerCorner(), nf.enclosingRegion.getLowerCorner());
			const glm::ivec3 newMaxs = glm::max(currentRegion.getUpperCorner(), nf.enclosingRegion.getUpperCorner());
			const voxel::Region needed(newMins, newMaxs);
			sceneMgr->nodeResize(nodeId, needed);
			node = &graph.node(nodeId);
			rv = node->volume();
			if (rv == nullptr) {
				continue;
			}
		}

		const uint8_t seamColor = resolveMarkerColor(node->palette(), debugColor);

		voxel::RawVolumeWrapper wrapper(rv);
		int filledInNode = 0;
		for (const voxelutil::HoleVoxel &hv : nf.localFills) {
			const voxel::Voxel colored = voxel::createVoxel(hv.voxel.getMaterial(), seamColor, hv.voxel.getNormal(),
															hv.voxel.getFlags(), hv.voxel.getBoneIdx());
			if (wrapper.setVoxel(hv.worldPos, colored)) {
				++filledInNode;
			}
		}
		if (wrapper.dirtyRegion().isValid()) {
			sceneMgr->modified(nodeId, wrapper.dirtyRegion());
			++nodesTouched;
		}
		totalFilled += filledInNode;
	}

	Log::info("3dprint seamfill: %d hole(s), assigned %d voxel(s), skipped %d, filled %d across %d node(s)",
			  (int)holes.size(), assigned, skipped, totalFilled, nodesTouched);
}

static void dispatch(SceneManager *sceneMgr, const command::CommandArgs &args) {
	if (!args.has("subcommand")) {
		Log::info("3dprint: usage: 3dprint <hello|sealgaps|fillholes|seamfill|regrid|faceclassify> [maxHoleSize|cellSize] [minSolidNeighbors] [debugColor]");
		Log::info("         debugColor: palette index (0-255) for newly placed voxels, or -1 for cyan marker (default -1)");
		Log::info("         all three commands mark new voxels with cyan by default so additions are visually inspectable");
		Log::info("         regrid: rebucket all model nodes into world-aligned cellSize^3 cells (default 128)");
		Log::info("         faceclassify: recolor all voxels: orange=outer surface, blue=inner cavity surface, magenta=thin wall, gray=buried");
		return;
	}
	const core::String &sub = args.str("subcommand");
	const int debugColor = args.intVal("debugColor", -1);
	if (sub == "hello") {
		Log::info("3dprint hello: dispatch works");
		return;
	}
	if (sub == "sealgaps") {
		runSealGaps(sceneMgr, debugColor);
		return;
	}
	if (sub == "fillholes") {
		int maxHoleSize = args.intVal("maxHoleSize", 1000);
		if (maxHoleSize <= 0) {
			maxHoleSize = 1000;
		}
		const int minSolidNeighbors = args.intVal("minSolidNeighbors", 3);
		runFillHoles(sceneMgr, maxHoleSize, minSolidNeighbors, debugColor);
		return;
	}
	if (sub == "seamfill") {
		int maxHoleSize = args.intVal("maxHoleSize", 1000);
		if (maxHoleSize <= 0) {
			maxHoleSize = 1000;
		}
		const int minSolidNeighbors = args.intVal("minSolidNeighbors", 3);
		runSeamFill(sceneMgr, maxHoleSize, minSolidNeighbors, debugColor);
		return;
	}
	if (sub == "regrid") {
		// maxHoleSize arg slot is reused as cellSize (avoids adding yet another arg to the schema).
		int cellSize = args.intVal("maxHoleSize", 128);
		if (cellSize < 1) {
			cellSize = 128;
		}
		runRegrid(sceneMgr, cellSize);
		return;
	}
	if (sub == "faceclassify") {
		runFaceClassify(sceneMgr);
		return;
	}
	Log::warn("3dprint: unknown subcommand '%s'", sub.c_str());
}

} // namespace

void registerCommands(SceneManager *sceneMgr) {
	command::Command::registerCommand("3dprint")
		.addArg({"subcommand", command::ArgType::String, false, "", "hello|sealgaps|fillholes|seamfill|regrid|faceclassify"})
		.addArg({"maxHoleSize", command::ArgType::Int, true, "1000", "Max voxel count per hole (fillholes/seamfill); also cellSize for regrid (default 128)"})
		.addArg({"minSolidNeighbors", command::ArgType::Int, true, "3",
				 "Min solid 6-neighbors for a hole seed (fillholes only)"})
		.addArg({"debugColor", command::ArgType::Int, true, "-1",
				 "Palette index (0-255) to force-colorize newly placed voxels, or -1 for auto"})
		.setHandler([sceneMgr](const command::CommandArgs &args) { dispatch(sceneMgr, args); })
		.setHelp(_("3D print preparation commands. Usage: 3dprint <hello|sealgaps|fillholes|seamfill|regrid|faceclassify> [args...]"));
}

} // namespace printing
} // namespace voxedit
