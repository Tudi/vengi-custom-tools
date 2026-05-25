/**
 * @file
 */

#include "PolygonLasso.h"
#include "Circle.h"
#include "math/Axis.h"
#include "voxedit-util/modifier/ModifierVolumeWrapper.h"
#include "voxedit-util/modifier/brush/Brush.h"
#include "voxel/RawVolume.h"
#include "voxelutil/VolumeSelect.h"
#include "voxelutil/VolumeVisitor.h"
#include <glm/common.hpp>

namespace voxedit {
namespace select {

glm::ivec3 PolygonLasso::snapToGrid(const glm::ivec3 &pos, int resolution) {
	if (resolution <= 1) {
		return pos;
	}
	glm::ivec3 out = pos;
	if (out.x % resolution != 0) {
		out.x = (out.x / resolution) * resolution;
	}
	if (out.y % resolution != 0) {
		out.y = (out.y / resolution) * resolution;
	}
	if (out.z % resolution != 0) {
		out.z = (out.z / resolution) * resolution;
	}
	return out;
}

void PolygonLasso::ellipseAxes(voxel::FaceNames face, int &uAxis, int &vAxis) {
	Circle::ellipseAxes(face, uAxis, vAxis);
}

void PolygonLasso::reset() {
	_path.clear();
	_edgeHistory.clear();
	_rubberBandHistory.clear();
	_accumulating = false;
	_face = voxel::FaceNames::Max;
	_lastCursorPos = glm::ivec3(InvalidCursorCoord);
	_modifiedFlags = SceneModifiedFlags::All;
}

void PolygonLasso::abort(BrushContext &ctx) {
	if (_accumulating) {
		_path.clear();
		_accumulating = false;
		_modifiedFlags = SceneModifiedFlags::All;
	}
}

void PolygonLasso::invalidate() {
	_accumulating = false;
	_path.clear();
	_edgeHistory.clear();
	_rubberBandHistory.clear();
}

void PolygonLasso::popLastPathEntry() {
	if (!_path.empty()) {
		_path.pop();
	}
}

void PolygonLasso::update(const BrushContext &ctx, double nowSeconds) {
	if (_accumulating) {
		if (ctx.cursorPosition != _lastCursorPos) {
			_lastCursorPos = ctx.cursorPosition;
			// Strategy has no markDirty(); SelectBrush::update() forwards via NoUndo modified flag.
			_modifiedFlags = SceneModifiedFlags::NoUndo;
		}
	}
}

voxel::Region PolygonLasso::revertChanges(voxel::RawVolume *volume) {
	voxel::Region dirtyRegion = voxel::Region::InvalidRegion;
	// Revert rubber-band first. Its entries may have captured edge-marked voxels at
	// overlap points; reverting edges afterwards restores the true original state.
	for (auto *entry : _rubberBandHistory) {
		volume->setVoxel(entry->key, entry->value);
		dirtyRegion.accumulate(entry->key);
	}
	_rubberBandHistory.clear();
	for (auto *entry : _edgeHistory) {
		volume->setVoxel(entry->key, entry->value);
		dirtyRegion.accumulate(entry->key);
	}
	_edgeHistory.clear();
	return dirtyRegion;
}

bool PolygonLasso::beginBrush(const BrushContext &ctx, const AABBBrushState &state) {
	if (_accumulating) {
		const glm::ivec3 cursor = snapToGrid(ctx.cursorPosition, ctx.gridResolution);
		// Close the polygon when clicking near the first vertex
		if (_path.size() >= 3) {
			const glm::ivec3 &firstPt = _path[0];
			const int du = glm::abs(cursor[_uAxis] - firstPt[_uAxis]);
			const int dv = glm::abs(cursor[_vAxis] - firstPt[_vAxis]);
			if (du <= CloseThresholdVoxels && dv <= CloseThresholdVoxels) {
				_accumulating = false;
				_modifiedFlags = SceneModifiedFlags::All;
				return true;
			}
		}
		// Add the next vertex to the polygon
		_path.push_back(cursor);
		_modifiedFlags = SceneModifiedFlags::NoUndo;
		return true;
	}
	// Start a new lasso polygon
	_path.clear();
	_edgeHistory.clear();
	_rubberBandHistory.clear();
	_path.reserve(PathInitialReserve);
	_path.push_back(snapToGrid(ctx.cursorPosition, ctx.gridResolution));
	_face = ctx.cursorFace;
	ellipseAxes(_face, _uAxis, _vAxis);
	_faceAxisIdx = math::getIndexForAxis(voxel::faceToAxis(_face));
	_accumulating = true;
	_modifiedFlags = SceneModifiedFlags::NoUndo;
	return true;
}

voxel::Region PolygonLasso::calcRegion(const BrushContext &ctx, const AABBBrushState &state) const {
	// For lasso accumulation, the rubber-band is drawn directly on the real volume
	// (no preview copy). hasPendingChanges() returns true so PreviewManager bails out
	// and the brush executes on the real node. The returned region is just the bbox
	// passed to drawLassoEdgeSurface for column scanning, so it must span the full
	// volume W-range to find surface voxels at any height.
	if (_accumulating && !_path.empty()) {
		const glm::ivec3 &lastPt = _path.back();
		const glm::ivec3 &cursor = ctx.cursorPosition;
		glm::ivec3 mins = glm::min(lastPt, cursor);
		glm::ivec3 maxs = glm::max(lastPt, cursor);
		mins[_faceAxisIdx] = ctx.targetVolumeRegion.getLowerCorner()[_faceAxisIdx];
		maxs[_faceAxisIdx] = ctx.targetVolumeRegion.getUpperCorner()[_faceAxisIdx];
		voxel::Region rubberBandRegion(mins, maxs);
		rubberBandRegion.cropTo(ctx.targetVolumeRegion);
		return rubberBandRegion;
	}
	return ctx.targetVolumeRegion;
}

void PolygonLasso::generate(scenegraph::SceneGraph &sceneGraph, ModifierVolumeWrapper &wrapper, const BrushContext &ctx,
							const voxel::Region &region, const AABBBrushState &state) {
	if (_face == voxel::FaceNames::Max || _path.empty()) {
		return;
	}
	const int uAxis = _uAxis;
	const int vAxis = _vAxis;
	const int wAxis = _faceAxisIdx;
	const bool positiveNormal = voxel::isPositiveFace(_face);

	auto func = [&wrapper](int x, int y, int z, const voxel::Voxel &voxel) {
		if (wrapper.modifierType() == ModifierType::Erase) {
			wrapper.removeFlagAt(x, y, z, voxel::FlagOutline);
		} else {
			wrapper.setFlagAt(x, y, z, voxel::FlagOutline);
		}
	};

	auto restoreHistory = [&](HistoryMap &history) {
		for (auto *entry : history) {
			wrapper.volume()->setVoxel(entry->key, entry->value);
			wrapper.addToDirtyRegion(entry->key);
		}
		history.clear();
	};
	auto bresenhamLength = [uAxis, vAxis](const glm::ivec3 &startPt, const glm::ivec3 &endPt) {
		return (size_t)(glm::max(glm::abs(endPt[uAxis] - startPt[uAxis]), glm::abs(endPt[vAxis] - startPt[vAxis])) + 1);
	};
	auto captureFunc = [&](HistoryMap &history) {
		return [&history, &func](int x, int y, int z, const voxel::Voxel &v) {
			const glm::ivec3 pos(x, y, z);
			// Store original voxel only on first encounter so overlapping segments
			// don't overwrite the true original with an already-marked voxel.
			if (history.find(pos) == history.end()) {
				history.put(pos, v);
			}
			func(x, y, z, v);
		};
	};

	if (_accumulating) {
		const int vertexCount = (int)_path.size();
		// Rubber-band + edges draw directly on the real volume (no preview copy).
		// hasPendingChanges() returns true while accumulating, so PreviewManager skips
		// allocation and this branch runs on the real node volume every frame -
		// sidestepping the 32^3 preview cap that would otherwise hide the line on long
		// diagonals. Revert rubber-band first (may have captured edge-marked voxels at
		// overlaps), then edges, so the volume is back to its original state before we
		// redraw both layers.
		restoreHistory(_rubberBandHistory);
		restoreHistory(_edgeHistory);
		// Reserve both histories up front so the DynamicMap pre-allocates one node
		// block for the expected count instead of allocating on first insert.
		size_t edgeReserve = 0u;
		for (int edgeIdx = 1; edgeIdx < vertexCount; ++edgeIdx) {
			edgeReserve += bresenhamLength(_path[edgeIdx - 1], _path[edgeIdx]);
		}
		_edgeHistory.reserve(edgeReserve);
		auto edgeFunc = captureFunc(_edgeHistory);
		// W-axis scan spans the full target volume so surface voxels at any height are
		// covered across all committed edges.
		for (int edgeIdx = 1; edgeIdx < vertexCount; ++edgeIdx) {
			voxelutil::drawLassoEdgeSurface([&](const glm::ivec3 &pos) { return wrapper.voxel(pos); },
											_path[edgeIdx - 1], _path[edgeIdx], uAxis, vAxis, wAxis, positiveNormal,
											ctx.targetVolumeRegion, edgeFunc);
		}
		// Rubber-band captures whatever the volume looks like after the edge pass; on
		// overlap points, reverting it restores the edge-marked state, and reverting
		// the edge history afterwards restores the true original.
		if (ctx.cursorFace != voxel::FaceNames::Max && vertexCount >= 1) {
			const glm::ivec3 &lastVertex = _path[vertexCount - 1];
			_rubberBandHistory.reserve(bresenhamLength(lastVertex, ctx.cursorPosition));
			auto rubberBandFunc = captureFunc(_rubberBandHistory);
			voxelutil::drawLassoEdgeSurface([&](const glm::ivec3 &pos) { return wrapper.voxel(pos); }, lastVertex,
											ctx.cursorPosition, uAxis, vAxis, wAxis, positiveNormal,
											ctx.targetVolumeRegion, rubberBandFunc);
		}
		_modifiedFlags = SceneModifiedFlags::NoUndo;
		return;
	}

	// Polygon closed: apply selection to surface voxels inside the polygon.
	_modifiedFlags = SceneModifiedFlags::All;
	if (_path.size() < 3) {
		return;
	}
	// Restore rubber-band + edge history voxels BEFORE applying selection so their
	// positions are included in the dirty region. This ensures the undo entry covers
	// ALL transient marks (inside and outside the polygon) and undo reverts cleanly.
	// Rubber-band first so edge-marked overlap points get restored to the true
	// original by the edge revert that follows.
	restoreHistory(_rubberBandHistory);
	restoreHistory(_edgeHistory);
	// Flood-fill from seeds rasterized along every polygon edge, walking 26-connected
	// surface voxels that project inside the polygon. Disjoint structures sharing the
	// (u, v) silhouette (e.g. a tower behind the rooftop) are no longer swept in by
	// the 2D point-in-polygon filter alone.
	voxelutil::lassoFloodFillSurface(wrapper, _path, uAxis, vAxis, wAxis, positiveNormal, ctx.targetVolumeRegion, func);
	_path.clear();
}

void PolygonLasso::redrawEdgesOnVolume(voxel::RawVolume *volume, const voxel::Region &region, voxel::Region &outDirty) {
	const int vertexCount = (int)_path.size();
	if (vertexCount < 2) {
		return;
	}
	const int uAxis = _uAxis;
	const int vAxis = _vAxis;
	const int wAxis = _faceAxisIdx;
	const bool positiveNormal = voxel::isPositiveFace(_face);

	size_t edgeReserve = 0u;
	for (int edgeIdx = 1; edgeIdx < vertexCount; ++edgeIdx) {
		const glm::ivec3 &startPt = _path[edgeIdx - 1];
		const glm::ivec3 &endPt = _path[edgeIdx];
		edgeReserve += (size_t)(glm::max(glm::abs(endPt[uAxis] - startPt[uAxis]),
										 glm::abs(endPt[vAxis] - startPt[vAxis])) +
								1);
	}
	_edgeHistory.reserve(edgeReserve);
	auto edgeFunc = [&](int x, int y, int z, const voxel::Voxel &v) {
		const glm::ivec3 pos(x, y, z);
		if (_edgeHistory.find(pos) == _edgeHistory.end()) {
			_edgeHistory.put(pos, v);
		}
		voxel::Voxel flagged = v;
		flagged.setFlags(flagged.getFlags() | voxel::FlagOutline);
		volume->setVoxel(pos, flagged);
		outDirty.accumulate(pos);
	};
	auto readVoxel = [&](const glm::ivec3 &pos) { return volume->voxel(pos); };
	for (int edgeIdx = 1; edgeIdx < vertexCount; ++edgeIdx) {
		voxelutil::drawLassoEdgeSurface(readVoxel, _path[edgeIdx - 1], _path[edgeIdx], uAxis, vAxis, wAxis,
										positiveNormal, region, edgeFunc);
	}
}

} // namespace select
} // namespace voxedit
