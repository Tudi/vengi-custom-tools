/**
 * @file
 */

#pragma once

#include "SelectStrategy.h"
#include "core/GLM.h"
#include "core/collection/DynamicArray.h"
#include "core/collection/DynamicMap.h"
#include "voxel/Face.h"
#include "voxel/Voxel.h"
#include <glm/vec3.hpp>

namespace voxel {
class RawVolume;
} // namespace voxel

namespace voxedit {

class SceneManager;

namespace select {

/**
 * @brief Click-vertex polygonal lasso with surface flood-fill on close.
 *
 * Distinct from the screen-space drag-fill Lasso strategy. The user clicks individual
 * voxels to drop polygon vertices; the brush draws committed edge marks plus a live
 * rubber-band segment directly on the real volume between clicks. On close (click near
 * first vertex), surface voxels are flood-filled from edge seeds inside the polygon,
 * avoiding disjoint structures that share the (u, v) silhouette.
 */
class PolygonLasso : public Strategy {
public:
	/** Max U/V distance from the first vertex that auto-closes the polygon on click */
	static constexpr int CloseThresholdVoxels = 1;
	/** Initial capacity reserved for the polygon vertex list */
	static constexpr int PathInitialReserve = 32;
	/** Sentinel value for the last-cursor-pos check to force a refresh on first update() */
	static constexpr int InvalidCursorCoord = -100000;
	/** Hash bucket count for history maps. Prime, sized for typical polygon edge counts. */
	static constexpr size_t HistoryBuckets = 1031;
	/** Stores original voxel state at positions we temporarily overwrote with FlagOutline
	 *  marks. Hash lookups for duplicate-position checks. */
	using HistoryMap = core::DynamicMap<glm::ivec3, voxel::Voxel, HistoryBuckets, glm::hash<glm::ivec3>>;

private:
	using Super = Strategy;
	SceneManager *_sceneManager;
	/** Polygon vertices accumulated across clicks */
	core::DynamicArray<glm::ivec3> _path;
	/** Original voxels at committed edge positions - reverted on cancel */
	HistoryMap _edgeHistory;
	/** Original voxels at the live rubber-band segment - redrawn on cursor move */
	HistoryMap _rubberBandHistory;
	bool _accumulating = false;
	int _uAxis = 0;
	int _vAxis = 1;
	int _faceAxisIdx = 2;
	voxel::FaceNames _face = voxel::FaceNames::Max;
	glm::ivec3 _lastCursorPos{InvalidCursorCoord};

	static glm::ivec3 snapToGrid(const glm::ivec3 &pos, int resolution);
	static void ellipseAxes(voxel::FaceNames face, int &uAxis, int &vAxis);

public:
	explicit PolygonLasso(SceneManager *sceneManager) : _sceneManager(sceneManager) {
	}

	void generate(scenegraph::SceneGraph &sceneGraph, ModifierVolumeWrapper &wrapper, const BrushContext &ctx,
				  const voxel::Region &region, const AABBBrushState &state) override;
	voxel::Region calcRegion(const BrushContext &ctx, const AABBBrushState &state) const override;
	bool beginBrush(const BrushContext &ctx, const AABBBrushState &state) override;
	void abort(BrushContext &ctx) override;
	void reset() override;
	void update(const BrushContext &ctx, double nowSeconds) override;
	bool active() const override {
		return _accumulating;
	}
	bool needsAdditionalAction(const BrushContext &ctx) const override {
		return false;
	}

	/** PreviewManager-skip predicate so generate() runs on the real volume during accumulation. */
	bool hasPendingChanges() const {
		return _accumulating;
	}
	/** Revert all in-progress edge + rubber-band marks. Returns the dirty region covered. */
	voxel::Region revertChanges(voxel::RawVolume *volume);

	bool accumulating() const {
		return _accumulating;
	}
	const core::DynamicArray<glm::ivec3> &path() const {
		return _path;
	}
	int uAxis() const {
		return _uAxis;
	}
	int vAxis() const {
		return _vAxis;
	}
	int faceAxisIdx() const {
		return _faceAxisIdx;
	}
	voxel::FaceNames face() const {
		return _face;
	}

	/** Discard the in-progress polygon (caller is responsible for cleaning up edge marks on the volume) */
	void invalidate();
	/** Remove the last placed polygon vertex. Caller must redrawEdgesOnVolume() afterwards. */
	void popLastPathEntry();
	/**
	 * @brief Redraw all committed polygon edges directly on a raw volume.
	 *        Updates _edgeHistory with the new set of edge positions.
	 *        Caller is responsible for calling revertChanges() first to clear old marks.
	 */
	void redrawEdgesOnVolume(voxel::RawVolume *volume, const voxel::Region &region, voxel::Region &outDirty);
};

} // namespace select
} // namespace voxedit
