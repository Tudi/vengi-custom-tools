/**
 * @file
 */

#pragma once

#include "core/collection/DynamicArray.h"
#include "core/collection/DynamicSet.h"
#include "core/GLM.h"
#include "voxel/Connectivity.h"
#include "voxel/Voxel.h"

namespace voxel {
class SparseVolume;
}

namespace voxelutil {

struct HoleVoxel {
	glm::ivec3 worldPos;
	voxel::Voxel voxel;
};

struct HoleInfo {
	int id;
	core::DynamicArray<HoleVoxel> voxels;
	glm::ivec3 centroid{0};
	int size() const {
		return (int)voxels.size();
	}
};

struct FillHolesResult {
	core::DynamicArray<HoleInfo> holes;
	int diagonalGapsSealed = 0;
	int totalHolesDetected = 0;
};

/**
 * @brief Seal diagonal gaps in a sparse volume to make the shell 6-connected.
 *
 * For each pair of solid voxels connected only by an edge (not face), fills one of the two
 * "elbow" air voxels between them to create a 6-connected path. Always picks the interior-side
 * elbow (the one with more solid 26-neighbors) to preserve the exterior surface shape.
 *
 * @param[in,out] volume The sparse volume to seal
 * @param[out] outSealed Filled voxel positions and values
 * @return Number of voxels added
 */
int sealDiagonalGaps(voxel::SparseVolume &volume, core::DynamicArray<HoleVoxel> &outSealed);

/**
 * @brief Seal corner-connected gaps in a sparse volume to make the shell 6-connected.
 *
 * Edge-connected gaps (differ in 2 axes) are handled by sealDiagonalGaps. This function
 * handles corner-connected gaps: two solid voxels that differ in all 3 axes, sitting on
 * opposite corners of a 2x2x2 cube. Closing such a gap requires filling one or two
 * intermediate voxels on the shortest face-connected path through the cube.
 *
 * Algorithm per corner-pair:
 * - Enumerate the 6 length-3 paths from s1 to s2 through the cube (pairs of intermediates).
 * - Skip if any path is already fully solid (already 6-connected).
 * - Otherwise choose the path with the minimum number of air intermediates (1 if possible, 2 if all
 *   intermediates are air). Among ties, pick the path whose air intermediates have the highest total
 *   solid-26-neighbor count (interior heuristic, mirrors sealDiagonalGaps).
 *
 * @param[in,out] volume The sparse volume to seal
 * @param[out] outSealed Filled voxel positions and values
 * @return Number of voxels added
 */
int sealCornerGaps(voxel::SparseVolume &volume, core::DynamicArray<HoleVoxel> &outSealed);

/**
 * @brief Detect holes in a sparse volume by scoring near-surface air voxels.
 *
 * For each solid voxel, checks its 6 air neighbors. Air voxels with enough solid neighbors
 * and opposing solid pairs are classified as hole candidates. Connected components of candidates
 * form individual holes.
 *
 * @param volume The sparse volume to analyze
 * @param[out] outHoles Detected holes sorted by size ascending
 * @param maxHoleSize Maximum voxel count per hole (larger components are capped and marked)
 * @param minSolidNeighbors Minimum solid 6-neighbors for a seed candidate (default 3)
 */
void detectHoles(const voxel::SparseVolume &volume, core::DynamicArray<HoleInfo> &outHoles,
				 int maxHoleSize = 1000, int minSolidNeighbors = 3);

/**
 * @brief Count the number of solid 26-neighbors for a position in a sparse volume.
 */
int countSolid26Neighbors(const voxel::SparseVolume &volume, const glm::ivec3 &pos);

} // namespace voxelutil
