/**
 * @file
 */

#pragma once

#include "core/GLM.h"
#include "core/collection/DynamicArray.h"
#include "core/collection/DynamicSet.h"
#include "voxel/Connectivity.h"
#include "voxel/Region.h"
#include "voxel/Voxel.h"
#include <glm/common.hpp>
#include <glm/vec3.hpp>

namespace voxelutil {

/**
 * @brief Point-in-polygon test using ray casting in 2D face-plane coordinates
 * @param path Polygon vertices
 * @param pu U-axis coordinate of the test point
 * @param pv V-axis coordinate of the test point
 * @param uAxis Index of the U axis in ivec3 (0=x,1=y,2=z)
 * @param vAxis Index of the V axis in ivec3
 */
bool lassoContains(const core::DynamicArray<glm::ivec3> &path, int pu, int pv, int uAxis, int vAxis);

/**
 * @brief Draw the lasso edge between two vertices by scanning the face-normal column
 *        at each Bresenham step and marking the outermost surface voxel.
 * @param sampler Callable(glm::ivec3) -> voxel::Voxel used to sample the volume
 * @param markFunc Callable(int x, int y, int z, voxel::Voxel) called for each hit surface voxel
 */
template<typename Sampler, typename MarkFunc>
void drawLassoEdgeSurface(Sampler &&sampler, const glm::ivec3 &a, const glm::ivec3 &b,
						  int uAxis, int vAxis, int wAxis, bool positiveNormal,
						  const voxel::Region &region, MarkFunc &&markFunc) {
	const int wLo = region.getLowerCorner()[wAxis];
	const int wHi = region.getUpperCorner()[wAxis];

	int u = a[uAxis];
	int v = a[vAxis];
	const int du = glm::abs(b[uAxis] - u);
	const int dv = glm::abs(b[vAxis] - v);
	const int su = a[uAxis] < b[uAxis] ? 1 : -1;
	const int sv = a[vAxis] < b[vAxis] ? 1 : -1;
	int err = du - dv;

	while (true) {
		glm::ivec3 pos;
		pos[uAxis] = u;
		pos[vAxis] = v;
		// Scan from the face inward along W to find the outermost surface voxel
		if (positiveNormal) {
			for (int w = wHi; w >= wLo; --w) {
				pos[wAxis] = w;
				const voxel::Voxel vox = sampler(pos);
				if (!voxel::isAir(vox.getMaterial())) {
					markFunc(pos.x, pos.y, pos.z, vox);
					break;
				}
			}
		} else {
			for (int w = wLo; w <= wHi; ++w) {
				pos[wAxis] = w;
				const voxel::Voxel vox = sampler(pos);
				if (!voxel::isAir(vox.getMaterial())) {
					markFunc(pos.x, pos.y, pos.z, vox);
					break;
				}
			}
		}

		if (u == b[uAxis] && v == b[vAxis]) {
			break;
		}
		const int e2 = 2 * err;
		if (e2 > -dv) {
			err -= dv;
			u += su;
		}
		if (e2 < du) {
			err += du;
			v += sv;
		}
	}
}

/**
 * @brief Flood-fill lasso selection across connected surface voxels inside the polygon.
 *
 * Rasterizes each edge of the polygon with drawLassoEdgeSurface to collect seed voxels
 * (outermost surface hits along the w-axis under each edge pixel), then BFS across 26-
 * connected surface voxels whose (u, v) projection lies inside the polygon. Compared to
 * naively selecting every surface voxel inside the 2D polygon, this avoids picking up
 * far-away structures that merely happen to share the same (u, v) silhouette (e.g. a
 * tower behind a rooftop). Multi-seed seeding along every edge prevents the flood from
 * trapping in a concave corner that the first seed alone couldn't reach.
 *
 * @param volume Read-only volume with voxel(x,y,z) access (RawVolume or any adapter
 *               exposing the same interface, e.g. ModifierVolumeWrapper).
 * @param path Polygon vertices (ordered).
 * @param uAxis Index (0/1/2) of the polygon U axis in ivec3.
 * @param vAxis Index of the polygon V axis.
 * @param wAxis Index of the face-normal (depth) axis.
 * @param positiveNormal True if the lasso face normal points in the +w direction.
 * @param region Volume region used for bounds checks and edge rasterization.
 * @param markFunc Callable(int x, int y, int z, const voxel::Voxel &) invoked for each
 *                 surface voxel reached by the flood. Expected to apply the selection.
 */
template<typename Volume, typename MarkFunc>
void lassoFloodFillSurface(const Volume &volume,
						   const core::DynamicArray<glm::ivec3> &path,
						   int uAxis, int vAxis, int wAxis, bool positiveNormal,
						   const voxel::Region &region, MarkFunc &&markFunc) {
	if (path.size() < 3) {
		return;
	}

	core::DynamicSet<glm::ivec3, 1031, glm::hash<glm::ivec3>> visited;

	auto sampler = [&](const glm::ivec3 &p) { return volume.voxel(p.x, p.y, p.z); };
	auto isSurfaceSolid = [&](const glm::ivec3 &p) {
		if (!region.containsPoint(p)) {
			return false;
		}
		const voxel::Voxel v = volume.voxel(p.x, p.y, p.z);
		if (voxel::isAir(v.getMaterial())) {
			return false;
		}
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			const glm::ivec3 n = p + off;
			if (!region.containsPoint(n)) {
				return true;
			}
			if (voxel::isAir(volume.voxel(n.x, n.y, n.z).getMaterial())) {
				return true;
			}
		}
		return false;
	};

	core::DynamicArray<glm::ivec3> stack;
	auto pushSeed = [&](int x, int y, int z, const voxel::Voxel &) {
		stack.push_back(glm::ivec3(x, y, z));
	};

	const int vertexCount = (int)path.size();
	for (int i = 0; i < vertexCount; ++i) {
		const glm::ivec3 &a = path[i];
		const glm::ivec3 &b = path[(i + 1) % vertexCount];
		drawLassoEdgeSurface(sampler, a, b, uAxis, vAxis, wAxis, positiveNormal, region, pushSeed);
	}

	auto enqueueNeighbors = [&](const glm::ivec3 &p) {
		for (const glm::ivec3 &off : voxel::arrayPathfinderFaces) {
			const glm::ivec3 n = p + off;
			if (!visited.has(n)) {
				stack.push_back(n);
			}
		}
		for (const glm::ivec3 &off : voxel::arrayPathfinderEdges) {
			const glm::ivec3 n = p + off;
			if (!visited.has(n)) {
				stack.push_back(n);
			}
		}
		for (const glm::ivec3 &off : voxel::arrayPathfinderCorners) {
			const glm::ivec3 n = p + off;
			if (!visited.has(n)) {
				stack.push_back(n);
			}
		}
	};

	while (!stack.empty()) {
		const glm::ivec3 p = stack.back();
		stack.pop();
		if (visited.has(p)) {
			continue;
		}
		visited.insert(p);
		if (!isSurfaceSolid(p)) {
			continue;
		}
		if (!lassoContains(path, p[uAxis], p[vAxis], uAxis, vAxis)) {
			continue;
		}
		markFunc(p.x, p.y, p.z, volume.voxel(p.x, p.y, p.z));
		enqueueNeighbors(p);
	}
}

} // namespace voxelutil
