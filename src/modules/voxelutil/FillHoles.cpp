/**
 * @file
 */

#include "FillHoles.h"
#include "VolumeVisitor.h"
#include "core/Log.h"
#include "core/collection/DynamicMap.h"
#include "voxel/Connectivity.h"
#include "voxel/SparseVolume.h"
#include "voxel/Voxel.h"

namespace voxelutil {

int countSolid26Neighbors(const voxel::SparseVolume &volume, const glm::ivec3 &pos) {
	int count = 0;
	for (const glm::ivec3 &offset : voxel::arrayPathfinderFaces) {
		if (!voxel::isAir(volume.voxel(pos + offset).getMaterial())) {
			++count;
		}
	}
	for (const glm::ivec3 &offset : voxel::arrayPathfinderEdges) {
		if (!voxel::isAir(volume.voxel(pos + offset).getMaterial())) {
			++count;
		}
	}
	for (const glm::ivec3 &offset : voxel::arrayPathfinderCorners) {
		if (!voxel::isAir(volume.voxel(pos + offset).getMaterial())) {
			++count;
		}
	}
	return count;
}

/**
 * @brief For two edge-connected solid positions, compute the two elbow positions between them.
 *
 * Edge-connected means they differ in exactly 2 axes. The two elbows are the positions
 * where one axis matches s1 and the other matches s2, and vice versa.
 */
static void computeElbows(const glm::ivec3 &s1, const glm::ivec3 &s2, glm::ivec3 &elbow1, glm::ivec3 &elbow2) {
	// They differ in exactly 2 of 3 axes. For the differing axes, swap one at a time.
	elbow1 = s1;
	elbow2 = s1;
	int diffIdx = 0;
	for (int axis = 0; axis < 3; ++axis) {
		if (s1[axis] != s2[axis]) {
			if (diffIdx == 0) {
				elbow1[axis] = s2[axis];
			} else {
				elbow2[axis] = s2[axis];
			}
			++diffIdx;
		}
	}
}

/**
 * @brief Pick the color for a fill voxel based on its solid neighbors.
 *
 * Returns the most common color among solid 6-neighbors. Falls back to the first
 * solid neighbor color if all are different.
 */
static voxel::Voxel pickFillVoxel(const voxel::SparseVolume &volume, const glm::ivec3 &pos) {
	// count occurrences of each color index among solid 6-neighbors
	static constexpr int MaxColors = 256;
	int colorCounts[MaxColors] = {};
	voxel::Voxel firstSolid;
	bool foundFirst = false;

	for (const glm::ivec3 &offset : voxel::arrayPathfinderFaces) {
		const voxel::Voxel &v = volume.voxel(pos + offset);
		if (!voxel::isAir(v.getMaterial())) {
			colorCounts[v.getColor()]++;
			if (!foundFirst) {
				firstSolid = v;
				foundFirst = true;
			}
		}
	}

	if (!foundFirst) {
		// check 26-neighbors as fallback
		for (const glm::ivec3 &offset : voxel::arrayPathfinderEdges) {
			const voxel::Voxel &v = volume.voxel(pos + offset);
			if (!voxel::isAir(v.getMaterial())) {
				firstSolid = v;
				foundFirst = true;
				break;
			}
		}
		if (!foundFirst) {
			for (const glm::ivec3 &offset : voxel::arrayPathfinderCorners) {
				const voxel::Voxel &v = volume.voxel(pos + offset);
				if (!voxel::isAir(v.getMaterial())) {
					firstSolid = v;
					foundFirst = true;
					break;
				}
			}
		}
	}

	if (!foundFirst) {
		return voxel::createVoxel(voxel::VoxelType::Generic, 0);
	}

	// find the most common color
	int bestColor = firstSolid.getColor();
	int bestCount = 0;
	for (int i = 0; i < MaxColors; ++i) {
		if (colorCounts[i] > bestCount) {
			bestCount = colorCounts[i];
			bestColor = i;
		}
	}
	return voxel::createVoxel(firstSolid.getMaterial(), bestColor);
}

int sealDiagonalGaps(voxel::SparseVolume &volume, core::DynamicArray<HoleVoxel> &outSealed) {
	// use the global VisitedSet typedef from VolumeVisitor.h

	// Collect all solid positions first (cannot modify volume during iteration)
	core::DynamicArray<glm::ivec3> solidPositions;
	struct Collector {
		core::DynamicArray<glm::ivec3> &positions;
		void setVoxel(int x, int y, int z, const voxel::Voxel &v) {
			positions.emplace_back(x, y, z);
		}
	};
	Collector collector{solidPositions};
	volume.copyTo(collector);

	Log::debug("Sealing diagonal gaps: %d solid voxels to check", (int)solidPositions.size());

	// Track which positions we've already sealed to avoid duplicates
	VisitedSet sealed;
	core::DynamicArray<HoleVoxel> toFill;

	for (const glm::ivec3 &pos : solidPositions) {
		// Check 12 edge neighbors
		for (const glm::ivec3 &edgeOffset : voxel::arrayPathfinderEdges) {
			const glm::ivec3 neighbor = pos + edgeOffset;
			if (voxel::isAir(volume.voxel(neighbor).getMaterial())) {
				continue; // edge neighbor is air, not a diagonal connection
			}

			// Both pos and neighbor are solid and edge-connected.
			// Check if they're already 6-connected through a shared face neighbor.
			glm::ivec3 elbow1, elbow2;
			computeElbows(pos, neighbor, elbow1, elbow2);

			const bool e1Solid = !voxel::isAir(volume.voxel(elbow1).getMaterial());
			const bool e2Solid = !voxel::isAir(volume.voxel(elbow2).getMaterial());

			if (e1Solid || e2Solid) {
				// Already 6-connected through at least one elbow
				continue;
			}

			// Both elbows are air. Need to fill one (the interior-side one).
			// Pick the elbow with more solid 26-neighbors (interior heuristic).
			// Only process this pair once (avoid duplicate from the other solid's perspective).
			// Use the lexicographically smaller position as the canonical pair key.
			if (pos.x > neighbor.x || (pos.x == neighbor.x && pos.y > neighbor.y) ||
				(pos.x == neighbor.x && pos.y == neighbor.y && pos.z > neighbor.z)) {
				continue; // let the other direction handle it
			}

			const int score1 = countSolid26Neighbors(volume, elbow1);
			const int score2 = countSolid26Neighbors(volume, elbow2);

			const glm::ivec3 &fillPos = (score1 >= score2) ? elbow1 : elbow2;
			if (sealed.has(fillPos)) {
				continue;
			}
			sealed.insert(fillPos);

			const voxel::Voxel fillVoxel = pickFillVoxel(volume, fillPos);
			toFill.push_back({fillPos, fillVoxel});
		}
	}

	// Apply fills
	for (const HoleVoxel &hv : toFill) {
		volume.setVoxel(hv.worldPos, hv.voxel);
		outSealed.push_back(hv);
	}

	Log::debug("Sealed %d diagonal gaps", (int)toFill.size());
	return (int)toFill.size();
}

int sealCornerGaps(voxel::SparseVolume &volume, core::DynamicArray<HoleVoxel> &outSealed) {
	core::DynamicArray<glm::ivec3> solidPositions;
	struct Collector {
		core::DynamicArray<glm::ivec3> &positions;
		void setVoxel(int x, int y, int z, const voxel::Voxel &v) {
			positions.emplace_back(x, y, z);
		}
	};
	Collector collector{solidPositions};
	volume.copyTo(collector);

	Log::debug("Sealing corner gaps: %d solid voxels to check", (int)solidPositions.size());

	VisitedSet sealed;
	core::DynamicArray<HoleVoxel> toFill;

	for (const glm::ivec3 &s1 : solidPositions) {
		for (const glm::ivec3 &cornerOffset : voxel::arrayPathfinderCorners) {
			const glm::ivec3 s2 = s1 + cornerOffset;
			if (voxel::isAir(volume.voxel(s2).getMaterial())) {
				continue;
			}

			// Canonical pair ordering to avoid duplicate work from the mirrored corner neighbor.
			if (s1.x > s2.x || (s1.x == s2.x && s1.y > s2.y) ||
				(s1.x == s2.x && s1.y == s2.y && s1.z > s2.z)) {
				continue;
			}

			// The 6 intermediate positions of the 2x2x2 cube.
			// s1adj[a] = s1 with axis a flipped to s2's value (face-adjacent to s1).
			// s2adj[a] = s2 with axis a flipped to s1's value (face-adjacent to s2).
			glm::ivec3 s1adj[3];
			glm::ivec3 s2adj[3];
			bool s1adjSolid[3];
			bool s2adjSolid[3];
			for (int a = 0; a < 3; ++a) {
				s1adj[a] = s1;
				s1adj[a][a] = s2[a];
				s2adj[a] = s2;
				s2adj[a][a] = s1[a];
				s1adjSolid[a] = !voxel::isAir(volume.voxel(s1adj[a]).getMaterial());
				s2adjSolid[a] = !voxel::isAir(volume.voxel(s2adj[a]).getMaterial());
			}

			// Enumerate the 6 length-3 face paths: (s1adj[i], s2adj[k]) for i != k.
			// Early out if any path is already fully solid.
			bool alreadyConnected = false;
			int bestFillCount = 3;
			int bestScore = -1;
			int bestI = -1;
			int bestK = -1;
			for (int i = 0; i < 3 && !alreadyConnected; ++i) {
				for (int k = 0; k < 3 && !alreadyConnected; ++k) {
					if (i == k) {
						continue;
					}
					const int fillCount = (s1adjSolid[i] ? 0 : 1) + (s2adjSolid[k] ? 0 : 1);
					if (fillCount == 0) {
						alreadyConnected = true;
						break;
					}
					// Interior score: sum of 26-neighbor counts on the air intermediates.
					int score = 0;
					if (!s1adjSolid[i]) {
						score += countSolid26Neighbors(volume, s1adj[i]);
					}
					if (!s2adjSolid[k]) {
						score += countSolid26Neighbors(volume, s2adj[k]);
					}
					if (fillCount < bestFillCount || (fillCount == bestFillCount && score > bestScore)) {
						bestFillCount = fillCount;
						bestScore = score;
						bestI = i;
						bestK = k;
					}
				}
			}

			if (alreadyConnected || bestI < 0) {
				continue;
			}

			// Fill the air intermediates on the chosen path.
			if (!s1adjSolid[bestI] && !sealed.has(s1adj[bestI])) {
				sealed.insert(s1adj[bestI]);
				toFill.push_back({s1adj[bestI], pickFillVoxel(volume, s1adj[bestI])});
			}
			if (!s2adjSolid[bestK] && !sealed.has(s2adj[bestK])) {
				sealed.insert(s2adj[bestK]);
				toFill.push_back({s2adj[bestK], pickFillVoxel(volume, s2adj[bestK])});
			}
		}
	}

	for (const HoleVoxel &hv : toFill) {
		volume.setVoxel(hv.worldPos, hv.voxel);
		outSealed.push_back(hv);
	}

	Log::debug("Sealed %d corner gap voxel(s)", (int)toFill.size());
	return (int)toFill.size();
}

struct AirCandidate {
	glm::ivec3 pos;
	int solidCount;
	int opposingPairs;
};

void detectHoles(const voxel::SparseVolume &volume, core::DynamicArray<HoleInfo> &outHoles,
				 int maxHoleSize, int minSolidNeighbors) {
	// use the global VisitedSet typedef from VolumeVisitor.h

	// Collect all solid positions
	core::DynamicArray<glm::ivec3> solidPositions;
	struct Collector {
		core::DynamicArray<glm::ivec3> &positions;
		void setVoxel(int x, int y, int z, const voxel::Voxel &) {
			positions.emplace_back(x, y, z);
		}
	};
	Collector collector{solidPositions};
	volume.copyTo(collector);

	Log::debug("Detecting holes: %d solid voxels to analyze", (int)solidPositions.size());

	// Score each near-surface air voxel
	// Use a map to deduplicate air positions and store their scores
	using CandidateMap = core::DynamicMap<glm::ivec3, AirCandidate, 1031, glm::hash<glm::ivec3>>;
	CandidateMap candidates;

	static constexpr glm::ivec3 opposingFaces[3][2] = {
		{{1, 0, 0}, {-1, 0, 0}},
		{{0, 1, 0}, {0, -1, 0}},
		{{0, 0, 1}, {0, 0, -1}}
	};

	for (const glm::ivec3 &pos : solidPositions) {
		for (const glm::ivec3 &offset : voxel::arrayPathfinderFaces) {
			const glm::ivec3 airPos = pos + offset;
			if (!voxel::isAir(volume.voxel(airPos).getMaterial())) {
				continue; // not air
			}

			// Check if we already scored this position
			auto iter = candidates.find(airPos);
			if (iter != candidates.end()) {
				continue; // already scored
			}

			// Score this air voxel
			int solidCount = 0;
			int opposingPairCount = 0;
			for (const glm::ivec3 &faceOffset : voxel::arrayPathfinderFaces) {
				if (!voxel::isAir(volume.voxel(airPos + faceOffset).getMaterial())) {
					++solidCount;
				}
			}
			for (int axis = 0; axis < 3; ++axis) {
				const bool posDir = !voxel::isAir(volume.voxel(airPos + opposingFaces[axis][0]).getMaterial());
				const bool negDir = !voxel::isAir(volume.voxel(airPos + opposingFaces[axis][1]).getMaterial());
				if (posDir && negDir) {
					++opposingPairCount;
				}
			}

			candidates.emplace(airPos, AirCandidate{airPos, solidCount, opposingPairCount});
		}
	}

	Log::debug("Found %d near-surface air voxels", (int)candidates.size());

	// BFS from high-score seeds to form connected components (holes)
	VisitedSet visited;
	int holeId = 0;

	for (auto iter = candidates.begin(); iter != candidates.end(); ++iter) {
		const AirCandidate &seed = iter->value;

		// Only start BFS from high-score seeds
		// Use solidCount threshold as primary filter. OpposingPairs boost confidence
		// but are not required (large holes in a wall face have no opposing pairs).
		if (seed.solidCount < minSolidNeighbors) {
			continue;
		}
		if (visited.has(seed.pos)) {
			continue;
		}

		// BFS to find connected hole component
		HoleInfo hole;
		hole.id = holeId++;
		glm::ivec3 centroidSum{0};

		core::DynamicArray<glm::ivec3> queue;
		queue.push_back(seed.pos);
		visited.insert(seed.pos);

		bool capped = false;
		int queueIdx = 0;
		while (queueIdx < (int)queue.size()) {
			if ((int)queue.size() > maxHoleSize) {
				capped = true;
				break;
			}

			const glm::ivec3 current = queue[queueIdx++];
			const voxel::Voxel fillVoxel = pickFillVoxel(volume, current);
			hole.voxels.push_back({current, fillVoxel});
			centroidSum += current;

			// Expand to 6-connected air neighbors that are also candidates
			for (const glm::ivec3 &offset : voxel::arrayPathfinderFaces) {
				const glm::ivec3 next = current + offset;
				if (visited.has(next)) {
					continue;
				}
				// Only expand to air voxels with at least 2 solid neighbors
				auto nextIter = candidates.find(next);
				if (nextIter == candidates.end()) {
					continue;
				}
				if (nextIter->value.solidCount < 2) {
					continue;
				}
				visited.insert(next);
				queue.push_back(next);
			}
		}

		if (hole.voxels.empty()) {
			continue;
		}

		const int count = hole.size();
		hole.centroid = centroidSum / count;

		if (capped) {
			Log::debug("Hole %d capped at %d voxels (max %d)", hole.id, count, maxHoleSize);
		}

		outHoles.push_back(core::move(hole));
	}

	// Sort by size ascending (small holes first for auto-fill)
	outHoles.sort([](const HoleInfo &a, const HoleInfo &b) { return a.size() < b.size(); });

	Log::debug("Detected %d holes", (int)outHoles.size());
}

} // namespace voxelutil
