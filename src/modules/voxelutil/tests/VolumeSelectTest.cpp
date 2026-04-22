/**
 * @file
 */

#include "voxelutil/VolumeSelect.h"
#include "app/tests/AbstractTest.h"
#include "voxel/Face.h"
#include "voxel/RawVolume.h"
#include "voxel/Voxel.h"

namespace voxelutil {

class VolumeSelectTest : public app::AbstractTest {};

static void fillBox(voxel::RawVolume &volume, const voxel::Region &r, const voxel::Voxel &v) {
	const glm::ivec3 &lo = r.getLowerCorner();
	const glm::ivec3 &hi = r.getUpperCorner();
	for (int z = lo.z; z <= hi.z; ++z) {
		for (int y = lo.y; y <= hi.y; ++y) {
			for (int x = lo.x; x <= hi.x; ++x) {
				volume.setVoxel(x, y, z, v);
			}
		}
	}
}

// Regression: lasso drawn on an upper structure must not sweep in a lower structure that
// merely shares the (u, v) silhouette. Before flood-fill, selectionFinalizeLasso filtered
// every surface voxel in the volume by 2D point-in-polygon only, so a tower behind a
// lassoed rooftop got picked up too.
TEST_F(VolumeSelectTest, testLassoFloodFillDoesNotCrossDisjointStructures) {
	voxel::Region region(0, 29);
	voxel::RawVolume volume(region);
	const voxel::Voxel solid = voxel::createVoxel(voxel::VoxelType::Generic, 1);

	// Upper slab at y=20, lower slab at y=5. Same X/Z footprint, disjoint in Y. With a
	// PositiveY lasso drawing U=Z, V=X, both slabs project to the same (u, v) silhouette.
	fillBox(volume, voxel::Region(glm::ivec3(5, 20, 5), glm::ivec3(15, 20, 15)), solid);
	fillBox(volume, voxel::Region(glm::ivec3(5, 5, 5), glm::ivec3(15, 5, 15)), solid);

	// Polygon vertices live on the upper slab's top surface so edge rasterization seeds
	// only the upper slab. Path traces a square covering the whole upper slab footprint.
	core::DynamicArray<glm::ivec3> path;
	path.push_back(glm::ivec3(5, 20, 5));
	path.push_back(glm::ivec3(15, 20, 5));
	path.push_back(glm::ivec3(15, 20, 15));
	path.push_back(glm::ivec3(5, 20, 15));

	int selectedUpper = 0;
	int selectedLower = 0;
	int selectedElsewhere = 0;
	auto markFunc = [&](int x, int y, int z, const voxel::Voxel &) {
		if (y == 20) {
			++selectedUpper;
		} else if (y == 5) {
			++selectedLower;
		} else {
			++selectedElsewhere;
		}
	};

	// PositiveY face: U=Z (axis 2), V=X (axis 0), W=Y (axis 1), positiveNormal=true
	lassoFloodFillSurface(volume, path, /*uAxis*/ 2, /*vAxis*/ 0, /*wAxis*/ 1,
						  /*positiveNormal*/ true, volume.region(), markFunc);

	EXPECT_GT(selectedUpper, 0) << "upper slab should be selected";
	EXPECT_EQ(selectedLower, 0) << "lower slab must not be swept in despite sharing (X,Z)";
	EXPECT_EQ(selectedElsewhere, 0);
}

// A lasso drawn around the visible silhouette of a column must select all sides (front,
// left, right, back) via 26-connected surface walk. This is why a face-orientation filter
// is deliberately NOT applied: a column's side voxels face +X/-X/+Z/-Z, and the user
// expects all of them selected regardless of the face the polygon is anchored to.
TEST_F(VolumeSelectTest, testLassoFloodFillWrapsAroundColumn) {
	voxel::Region region(0, 19);
	voxel::RawVolume volume(region);
	const voxel::Voxel solid = voxel::createVoxel(voxel::VoxelType::Generic, 1);

	// Vertical column at (8..11, 0..15, 8..11) - hollow isn't necessary, all surface
	// voxels of the outer shell should be reached.
	fillBox(volume, voxel::Region(glm::ivec3(8, 0, 8), glm::ivec3(11, 15, 11)), solid);

	// Lasso drawn on the +X face: U=Y (axis 1), V=Z (axis 2), W=X (axis 0).
	// Polygon covers the full height and depth of the column silhouette on that face.
	core::DynamicArray<glm::ivec3> path;
	path.push_back(glm::ivec3(11, 2, 8));
	path.push_back(glm::ivec3(11, 13, 8));
	path.push_back(glm::ivec3(11, 13, 11));
	path.push_back(glm::ivec3(11, 2, 11));

	core::DynamicSet<glm::ivec3, 257, glm::hash<glm::ivec3>> selected;
	auto markFunc = [&](int x, int y, int z, const voxel::Voxel &) {
		selected.insert(glm::ivec3(x, y, z));
	};

	// PositiveX face: U=Y (1), V=Z (2), W=X (0), positiveNormal=true
	lassoFloodFillSurface(volume, path, /*uAxis*/ 1, /*vAxis*/ 2, /*wAxis*/ 0,
						  /*positiveNormal*/ true, volume.region(), markFunc);

	// Front (+X face at x=11), back (-X at x=8), and side walls all project inside the
	// polygon (same U=Y, V=Z as the silhouette). The back face must be reached via 26-
	// connected wrap through the side walls.
	bool frontHit = false;
	bool backHit = false;
	bool leftHit = false;
	bool rightHit = false;
	for (auto *entry : selected) {
		if (entry->key.x == 11) {
			frontHit = true;
		} else if (entry->key.x == 8) {
			backHit = true;
		}
		if (entry->key.z == 8) {
			leftHit = true;
		} else if (entry->key.z == 11) {
			rightHit = true;
		}
	}
	EXPECT_TRUE(frontHit) << "front face should be selected (directly in polygon)";
	EXPECT_TRUE(backHit) << "back face should be reached via 26-connected surface walk";
	EXPECT_TRUE(leftHit);
	EXPECT_TRUE(rightHit);
}

// Concave polygon: if the first seed alone couldn't reach every interior corner via flood,
// seeding from every edge pixel protects us from the trap. Drawing an L-shape polygon on a
// flat surface must still cover the whole inside of the L.
TEST_F(VolumeSelectTest, testLassoFloodFillConcavePolygon) {
	voxel::Region region(0, 19);
	voxel::RawVolume volume(region);
	const voxel::Voxel solid = voxel::createVoxel(voxel::VoxelType::Generic, 1);

	// 12x12 flat top at y=5
	fillBox(volume, voxel::Region(glm::ivec3(2, 5, 2), glm::ivec3(13, 5, 13)), solid);

	// L-shape polygon in (Z=U, X=V):
	//   Z in [2..13], X in [2..7] (wide stroke)
	//   Z in [8..13], X in [7..13] (narrow stroke) - forms an L
	core::DynamicArray<glm::ivec3> path;
	path.push_back(glm::ivec3(2, 5, 2));
	path.push_back(glm::ivec3(7, 5, 2));
	path.push_back(glm::ivec3(7, 5, 8));
	path.push_back(glm::ivec3(13, 5, 8));
	path.push_back(glm::ivec3(13, 5, 13));
	path.push_back(glm::ivec3(2, 5, 13));

	int selectedCount = 0;
	bool insideLCorner = false;	 // voxel at (3, 5, 12) - deep inside wide stroke
	bool insideLNarrow = false;	 // voxel at (10, 5, 10) - inside narrow stroke
	bool outsideCorner = false;	 // voxel at (12, 5, 3) - outside the L's notch
	auto markFunc = [&](int x, int y, int z, const voxel::Voxel &) {
		++selectedCount;
		if (x == 3 && y == 5 && z == 12) {
			insideLCorner = true;
		}
		if (x == 10 && y == 5 && z == 10) {
			insideLNarrow = true;
		}
		if (x == 12 && y == 5 && z == 3) {
			outsideCorner = true;
		}
	};

	lassoFloodFillSurface(volume, path, /*uAxis*/ 2, /*vAxis*/ 0, /*wAxis*/ 1,
						  /*positiveNormal*/ true, volume.region(), markFunc);

	EXPECT_GT(selectedCount, 0);
	EXPECT_TRUE(insideLCorner) << "flood should reach deep inside the wide stroke";
	EXPECT_TRUE(insideLNarrow) << "flood should reach the narrow stroke past the bend";
	EXPECT_FALSE(outsideCorner) << "voxels outside the L notch must not be selected";
}

} // namespace voxelutil
