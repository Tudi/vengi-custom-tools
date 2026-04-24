/**
 * @file
 */

#include "app/tests/AbstractTest.h"
#include "voxel/SparseVolume.h"
#include "voxel/Voxel.h"
#include "voxelutil/FillHoles.h"

namespace voxelutil {

class FillHolesTest : public app::AbstractTest {};

static voxel::Voxel solid(uint8_t color = 1) {
	return voxel::createVoxel(voxel::VoxelType::Generic, color);
}

// Helper: fill a hollow cube shell (1 voxel thick walls) into a SparseVolume
static void fillShell(voxel::SparseVolume &vol, const glm::ivec3 &mins, const glm::ivec3 &maxs, uint8_t color = 1) {
	for (int z = mins.z; z <= maxs.z; ++z) {
		for (int y = mins.y; y <= maxs.y; ++y) {
			for (int x = mins.x; x <= maxs.x; ++x) {
				if (x == mins.x || x == maxs.x || y == mins.y || y == maxs.y || z == mins.z || z == maxs.z) {
					vol.setVoxel(x, y, z, solid(color));
				}
			}
		}
	}
}

// Helper: fill a solid cube into a SparseVolume
static void fillSolid(voxel::SparseVolume &vol, const glm::ivec3 &mins, const glm::ivec3 &maxs, uint8_t color = 1) {
	for (int z = mins.z; z <= maxs.z; ++z) {
		for (int y = mins.y; y <= maxs.y; ++y) {
			for (int x = mins.x; x <= maxs.x; ++x) {
				vol.setVoxel(x, y, z, solid(color));
			}
		}
	}
}

TEST_F(FillHolesTest, testSingleVoxelHole) {
	voxel::SparseVolume vol;
	// 5x5x5 hollow shell
	fillShell(vol, {0, 0, 0}, {4, 4, 4});
	// Poke a 1-voxel hole in one wall
	vol.setVoxel(2, 2, 0, voxel::Voxel());

	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes);
	ASSERT_GE((int)holes.size(), 1);

	// The hole at (2,2,0) should be found - it's air with 4 solid face-neighbors and at least 1 opposing pair
	bool foundHole = false;
	for (const HoleInfo &hole : holes) {
		for (const HoleVoxel &hv : hole.voxels) {
			if (hv.worldPos == glm::ivec3(2, 2, 0)) {
				foundHole = true;
			}
		}
	}
	EXPECT_TRUE(foundHole);
}

TEST_F(FillHolesTest, testNoHoles) {
	voxel::SparseVolume vol;
	// Solid 3x3x3 cube - no air inside, no holes
	fillSolid(vol, {0, 0, 0}, {2, 2, 2});

	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes);
	// There might be near-surface air that scores high enough, but for a solid cube,
	// no air voxel should have opposing solid pairs on all sides
	// Actually exterior air around a solid cube might have some neighbors,
	// but opposing pairs require solid on BOTH sides of an axis - unlikely for exterior
	EXPECT_EQ(0, (int)holes.size());
}

TEST_F(FillHolesTest, testMultipleSmallHoles) {
	voxel::SparseVolume vol;
	// 7x7x7 hollow shell
	fillShell(vol, {0, 0, 0}, {6, 6, 6});

	// Poke 3 separate 1-voxel holes in different walls
	vol.setVoxel(3, 3, 0, voxel::Voxel()); // front wall
	vol.setVoxel(3, 3, 6, voxel::Voxel()); // back wall
	vol.setVoxel(0, 3, 3, voxel::Voxel()); // left wall

	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes);
	EXPECT_GE((int)holes.size(), 3);
}

TEST_F(FillHolesTest, testLargeHole) {
	voxel::SparseVolume vol;
	// 9x9x9 hollow shell
	fillShell(vol, {0, 0, 0}, {8, 8, 8});

	// Poke a 3x3 hole in one wall (remove a 3x3 section from z=0 face)
	for (int y = 3; y <= 5; ++y) {
		for (int x = 3; x <= 5; ++x) {
			vol.setVoxel(x, y, 0, voxel::Voxel());
		}
	}

	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes, 1000, 3);
	ASSERT_GE((int)holes.size(), 1);
	// The hole should contain multiple voxels
	bool foundLargeHole = false;
	for (const HoleInfo &hole : holes) {
		if (hole.size() >= 3) {
			foundLargeHole = true;
		}
	}
	EXPECT_TRUE(foundLargeHole);
}

TEST_F(FillHolesTest, testHoleFilling) {
	voxel::SparseVolume vol;
	// 5x5x5 hollow shell with 1-voxel hole
	fillShell(vol, {0, 0, 0}, {4, 4, 4});
	vol.setVoxel(2, 2, 0, voxel::Voxel());

	// Verify the hole exists
	EXPECT_TRUE(voxel::isAir(vol.voxel(2, 2, 0).getMaterial()));

	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes);
	ASSERT_GE((int)holes.size(), 1);

	// Apply the fill
	for (const HoleInfo &hole : holes) {
		for (const HoleVoxel &hv : hole.voxels) {
			vol.setVoxel(hv.worldPos, hv.voxel);
		}
	}

	// Verify the hole is filled
	EXPECT_FALSE(voxel::isAir(vol.voxel(2, 2, 0).getMaterial()));
}

TEST_F(FillHolesTest, testThinShellGap) {
	voxel::SparseVolume vol;
	// Create a thin flat wall (1 voxel thick, 5x5) at z=0
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			vol.setVoxel(x, y, 0, solid());
		}
	}
	// Poke a hole in the middle
	vol.setVoxel(2, 2, 0, voxel::Voxel());

	core::DynamicArray<HoleInfo> holes;
	// For a flat wall, the hole voxel has 4 solid face-neighbors (up/down/left/right) and 2 opposing pairs
	detectHoles(vol, holes, 1000, 3);
	ASSERT_GE((int)holes.size(), 1);
	bool foundGap = false;
	for (const HoleInfo &hole : holes) {
		for (const HoleVoxel &hv : hole.voxels) {
			if (hv.worldPos == glm::ivec3(2, 2, 0)) {
				foundGap = true;
			}
		}
	}
	EXPECT_TRUE(foundGap);
}

TEST_F(FillHolesTest, testDiagonalGapSealing) {
	voxel::SparseVolume vol;
	// Create a staircase slope: diagonal connection only
	// S at (0,0,0), S at (1,0,1) - connected by edge, not face
	vol.setVoxel(0, 0, 0, solid());
	vol.setVoxel(1, 0, 1, solid());

	// Verify elbows are air
	EXPECT_TRUE(voxel::isAir(vol.voxel(1, 0, 0).getMaterial()));
	EXPECT_TRUE(voxel::isAir(vol.voxel(0, 0, 1).getMaterial()));

	core::DynamicArray<HoleVoxel> sealed;
	const int count = sealDiagonalGaps(vol, sealed);
	EXPECT_EQ(1, count);

	// One of the elbows should now be filled
	const bool e1Filled = !voxel::isAir(vol.voxel(1, 0, 0).getMaterial());
	const bool e2Filled = !voxel::isAir(vol.voxel(0, 0, 1).getMaterial());
	EXPECT_TRUE(e1Filled || e2Filled);
	// But not both
	EXPECT_FALSE(e1Filled && e2Filled);
}

TEST_F(FillHolesTest, testDiagonalSlopeNoFalsePositive) {
	voxel::SparseVolume vol;
	// Create a staircase slope (without sealing)
	vol.setVoxel(0, 0, 0, solid());
	vol.setVoxel(1, 0, 1, solid());
	vol.setVoxel(2, 0, 2, solid());
	vol.setVoxel(3, 0, 3, solid());

	// Detect holes WITHOUT sealing first
	// The air between diagonal steps should NOT be flagged as holes
	// because each air voxel has at most 2 solid 6-neighbors (no opposing pairs typical)
	core::DynamicArray<HoleInfo> holes;
	detectHoles(vol, holes, 1000, 3);

	// No holes should be detected in a simple staircase
	EXPECT_EQ(0, (int)holes.size());
}

TEST_F(FillHolesTest, testDiagonalSealingInteriorPreference) {
	voxel::SparseVolume vol;
	// S1=(0,0,0) and S2=(1,0,1) are edge-connected (differ in X and Z by 1).
	// Elbows: (1,0,0) and (0,0,1)
	// Add extra solid at (2,0,1) — face-connected to S2 (not edge-connected to S1).
	// This biases elbow (1,0,0) which has (2,0,1) as a 26-neighbor.
	//
	// Elbow (1,0,0) 26-neighbors: S1=(0,0,0), S2=(1,0,1), extra=(2,0,1) → 3 solid
	// Elbow (0,0,1) 26-neighbors: S1=(0,0,0), S2=(1,0,1) → 2 solid
	// So (1,0,0) should be chosen (interior side).
	vol.setVoxel(0, 0, 0, solid());
	vol.setVoxel(1, 0, 1, solid());
	vol.setVoxel(2, 0, 1, solid()); // face-connected to S2, biases elbow (1,0,0)

	core::DynamicArray<HoleVoxel> sealed;
	sealDiagonalGaps(vol, sealed);

	// Only pair is S1-S2. Extra is face-connected to S2, no edge pairs with S1.
	ASSERT_EQ(1, (int)sealed.size());
	EXPECT_EQ(glm::ivec3(1, 0, 0), sealed[0].worldPos);
}

} // namespace voxelutil
