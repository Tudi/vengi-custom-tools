/**
 * @file
 */

#pragma once

namespace voxedit {
class SceneManager;

namespace printing {

// minCellSize: finest probe box size for multi-resolution refinement.
// 0 = no refinement (coarse only, inherited from regrid cell size).
// Must be a power-of-two divisor of the regrid cell size (e.g. 32 when cellSize=128).
void runFaceClassify(SceneManager *sceneMgr, int minCellSize = 0);

// Diagnostic: run coarse exterior/interior BFS, then a blocking fine BFS at cellSize.
// Cells where exterior tries to enter the coarse-interior region are quarantined (not
// propagated further). Adjacent solid voxels are recolored bright red so holes are
// immediately visible in the scene. Does not modify classification colors.
void runHoleMap(SceneManager *sceneMgr, int minCellSize = 0);

// Fill holes: same bidirectional BFS as runHoleMap, but places solid voxels (green)
// into the gap positions instead of coloring adjacent walls.
// Positions outside all node regions are skipped and logged.
void runHoleFill(SceneManager *sceneMgr, int minCellSize = 0);

// Debug: paint the BFS frontier shell at a given cellSize. Single-level BFS:
// orange = exterior frontier, blue = interior frontier. cellSize controls the
// resolution of the wrap; pass 0 for "auto = modal regridded width" (typically 128).
void runDebugFrontier(SceneManager *sceneMgr, int cellSize = 0);

} // namespace printing
} // namespace voxedit
