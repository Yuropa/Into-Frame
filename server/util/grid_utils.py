"""
General-purpose 2-D grid utilities.
"""

import numpy as np


def confidence_flood_fill(
    grid: np.ndarray,
    has_data: np.ndarray,
    confidence: np.ndarray,
    confidence_decay: float = 0.95,
) -> np.ndarray:
    """
    Confidence-weighted BFS flood fill for a 2-D grid of discrete values.

    Cells with high confidence propagate their value first, claiming adjacent
    empty territory before lower-confidence sources reach it.  Each hop
    multiplies confidence by confidence_decay, so proximity to a
    high-confidence source beats proximity to a distant low-confidence one.

    Uses 8-connectivity with a decay exponent scaled by physical hop
    distance (diagonal hops cover sqrt(2) the distance of orthogonal ones).
    Plain 4-connectivity measures reachability in Manhattan distance, which
    grows a diamond/star from any seed regardless of the actual terrain --
    purely a grid-connectivity artifact; the scaled 8-connectivity front
    approximates a circular ball instead.

    Parameters
    ----------
    grid           : (H, W) array of discrete values; filled cells hold their
                     assigned value, unfilled cells can hold any placeholder.
    has_data       : (H, W) bool — True where grid already has a valid value.
    confidence     : (H, W) float32 — certainty of each filled cell [0, 1].
                     Unfilled cells are ignored.
    confidence_decay : multiplier applied to confidence at each propagation hop.

    Returns
    -------
    (H, W) array of the same dtype as grid with all reachable empty cells filled.
    """
    if not np.any(~has_data) or not np.any(has_data):
        return grid

    import heapq
    import math

    gh, gw = grid.shape
    result = grid.copy()
    filled = has_data.copy()

    orth_decay = confidence_decay
    diag_decay = confidence_decay ** math.sqrt(2)
    neighbors = (
        (-1, 0, orth_decay), (1, 0, orth_decay), (0, -1, orth_decay), (0, 1, orth_decay),
        (-1, -1, diag_decay), (-1, 1, diag_decay), (1, -1, diag_decay), (1, 1, diag_decay),
    )

    heap: list = []
    rows, cols = np.where(has_data)
    for r, c in zip(rows.tolist(), cols.tolist()):
        conf = max(float(confidence[r, c]), 1e-6)
        heapq.heappush(heap, (-conf, r, c))

    while heap:
        neg_conf, r, c = heapq.heappop(heap)
        conf = -neg_conf
        value = result[r, c]
        for dr, dc, decay in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < gh and 0 <= nc < gw and not filled[nr, nc]:
                filled[nr, nc] = True
                result[nr, nc] = value
                heapq.heappush(heap, (-(conf * decay), nr, nc))

    return result
