"""
Pattern synthesis — distribute and synthesize point patterns on 3D surfaces.

Typical workflow
----------------
>>> import trimesh, pattern_synthesis as ps
>>> mesh = ps.Mesh(trimesh_mesh.vertices, trimesh_mesh.faces)
>>> param = mesh.parameterize()

# Simple well-distributed placement:
>>> points = param.distribute(n_points=200)
>>> points.xyz   # (200, 3) world-space positions

# Exemplar-based synthesis — match a reference pattern's statistics:
>>> result = param.synthesize(
...     exemplar_uv=my_exemplar_uv,          # (N, 2) reference points in UV space
...     input_boundary_uv=exemplar_region,   # (K, 2) polygon around exemplar
...     output_boundary_uv=target_region,    # (M, 2) polygon for output
... )
>>> result.to_3d()   # (n_out, 3) synthesized points in world space
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _pattern_synthesis as _ps

__all__ = [
    "Mesh",
    "ParameterizedMesh",
    "PointSet",
    "SynthesisResult",
]


class Mesh:
    """A triangulated 3D surface mesh."""

    def __init__(self, V: np.ndarray, F: np.ndarray):
        """
        Args:
            V: (n_vertices, 3) float64 vertex positions
            F: (n_faces, 3) int32 triangle face indices
        """
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int32)

    @property
    def n_vertices(self) -> int:
        return int(self.V.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.F.shape[0])

    def parameterize(self) -> "ParameterizedMesh":
        """Compute LSCM UV parameterization.

        The mesh must have at least one boundary loop (i.e. be an open mesh).

        Returns:
            ParameterizedMesh with UV coordinates and per-vertex area distortion.

        Raises:
            RuntimeError: if the mesh is closed or LSCM fails.
        """
        UV = _ps.parameterize(self.V, self.F)
        distortion = _ps.compute_distortion(self.V, UV, self.F)
        return ParameterizedMesh(self.V, self.F, UV, distortion)

    def __repr__(self) -> str:
        return f"Mesh(n_vertices={self.n_vertices}, n_faces={self.n_faces})"


class ParameterizedMesh:
    """A mesh with a computed UV parameterization.

    Can be constructed directly if you already have UV coordinates:

        param = ParameterizedMesh(V, F, UV)
    """

    def __init__(
        self,
        V: np.ndarray,
        F: np.ndarray,
        UV: np.ndarray,
        distortion: Optional[np.ndarray] = None,
    ):
        """
        Args:
            V:          (n_vertices, 3) vertex positions
            F:          (n_faces, 3) triangle indices
            UV:         (n_vertices, 2) UV coordinates
            distortion: (n_vertices,) per-vertex area distortion, computed if omitted
        """
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int32)
        self.UV = np.asarray(UV, dtype=np.float64)
        self.distortion = (
            np.asarray(distortion, dtype=np.float64)
            if distortion is not None
            else _ps.compute_distortion(self.V, self.UV, self.F)
        )

    @property
    def n_vertices(self) -> int:
        return int(self.V.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.F.shape[0])

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        seed: int = 42,
        uniform_uv: bool = True,
    ) -> "PointSet":
        """Sample n random points on the mesh surface.

        Args:
            n:          number of points
            seed:       random seed
            uniform_uv: weight by UV face area for uniform UV coverage

        Returns:
            PointSet with uv, xyz, tri_idx, and bary attributes.
        """
        uv_pts, tri_idx, bary = _ps.sample(self.UV, self.F, n, seed, uniform_uv)
        xyz = _ps.project_3d(self.V, self.F, tri_idx, bary)
        return PointSet(uv=uv_pts, xyz=xyz, tri_idx=tri_idx, bary=bary)

    def distribute(
        self,
        n_points: int = 5000,
        seed: int = 42,
        relax_iters: int = 50,
        uniform_uv: bool = True,
    ) -> "PointSet":
        """Sample points and apply Lloyd relaxation for a well-distributed set.

        Args:
            n_points:    target point count
            seed:        random seed
            relax_iters: Lloyd relaxation iterations (0 to skip)
            uniform_uv:  weight sampling by UV face area

        Returns:
            PointSet with relaxed UV positions and projected 3D positions.
        """
        d = _ps.distribute(self.V, self.F, n_points, seed, relax_iters, uniform_uv)
        return PointSet(uv=d["points_uv"], xyz=d["points_3d"])

    # ------------------------------------------------------------------
    # Relaxation
    # ------------------------------------------------------------------

    def relax(
        self,
        points_uv: np.ndarray,
        max_iters: int = 50,
        convergence: float = 1e-4,
        grid_size: int = 64,
    ) -> np.ndarray:
        """Lloyd relaxation on arbitrary UV points.

        Moves points toward their Voronoi centroids, weighted by surface area
        distortion, so they spread evenly across the 3D surface.

        Args:
            points_uv:   (n, 2) initial UV positions
            max_iters:   maximum iterations
            convergence: stop when max displacement falls below this
            grid_size:   acceleration grid resolution

        Returns:
            (n, 2) relaxed UV positions
        """
        return _ps.relax(
            np.asarray(points_uv, dtype=np.float64),
            self.UV,
            self.F,
            self.distortion,
            max_iters,
            convergence,
            grid_size,
        )

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        exemplar_uv: np.ndarray,
        input_boundary_uv: np.ndarray,
        output_boundary_uv: np.ndarray,
        n_points: int = -1,
        bin_count: int = 32,
        max_iters: int = 400,
        seed: int = 42,
    ) -> "SynthesisResult":
        """Exemplar-based point pattern synthesis via PCF matching.

        Places points inside ``output_boundary_uv`` whose pair-correlation
        function (measured via Delaunay hop distances in the UV domain) matches
        that of ``exemplar_uv`` points measured inside ``input_boundary_uv``.

        Args:
            exemplar_uv:         (N, 2) reference point positions in UV space
            input_boundary_uv:   (K, 2) polygon enclosing the exemplar region
            output_boundary_uv:  (M, 2) polygon enclosing the output region
            n_points:            target output count; -1 infers from exemplar density
            bin_count:           PCF histogram bin count (default 32)
            max_iters:           optimizer sweep iterations (default 400)
            seed:                random seed (default 42)

        Returns:
            SynthesisResult with output UV positions and PCF diagnostics.
        """
        d = _ps.synthesize(
            self.UV,
            self.F,
            np.asarray(exemplar_uv, dtype=np.float64),
            np.asarray(input_boundary_uv, dtype=np.float64),
            np.asarray(output_boundary_uv, dtype=np.float64),
            n_points,
            bin_count,
            max_iters,
            seed,
        )
        return SynthesisResult(
            output_uv=d["output_uv"],
            exemplar_pcf=d["exemplar_pcf"],
            output_pcf=d["output_pcf"],
            energy=float(d["energy"]),
            iterations=int(d["iterations"]),
            _mesh=self,
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project_uv_to_3d(self, points_uv: np.ndarray) -> np.ndarray:
        """Project UV positions to 3D world space.

        Locates the containing triangle for each point and interpolates
        barycentric coordinates. Points outside the mesh UV layout are zero.

        Args:
            points_uv: (n, 2) UV positions

        Returns:
            (n, 3) world-space positions
        """
        return _ps.project_uv_to_3d(
            self.V,
            self.F,
            self.UV,
            np.asarray(points_uv, dtype=np.float64),
        )

    def __repr__(self) -> str:
        return (
            f"ParameterizedMesh(n_vertices={self.n_vertices}, "
            f"n_faces={self.n_faces})"
        )


class PointSet:
    """A set of points on a mesh surface.

    Attributes:
        uv:      (n, 2) UV positions
        xyz:     (n, 3) 3D world-space positions, or None
        tri_idx: (n,) containing triangle index for each point, or None
        bary:    (n, 3) barycentric coordinates, or None
    """

    def __init__(
        self,
        uv: np.ndarray,
        xyz: Optional[np.ndarray] = None,
        tri_idx: Optional[np.ndarray] = None,
        bary: Optional[np.ndarray] = None,
    ):
        self.uv = np.asarray(uv, dtype=np.float64)
        self.xyz = np.asarray(xyz, dtype=np.float64) if xyz is not None else None
        self.tri_idx = (
            np.asarray(tri_idx, dtype=np.int32) if tri_idx is not None else None
        )
        self.bary = (
            np.asarray(bary, dtype=np.float64) if bary is not None else None
        )

    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.uv.shape[0])

    def to_3d(self, mesh: ParameterizedMesh) -> np.ndarray:
        """Project UV positions to 3D using the given mesh.

        Args:
            mesh: ParameterizedMesh providing V, F, UV for projection

        Returns:
            (n, 3) world-space positions
        """
        return mesh.project_uv_to_3d(self.uv)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"PointSet(n={self.n}, has_3d={self.xyz is not None})"


class SynthesisResult:
    """Result of exemplar-based PCF pattern synthesis.

    Attributes:
        output_uv:    (n, 2) synthesized point positions in UV space
        exemplar_pcf: (bin_count,) normalized exemplar pair-correlation histogram
        output_pcf:   (bin_count,) normalized output pair-correlation histogram
        energy:       final L2 distance between output and exemplar PCF (lower is better)
        iterations:   number of optimizer sweeps completed
    """

    def __init__(
        self,
        output_uv: np.ndarray,
        exemplar_pcf: np.ndarray,
        output_pcf: np.ndarray,
        energy: float,
        iterations: int,
        _mesh: Optional[ParameterizedMesh] = None,
    ):
        self.output_uv = np.asarray(output_uv, dtype=np.float64)
        self.exemplar_pcf = np.asarray(exemplar_pcf, dtype=np.float64)
        self.output_pcf = np.asarray(output_pcf, dtype=np.float64)
        self.energy = energy
        self.iterations = iterations
        self._mesh = _mesh

    @property
    def n(self) -> int:
        """Number of synthesized points."""
        return int(self.output_uv.shape[0])

    def to_3d(self, mesh: Optional[ParameterizedMesh] = None) -> np.ndarray:
        """Project synthesized points to 3D world space.

        Uses the mesh from ``synthesize()`` if none is provided.

        Args:
            mesh: ParameterizedMesh to use for projection; uses the originating
                  mesh if omitted.

        Returns:
            (n, 3) world-space positions
        """
        m = mesh if mesh is not None else self._mesh
        if m is None:
            raise RuntimeError(
                "No mesh available — pass a ParameterizedMesh to to_3d()"
            )
        return m.project_uv_to_3d(self.output_uv)

    def as_point_set(self, mesh: Optional[ParameterizedMesh] = None) -> PointSet:
        """Convert to a PointSet, projecting to 3D if a mesh is available.

        Args:
            mesh: optional mesh override for 3D projection

        Returns:
            PointSet with uv set and xyz set if a mesh is available.
        """
        m = mesh if mesh is not None else self._mesh
        xyz = m.project_uv_to_3d(self.output_uv) if m is not None else None
        return PointSet(uv=self.output_uv, xyz=xyz)

    def __repr__(self) -> str:
        return (
            f"SynthesisResult(n={self.n}, energy={self.energy:.4f}, "
            f"iterations={self.iterations})"
        )
