# cg3d/pipeline.py
from __future__ import annotations
from typing import List, Tuple, Sequence

from .geom import Pt, unique_points
from .hull import ConvexHull3D
from .mesh import Delaunay3D

def _to_pts(raw: Sequence[Tuple[float, float, float]]) -> List[Pt]:
    return [Pt(x, y, z) for (x, y, z) in unique_points(raw)]

def tetrahedralize_convex(
    points: Sequence[Tuple[float, float, float]],
    backend: str = "internal",  # "internal" або "scipy"
):
    """
    Тетраедралізація опуклого політопу.
    Повертає (pts, surface_triangles, tetrahedra), де:
      - pts: список Pt у фінальному порядку (без дублікатів),
      - surface_triangles: List[Tuple[int,int,int]] (грані опуклої оболонки),
      - tetrahedra: List[Tuple[int,int,int,int]] (тетри, що заповнюють об’єм).

    backend:
      - "internal": наш власний ConvexHull3D + Delaunay3D (без залежностей)
      - "scipy": спробувати SciPy (ConvexHull, Delaunay), якщо встановлено
    """
    # --- підготовка точок (дедуп) ---
    pts = _to_pts(points)

    if backend.lower() == "scipy":
        try:
            import numpy as np
            from scipy.spatial import ConvexHull, Delaunay
            arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=float)

            # 1) опукла оболонка -> трикутні грані
            hull = ConvexHull(arr)                 # O(n log n), Qhull
            surface = [tuple(tri.tolist()) for tri in hull.simplices]

            # 2) 3D Делоне -> тетраедри
            tri = Delaunay(arr, qhull_options="QJ")  # QJ = joggle для робастності
            tets = [tuple(t.tolist()) for t in tri.simplices]
            return pts, surface, tets

        except Exception:
            # падаємо на внутрішній бекенд, якщо SciPy недоступний
            backend = "internal"

    # --- внутрішній бекенд (наші структури) ---
    # 1) опукла оболонка
    hull = ConvexHull3D(pts)
    surface = hull.faces()  # List[Tuple[int,int,int]]

    # 2) 3D Делоне (Bowyer–Watson) + вирізати супер-тетра
    d3 = Delaunay3D(pts)
    d3.build()
    d3.remove_super_tetra()

    tets = [t.v for t in d3.mesh.tets if t.alive]  # List[Tuple[int,int,int,int]]
    return pts, surface, tets
