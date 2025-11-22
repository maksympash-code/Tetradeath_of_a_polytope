from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple

from .geom import Pt, unique_points
from .hull import ConvexHull3D


def tetrahedralize_convex(
    points: Iterable[Tuple[float, float, float]],
    backend: str = "scipy",
) -> Tuple[List[Pt], List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Повний пайплайн:
      - прибирає дублікати точок;
      - будує опуклу оболонку (наш ConvexHull3D) -> surface_triangles;
      - будує 3D Делоне-тетраедралізацію через SciPy Delaunay -> tetrahedra.

    Повертає:
      pts       — список Pt у фінальному порядку;
      surface   — список трикутників оболонки (індекси у pts);
      tets      — список тетраедрів (індекси у pts).
    """
    pts: List[Pt] = unique_points(points)  # твоя існуюча функція, що вертає List[Pt]

    # 1) Опукла оболонка нашою реалізацією
    hull = ConvexHull3D(pts)
    surface = hull.faces()  # List[Tuple[int,int,int]]

    if backend.lower() == "scipy":
        try:
            import numpy as np
            from scipy.spatial import Delaunay
        except ImportError as e:
            raise RuntimeError(
                "backend='scipy', але SciPy не встановлено. "
                "Встанови scipy або використай інший backend."
            ) from e

        arr = np.array([(p.x, p.y, p.z) for p in pts], dtype=float)

        # 2) 3D Delaunay (Qhull під капотом)
        dela = Delaunay(arr, qhull_options="QJ")  # QJ = joggle для робастності
        tets = [tuple(int(i) for i in simplex) for simplex in dela.simplices]

        return pts, surface, tets

    else:
        raise ValueError(f"Невідомий backend: {backend}")
