"""
cg3d — мінімальна бібліотека для 3D комп'ютерної геометрії (Py 3.13).
Зараз: рандомізований інкрементальний 3D convex hull + conflict graph.
"""

__version__ = "0.1.0"

from cg3d.geom import Pt, EPS, centroid, unique_points
from cg3d.predicates import orient3d, signed_distance_to_plane, visible_from_point
from cg3d.hull import ConvexHull3D

__all__ = [
    "Pt", "EPS", "centroid", "unique_points",
    "orient3d", "signed_distance_to_plane", "visible_from_point",
    "ConvexHull3D", "__version__",
]
