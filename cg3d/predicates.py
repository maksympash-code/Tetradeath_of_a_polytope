from __future__ import annotations
from cg3d.geom import Pt, sub, cross, dot, norm, EPS

def orient3d(a: Pt, b: Pt, c: Pt, d: Pt) -> float:
    """
    Орієнтований об'єм *6 тетра (a,b,c,d).
    >0 якщо d по той бік площини abc, куди вказує нормаль (a->b)x(a->c).
    """
    ab = sub(b, a)
    ac = sub(c, a)
    ad = sub(d, a)
    vol6 = dot(cross(ab, ac), ad)
    return vol6

def signed_distance_to_plane(a: Pt, b: Pt, c: Pt, p: Pt) -> float:
    """
    Підписана відстань від p до площини (a,b,c). Знак = sign(orient3d(a,b,c,p)).
    """
    n = cross(sub(b, a), sub(c, a))
    area2 = norm(n)
    if area2 == 0.0:
        return 0.0
    return orient3d(a, b, c, p) / area2

def visible_from_point(a: Pt, b: Pt, c: Pt, p: Pt, eps: float = EPS) -> bool:
    """Чи бачить точка p грань (a,b,c)?"""
    return orient3d(a, b, c, p) > eps
