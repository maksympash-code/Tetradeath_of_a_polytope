# cg3d/predicates.py
from __future__ import annotations
from math import fabs
from typing import List
from .geom import Pt, sub, cross, dot, norm, EPS

def orient3d(a: Pt, b: Pt, c: Pt, d: Pt) -> float:
    ab = sub(b, a)
    ac = sub(c, a)
    ad = sub(d, a)
    return dot(cross(ab, ac), ad)

def signed_distance_to_plane(a: Pt, b: Pt, c: Pt, p: Pt) -> float:
    n = cross(sub(b, a), sub(c, a))
    area2 = norm(n)
    if area2 == 0.0:
        return 0.0
    return orient3d(a, b, c, p) / area2

def visible_from_point(a: Pt, b: Pt, c: Pt, p: Pt, eps: float = EPS) -> bool:
    return orient3d(a, b, c, p) > eps

# ---------- інструмент для детермінанта ----------
def _det(m: List[List[float]]) -> float:
    """Детермінант через Гауса з частковим вибором опорного елемента (float)."""
    n = len(m)
    a = [row[:] for row in m]
    det = 1.0
    for i in range(n):
        # півод
        piv = i
        maxv = abs(a[i][i])
        for r in range(i+1, n):
            v = abs(a[r][i])
            if v > maxv:
                maxv = v; piv = r
        if maxv == 0.0:
            return 0.0
        if piv != i:
            a[i], a[piv] = a[piv], a[i]
            det = -det
        det *= a[i][i]
        inv = 1.0 / a[i][i]
        # елімінація
        for r in range(i+1, n):
            factor = a[r][i] * inv
            if factor != 0.0:
                for c in range(i, n):
                    a[r][c] -= factor * a[i][c]
    return det

def insphere(a: Pt, b: Pt, c: Pt, d: Pt, e: Pt) -> float:
    """
    Знак тесту «чи всередині сфери, що проходить через a,b,c,d, лежить e?».
    Повертає:
      >0  якщо e всередині circumsphere(a,b,c,d),
      <0  якщо зовні,
       0  якщо на сфері (з точністю до похибки).
    Зв'язаний з орієнтацією (a,b, c, d): знак множимо на sign(orient3d(a,b,c,d)).
    """
    def row(p: Pt) -> list[float]:
        s = p.x*p.x + p.y*p.y + p.z*p.z
        return [p.x, p.y, p.z, s, 1.0]

    M = [row(a), row(b), row(c), row(d), row(e)]
    val = _det(M)
    ori = orient3d(a, b, c, d)
    if ori > 0:
        return val
    elif ori < 0:
        return -val
    else:
        # виродження: a,b,c,d копланарні — для CDT таку ситуацію уникаємо/джиттеримо
        return 0.0
