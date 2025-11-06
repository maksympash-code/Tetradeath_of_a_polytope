from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Tuple

EPS = 1e-10  # обережний епс для перевірок

@dataclass(frozen=True)
class Pt:
    x: float
    y: float
    z: float
    def __iter__(self):
        yield self.x; yield self.y; yield self.z

def sub(a: Pt, b: Pt) -> Pt:
    return Pt(a.x - b.x, a.y - b.y, a.z - b.z)

def dot(a: Pt, b: Pt) -> float:
    return a.x*b.x + a.y*b.y + a.z*b.z

def cross(a: Pt, b: Pt) -> Pt:
    return Pt(a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x)

def norm(a: Pt) -> float:
    return sqrt(dot(a, a))

def centroid(points: Iterable[Pt]) -> Pt:
    xs = ys = zs = 0.0
    n = 0
    for p in points:
        xs += p.x; ys += p.y; zs += p.z; n += 1
    if n == 0:
        raise ValueError("empty set")
    inv = 1.0 / n
    return Pt(xs*inv, ys*inv, zs*inv)

def unique_points(points: Iterable[Tuple[float, float, float]], scale: float = 1e9) -> list[Pt]:
    """
    Груба дедуплікація з квантуванням (стабільніше для float).
    `scale=1e9` ≈ EPS=1e-9 на координату.
    """
    seen: dict[Tuple[int, int, int], Pt] = {}
    for x, y, z in points:
        key = (int(round(x*scale)), int(round(y*scale)), int(round(z*scale)))
        if key not in seen:
            seen[key] = Pt(x, y, z)
    return list(seen.values())
