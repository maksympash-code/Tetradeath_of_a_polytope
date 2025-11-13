from __future__ import annotations
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, List, Optional, Set, Tuple

from .geom import Pt, centroid, EPS, sub, cross, norm
from .predicates import orient3d, signed_distance_to_plane, visible_from_point

Edge = Tuple[int, int]          # орієнтоване ребро (u, v)
UEdge = Tuple[int, int]         # неорієнтоване ребро (min(u,v), max(u,v))


@dataclass
class Face:
    """
    Трикутна грань опуклої оболонки.
    v: індекси вершин із узгодженою орієнтацією (нормаль назовні).
    nbr[i]: сусідня грань через локальне ребро i (0:(a,b), 1:(b,c), 2:(c,a)), або None.
    alive: чи грань активна (у hull).
    conflict: множина індексів точок, що «бачать» цю грань (Quickhull conflict set).
    """
    v: Tuple[int, int, int]
    nbr: List[Optional[int]] = field(default_factory=lambda: [None, None, None])
    alive: bool = True
    conflict: Set[int] = field(default_factory=set)

    def edge(self, i: int) -> Edge:
        a, b, c = self.v
        if i == 0:
            return (a, b)
        if i == 1:
            return (b, c)
        return (c, a)

    def tri(self) -> Tuple[int, int, int]:
        return self.v


class ConvexHull3D:
    """
    Рандомізований інкрементальний 3D convex hull із conflict graph.

    Вхід: список Pt (мінімум 4, не всі копланарні).
    Вихід: self.faces_list — масив Face; публічний метод faces() повертає лише активні трикутники.
    """

    def __init__(self, points: List[Pt], eps: float = EPS):
        if len(points) < 4:
            raise ValueError("Need at least 4 points")
        self.P: List[Pt] = points[:]  # індексована копія
        self.eps = eps

        # Динамічні структури
        self.faces_list: List[Face] = []                             # усі створені грані (деякі з них можуть бути dead)
        self.edge2face: Dict[UEdge, List[Tuple[int, int]]] = {}      # (min(u,v),max(u,v)) -> [(face_id, local_edge), ...]
        self.point2faces: Dict[int, Set[int]] = {}                   # p -> видимі грані (conflict adjacency)

        # 1) стартовий тетраедр
        base_faces = self._build_initial_tetra()
        self._init_conflicts(base_faces)

        # 2) основний цикл: поки існують конфлікти (зовнішні точки)
        self._expand_until_done()

        # ВАЖЛИВО: не ущільнюємо faces_list (alive/dead) — щоб не ламати індекси в nbr

    # ---------------- Публічний API ----------------
    def faces(self) -> List[Tuple[int, int, int]]:
        """Активні грані (трикутники) як індекси вершин."""
        return [f.v for f in self.faces_list if f.alive]

    # ---------------- Внутрішні методи ----------------
    def _add_face(self, v0: int, v1: int, v2: int) -> int:
        """Створити грань і зареєструвати її ребра в edge2face."""
        fid = len(self.faces_list)
        face = Face((v0, v1, v2))
        self.faces_list.append(face)
        for ei in range(3):
            u, v = face.edge(ei)
            key = (min(u, v), max(u, v))
            self.edge2face.setdefault(key, []).append((fid, ei))
        return fid

    def _set_neighbor(self, fid_a: int, edge_a: int, fid_b: Optional[int]) -> None:
        """Прописати сусіда через локальне ребро edge_a у грані fid_a."""
        self.faces_list[fid_a].nbr[edge_a] = fid_b

    def _rebuild_all_adjacencies(self) -> None:
        """Переприв'язати nbr для всіх граней за edge2face."""
        for face in self.faces_list:
            face.nbr = [None, None, None]
        for key, lst in self.edge2face.items():
            if len(lst) == 2:
                (fa, ea), (fb, eb) = lst[0], lst[1]
                self._set_neighbor(fa, ea, fb)
                self._set_neighbor(fb, eb, fa)

    def _build_initial_tetra(self) -> List[int]:
        """
        Знайти перші 4 не копланарні точки:
          - p0,p1,p2: неколінеарні (|| (b-a) x (c-a) || > eps)
          - p3: не копланарна базовій площині (|orient3d| > eps)
        Створити 4 грані тетра з правильною (зовнішньою) орієнтацією.
        """
        idx = list(range(len(self.P)))
        shuffle(idx)

        # 1) базовий трикутник
        p0 = p1 = p2 = None
        for i in range(len(idx) - 2):
            for j in range(i + 1, len(idx) - 1):
                for k in range(j + 1, len(idx)):
                    a, b, c = self.P[idx[i]], self.P[idx[j]], self.P[idx[k]]
                    area2 = norm(cross(sub(b, a), sub(c, a)))
                    if area2 > self.eps:
                        p0, p1, p2 = idx[i], idx[j], idx[k]
                        break
                if p0 is not None:
                    break
            if p0 is not None:
                break
        if p0 is None:
            raise ValueError("All points collinear: cannot form a base triangle")

        # 2) четверта точка — не копланарна (p0,p1,p2)
        p3 = None
        for t in idx:
            if t in (p0, p1, p2):
                continue
            if abs(orient3d(self.P[p0], self.P[p1], self.P[p2], self.P[t])) > self.eps:
                p3 = t
                break
        if p3 is None:
            raise ValueError("All points coplanar: 3D hull is impossible")

        # 3) чотири грані тетра
        F = [
            self._add_face(p0, p1, p2),
            self._add_face(p0, p2, p3),
            self._add_face(p0, p3, p1),
            self._add_face(p1, p3, p2),
        ]
        # 4) склеїти сусідів
        self._rebuild_all_adjacencies()

        # 5) зорієнтувати назовні (всередині — центроїд тетра)
        O = centroid([self.P[p0], self.P[p1], self.P[p2], self.P[p3]])
        for fid in F:
            a, b, c = self.faces_list[fid].v
            # хочемо orient3d(a,b,c,O) < 0 (O всередині, нормаль назовні)
            if orient3d(self.P[a], self.P[b], self.P[c], O) > 0:
                self.faces_list[fid].v = (a, c, b)

        return F

    def _init_conflicts(self, base_faces: List[int]) -> None:
        """Початковий conflict graph: хто що бачить із решти точок."""
        base_vs = set()
        for fid in base_faces:
            base_vs.update(self.faces_list[fid].v)

        for pi, p in enumerate(self.P):
            if pi in base_vs:
                continue
            for fid in base_faces:
                a, b, c = self.faces_list[fid].v
                if visible_from_point(self.P[a], self.P[b], self.P[c], p, self.eps):
                    self.faces_list[fid].conflict.add(pi)
                    self.point2faces.setdefault(pi, set()).add(fid)

    def _pick_face_with_conflict(self) -> Optional[int]:
        for fid, f in enumerate(self.faces_list):
            if f.alive and f.conflict:
                return fid
        return None

    def _pick_farthest_point(self, fid: int) -> int:
        """Найвіддаленіша від грані точка з її conflict-сету (евристика Quickhull)."""
        a, b, c = (self.P[i] for i in self.faces_list[fid].v)
        best_p = None
        best_dist = -1.0
        for pi in self.faces_list[fid].conflict:
            d = abs(signed_distance_to_plane(a, b, c, self.P[pi]))
            if d > best_dist:
                best_dist = d
                best_p = pi
        assert best_p is not None
        return best_p

    def _collect_visible_region(
        self, seed_fid: int, p_idx: int
    ) -> Tuple[Set[int], List[Tuple[Edge, int, int]]]:
        """
        BFS по видимих граннях від seed_fid (щодо точки p_idx).
        Повертає:
          visible — множину id видимих граней,
          horizon — список кортежів ((u,v), opp_fid, opp_edge_local_index),
                    де opp_fid — сусід по цьому ребру, який НЕ видимий (або -1, якщо None).
        """
        visible: Set[int] = set()
        stack = [seed_fid]
        while stack:
            fid = stack.pop()
            if fid in visible:
                continue
            f = self.faces_list[fid]
            if not f.alive:
                continue
            a, b, c = (self.P[i] for i in f.v)
            if not visible_from_point(a, b, c, self.P[p_idx], self.eps):
                continue
            visible.add(fid)
            # штовхаємо всіх сусідів — видимість перевіримо, коли дістанемо
            for ei in range(3):
                nb = f.nbr[ei]
                if nb is not None and nb not in visible:
                    stack.append(nb)

        # Зібрати горизонт: ребра, де з іншого боку немає видимої грані
        horizon: List[Tuple[Edge, int, int]] = []
        for fid in visible:
            f = self.faces_list[fid]
            for ei in range(3):
                nb = f.nbr[ei]
                u, v = f.edge(ei)  # орієнтація від видимої грані
                if (nb is None) or (nb not in visible):
                    opp_fid = nb if nb is not None else -1
                    opp_edge_idx = -1
                    if nb is not None:
                        nb_f = self.faces_list[nb]
                        # знайдемо локальне ребро у сусіда з тим самим (неорієнтованим) ребром
                        for ej in range(3):
                            uu, vv = nb_f.edge(ej)
                            if (min(uu, vv), max(uu, vv)) == (min(u, v), max(u, v)):
                                opp_edge_idx = ej
                                break
                    horizon.append(((u, v), opp_fid, opp_edge_idx))
        return visible, horizon

    def _add_point_and_update(self, p_idx: int, seed_fid: int) -> None:
        """
        Додати точку p_idx до оболонки:
          1) знайти видимий «ковпак» і горизонт,
          2) знести видимі грані,
          3) пришити нові грані вздовж горизонту, орієнтуючи назовні,
          4) перекинути конфлікти.
        """
        visible, horizon = self._collect_visible_region(seed_fid, p_idx)

        # 1) зібрати всі конфліктні точки з видимих граней
        conflict_points: Set[int] = set()
        for fid in visible:
            conflict_points.update(self.faces_list[fid].conflict)

        # 2) позначити видимі грані мертвими й прибрати їх з edge2face та point2faces
        for fid in visible:
            f = self.faces_list[fid]
            f.alive = False
            # чистимо edge2face
            for ei in range(3):
                u, v = f.edge(ei)
                key = (min(u, v), max(u, v))
                lst = self.edge2face.get(key, [])
                if lst:
                    self.edge2face[key] = [(fa, ea) for (fa, ea) in lst if fa != fid]
            # зняти посилання з point2faces
            for pi in list(f.conflict):
                s = self.point2faces.get(pi)
                if s is not None and fid in s:
                    s.remove(fid)
            f.conflict.clear()

        # 3) створити нові грані вздовж горизонту
        new_fids: List[int] = []
        # для лінкування нових граней між собою по ребрах із p_idx
        edgeP_map: Dict[UEdge, Tuple[int, int]] = {}  # (min,max) -> (fid, local_edge)
        # внутрішня точка для орієнтації нових граней
        O = centroid(self.P)

        for (u, v), opp_fid, opp_ei in horizon:
            a, b, c = u, v, p_idx
            # орієнтація: хочемо orient3d(a,b,c,O) < 0 (O «всередині»)
            if orient3d(self.P[a], self.P[b], self.P[c], O) > 0:
                b, c = c, b
            nf = self._add_face(a, b, c)
            new_fids.append(nf)

            # з'єднати з «невидимим» сусідом через ребро (u,v), якщо він існує
            if opp_fid != -1 and opp_ei != -1:
                self._set_neighbor(nf, 0, opp_fid)      # локальне ребро 0 = (a,b) — це той самий (u,v)
                self._set_neighbor(opp_fid, opp_ei, nf)

            # зв'язати нові грані між собою по ребрах, що містять p_idx
            for (x, y), e_local in [((b, c), 1), ((c, a), 2)]:
                key = (min(x, y), max(x, y))
                if key in edgeP_map:
                    ofid, oei = edgeP_map[key]
                    self._set_neighbor(nf, e_local, ofid)
                    self._set_neighbor(ofid, oei, nf)
                else:
                    edgeP_map[key] = (nf, e_local)

        # 4) пере-розкидати конфліктні точки (окрім самої p_idx)
        conflict_points.discard(p_idx)
        for pi in conflict_points:
            faces_for_pi = self.point2faces.get(pi, set())
            faces_for_pi.clear()
            for nf in new_fids:
                a, b, c = self.faces_list[nf].v
                if visible_from_point(self.P[a], self.P[b], self.P[c], self.P[pi], self.eps):
                    self.faces_list[nf].conflict.add(pi)
                    faces_for_pi.add(nf)
            if faces_for_pi:
                self.point2faces[pi] = faces_for_pi
            elif pi in self.point2faces:
                del self.point2faces[pi]

    def _expand_until_done(self) -> None:
        """Головний цикл: поки існує грань із зовнішніми точками, розширюємо hull."""
        while True:
            fid = self._pick_face_with_conflict()
            if fid is None:
                break
            p_idx = self._pick_farthest_point(fid)
            self._add_point_and_update(p_idx, fid)

    # ---------------- Діагностика / Експорт ----------------
    def validate(self) -> dict:
        """
        Перевірка коректності:
          - кожне неорієнтоване ребро зустрічається рівно у 2 активних гранях;
          - сусідства симетричні (зворотні посилання);
          - орієнтації активних граней «назовні» щодо внутрішньої точки O.
        Повертає словник із діагностикою (порожні списки = все ок).
        """
        faces = [f for f in self.faces_list if f.alive]
        O = centroid(self.P)

        # 1) ребра мають кратність 2
        edge_count: Dict[UEdge, int] = {}
        for f in faces:
            a, b, c = f.v
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edge_count[key] = edge_count.get(key, 0) + 1
        bad_edges = [(e, k) for e, k in edge_count.items() if k != 2]

        # 2) симетрія сусідств (ігноруємо мертвих)
        bad_nbr: List[Tuple[int, int, str]] = []
        for fid, f in enumerate(self.faces_list):
            if not f.alive:
                continue
            for ei in range(3):
                nb = f.nbr[ei]
                if nb is None or not (0 <= nb < len(self.faces_list)) or not self.faces_list[nb].alive:
                    bad_nbr.append((fid, ei, "missing_or_dead_neighbor"))
                    continue
                # знайти відповідне ребро у сусіда
                u, v = f.edge(ei)
                nb_f = self.faces_list[nb]
                found_back = False
                for ej in range(3):
                    uu, vv = nb_f.edge(ej)
                    if {u, v} == {uu, vv} and nb_f.nbr[ej] == fid:
                        found_back = True
                        break
                if not found_back:
                    bad_nbr.append((fid, ei, f"no_backlink_to_{nb}"))

        # 3) орієнтації: хочемо orient3d(a,b,c,O) < 0
        bad_orient: List[int] = []
        for fid, f in enumerate(self.faces_list):
            if not f.alive:
                continue
            a, b, c = f.v
            if orient3d(self.P[a], self.P[b], self.P[c], O) >= 0:
                bad_orient.append(fid)

        return {
            "faces": len(faces),
            "unique_vertices": len({i for f in faces for i in f.v}),
            "bad_edges": bad_edges,
            "bad_neighbors": bad_nbr,
            "bad_orient_faces": bad_orient,
        }

    def to_off(self) -> str:
        """
        Експорт опуклої оболонки у формат OFF (активні грані).
        """
        faces = [f for f in self.faces_list if f.alive]
        used = sorted({i for f in faces for i in f.v})
        remap = {old: new for new, old in enumerate(used)}
        lines = ["OFF", f"{len(used)} {len(faces)} 0"]
        for i in used:
            p = self.P[i]
            lines.append(f"{p.x} {p.y} {p.z}")
        for f in faces:
            a, b, c = (remap[i] for i in f.v)
            lines.append(f"3 {a} {b} {c}")
        return "\n".join(lines)
