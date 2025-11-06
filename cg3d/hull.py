from __future__ import annotations
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, List, Optional, Set, Tuple

from cg3d.geom import Pt, centroid, EPS, sub, cross, norm
from cg3d.predicates import orient3d, signed_distance_to_plane, visible_from_point

Edge = Tuple[int, int]         # орієнтоване ребро (u,v)
UEdge = Tuple[int, int]        # неорієнтоване ребро (min,max)

@dataclass
class Face:
    """Трикутна грань опуклої оболонки."""
    v: Tuple[int, int, int]                     # індекси вершин (узгоджена орієнтація: нормаль назовні)
    nbr: List[Optional[int]] = field(default_factory=lambda: [None, None, None])   # сусіди через ребра 0,1,2
    alive: bool = True
    conflict: Set[int] = field(default_factory=set)  # індекси точок, що бачать цю грань

    def edge(self, i: int) -> Edge:
        a, b, c = self.v
        if i == 0:   return (a, b)
        if i == 1:   return (b, c)
        return (c, a)

    def tri(self) -> Tuple[int, int, int]:
        return self.v


class ConvexHull3D:
    """
    Рандомізований інкрементальний 3D convex hull з conflict graph.

    Вхід: список точок Pt (мінімум 4 не копланарні).
    Вихід: self.faces_list — трикутні грані з правильною орієнтацією (нормалі назовні).
    """

    def __init__(self, points: List[Pt], eps: float = EPS):
        if len(points) < 4:
            raise ValueError("Need at least 4 points")
        self.P: List[Pt] = points[:]  # індексована копія
        self.eps = eps

        # динамічні структури
        self.faces_list: List[Face] = []          # всі створені грані (деякі можуть бути dead)
        self.edge2face: Dict[UEdge, List[Tuple[int, int]]] = {}  # (u,v sorted) -> [(face_id, local_edge), ...]
        self.point2faces: Dict[int, Set[int]] = {}               # p -> видимі грані

        # 1) стартовий тетраедр
        base = self._build_initial_tetra()
        self._init_conflicts(base)

        # 2) основний цикл: поки є "зовнішні" точки
        self._expand_until_done()

        # прибрати мертві грані
        self.faces_list = [f for f in self.faces_list if f.alive]

    # ---------- паблік API ----------
    def faces(self) -> List[Tuple[int, int, int]]:
        """Список трикутників у вигляді індексів вершин."""
        return [f.v for f in self.faces_list]

    # ---------- внутрішня кухня ----------
    def _add_face(self, v0: int, v1: int, v2: int) -> int:
        """Створити грань і внести її ребра до карти."""
        fid = len(self.faces_list)
        face = Face((v0, v1, v2))
        self.faces_list.append(face)
        # запишемо у edge2face
        for ei in range(3):
            u, v = face.edge(ei)
            key = (min(u, v), max(u, v))
            self.edge2face.setdefault(key, []).append((fid, ei))
        return fid

    def _set_neighbor(self, fid_a: int, edge_a: int, fid_b: Optional[int]):
        """Прописати сусіда через локальне ребро edge_a у грані fid_a."""
        self.faces_list[fid_a].nbr[edge_a] = fid_b

    def _build_initial_tetra(self) -> List[int]:
        """
        Знаходимо перші 4 не копланарні точки:
          - p0,p1,p2: неколінеарні (|| (b-a) x (c-a) || > eps)
          - p3: не копланарна до (p0,p1,p2) (|orient3d| > eps)
        Створюємо 4 грані тетра з правильною (зовнішньою) орієнтацією.
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
                if p0 is not None: break
            if p0 is not None: break
        if p0 is None:
            raise ValueError("All points collinear: cannot form a base triangle")

        # 2) четверта точка — не копланарна базовій площині
        p3 = None
        for t in idx:
            if t in (p0, p1, p2):
                continue
            if abs(orient3d(self.P[p0], self.P[p1], self.P[p2], self.P[t])) > self.eps:
                p3 = t
                break
        if p3 is None:
            raise ValueError("All points coplanar: 3D hull is impossible")

        # 3) створюємо 4 грані тетра
        F = [
            self._add_face(p0, p1, p2),
            self._add_face(p0, p2, p3),
            self._add_face(p0, p3, p1),
            self._add_face(p1, p3, p2),
        ]
        # склеїти сусідства
        self._rebuild_all_adjacencies()

        # 4) зорієнтувати назовні (всередині — центроїд тетра)
        O = centroid([self.P[p0], self.P[p1], self.P[p2], self.P[p3]])
        for fid in F:
            a, b, c = self.faces_list[fid].v
            # хочемо orient3d(a,b,c,O) < 0 (O всередині, нормаль назовні)
            if orient3d(self.P[a], self.P[b], self.P[c], O) > 0:
                self.faces_list[fid].v = (a, c, b)

        return F

    def _rebuild_all_adjacencies(self):
        """Переприв'язати nbr для всіх граней за edge2face."""
        for face in self.faces_list:
            face.nbr = [None, None, None]
        for key, lst in self.edge2face.items():
            if len(lst) == 2:
                (fa, ea), (fb, eb) = lst[0], lst[1]
                self._set_neighbor(fa, ea, fb)
                self._set_neighbor(fb, eb, fa)

    def _init_conflicts(self, base_faces: List[int]):
        """Початковий conflict graph: хто що бачить."""
        base_set = set()
        for fid in base_faces:
            base_set.update(self.faces_list[fid].v)
        # для всіх інших точок — розкидати по видимих гранях
        for pi, p in enumerate(self.P):
            if pi in base_set:
                continue
            for fid in base_faces:
                a, b, c = self.faces_list[fid].v
                if visible_from_point(self.P[a], self.P[b], self.P[c], p, self.eps):
                    self.faces_list[fid].conflict.add(pi)
                    self.point2faces.setdefault(pi, set()).add(fid)

    def _expand_until_done(self):
        """Основний цикл: поки є конфлікти (зовнішні точки), розширюємо оболонку."""
        while True:
            fid = self._pick_face_with_conflict()
            if fid is None:
                break
            p_idx = self._pick_farthest_point(fid)
            self._add_point_and_update(p_idx, fid)

    def _pick_face_with_conflict(self) -> Optional[int]:
        for fid, f in enumerate(self.faces_list):
            if f.alive and f.conflict:
                return fid
        return None

    def _pick_farthest_point(self, fid: int) -> int:
        """Найвіддаленіша від грані точка з її конфлікт-сету (евристика Quickhull)."""
        a, b, c = (self.P[i] for i in self.faces_list[fid].v)
        best_p = None
        best_dist = -1.0
        for pi in self.faces_list[fid].conflict:
            d = abs(signed_distance_to_plane(a, b, c, self.P[pi]))
            if d > best_dist:
                best_dist = d; best_p = pi
        assert best_p is not None
        return best_p

    def _collect_visible_region(self, seed_fid: int, p_idx: int) -> Tuple[Set[int], List[Tuple[Edge, int, int]]]:
        """
        BFS по видимих граннях від seed_fid.
        Повертає:
          - visible: множина id видимих граней;
          - horizon: список (орієнтоване ребро (u,v), opp_fid, opp_edge_local_index),
            де opp_fid — сусід через це ребро, який НЕ видимий (або None).
        """
        visible: Set[int] = set()
        stack = [seed_fid]
        while stack:
            fid = stack.pop()
            if fid in visible or not self.faces_list[fid].alive:
                continue
            f = self.faces_list[fid]
            a, b, c = (self.P[i] for i in f.v)
            if not visible_from_point(a, b, c, self.P[p_idx], self.eps):
                continue
            visible.add(fid)
            # дивимось сусідів через ребра
            for ei in range(3):
                nb = f.nbr[ei]
                if nb is None:
                    # на межі — це теж горизонт
                    pass
                else:
                    # якщо сусід теж видимий — добалансимо пізніше
                    if nb not in visible:
                        stack.append(nb)
        # Зібрати горизонт
        horizon: List[Tuple[Edge, int, int]] = []
        for fid in visible:
            f = self.faces_list[fid]
            for ei in range(3):
                nb = f.nbr[ei]
                u, v = f.edge(ei)  # орієнтація з видимої грані
                if (nb is None) or (nb not in visible):
                    # межа "видимий—невидимий": це горизонт
                    # треба знати локальний edge index у сусіда nb
                    opp_fid = nb if nb is not None else -1
                    opp_edge_idx = -1
                    if nb is not None:
                        # знайдемо локальний індекс ребра у сусіда
                        nb_face = self.faces_list[nb]
                        for e2 in range(3):
                            uu, vv = nb_face.edge(e2)
                            if (min(uu, vv), max(uu, vv)) == (min(u, v), max(u, v)):
                                opp_edge_idx = e2
                                break
                    horizon.append(((u, v), opp_fid, opp_edge_idx))
        return visible, horizon

    def _add_point_and_update(self, p_idx: int, seed_fid: int):
        """Додати точку p_idx до оболонки: зрізати видимий 'ковпак' і зашити по горизонту новими гранями."""
        visible, horizon = self._collect_visible_region(seed_fid, p_idx)

        # 1) зібрати всі конфліктні точки, пов'язані з видимими гранями
        conflict_points: Set[int] = set()
        for fid in visible:
            conflict_points.update(self.faces_list[fid].conflict)

        # 2) позначити видимі грані мертвими та прибрати їх з edge2face
        for fid in visible:
            f = self.faces_list[fid]
            f.alive = False
            # прибрати записи ребер -> face (залишимо "дірки" — це норм)
            for ei in range(3):
                u, v = f.edge(ei)
                key = (min(u, v), max(u, v))
                lst = self.edge2face.get(key, [])
                self.edge2face[key] = [(fa, ea) for (fa, ea) in lst if fa != fid]
            # прибрати з point2faces
            for pi in f.conflict:
                s = self.point2faces.get(pi)
                if s is not None and fid in s:
                    s.remove(fid)
            f.conflict.clear()

        # 3) створити нові грані вздовж горизонту
        new_fids: List[int] = []
        # Для лінкування між новими гранями по ребрах, що містять p_idx
        edgeP_map: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (min, max) -> (fid, local_edge)

        # внутрішня опорна точка для орієнтації (після зрізу всередині залишається стара геометрія,
        # візьмемо центр мас вершин усіх НЕмертвих граней; спрощено — центроїд усіх вершин hull)
        # тут просто візьмемо центроїд усіх точок (ок для орієнтації нової грані)
        O = centroid(self.P)

        for (u, v), opp_fid, opp_ei in horizon:
            # нова грань (u, v, p_idx); зорієнтуємо так, щоб O була "зсередини" (neg)
            a, b, c = u, v, p_idx
            if orient3d(self.P[a], self.P[b], self.P[c], O) > 0:
                b, c = c, b  # інверсія
            nf = self._add_face(a, b, c)
            new_fids.append(nf)

            # сусід через ребро (u,v) — старий невидимий opp_fid (якщо існує)
            if opp_fid != -1:
                self._set_neighbor(nf, 0, opp_fid)  # локальне ребро 0 = (a,b)
                # і з боку opp_fid прописати зворотній зв'язок
                if opp_ei != -1:
                    self._set_neighbor(opp_fid, opp_ei, nf)

            # підготуємо лінкування між новими гранями по ребрах, що містять p_idx
            # ребро (b,c)=(b,p) -> локальне 1; ребро (c,a)=(p,a) -> локальне 2
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
            # після видалення видимих граней зв'язки могли спорожніти
            faces_for_pi.clear()
            for nf in new_fids:
                a, b, c = self.faces_list[nf].v
                if visible_from_point(self.P[a], self.P[b], self.P[c], self.P[pi], self.eps):
                    self.faces_list[nf].conflict.add(pi)
                    faces_for_pi.add(nf)
            if faces_for_pi:
                self.point2faces[pi] = faces_for_pi
            elif pi in self.point2faces:
                # точка стала внутрішньою — більше не зовнішня
                del self.point2faces[pi]



    def validate(self) -> dict:
        """
        Легка самоперевірка:
          - кожне ребро має рівно 2 інцидентні граничні грані;
          - орієнтація граней «назовні» (всередині точка O).
        Повертає словник із діагностикою.
        """
        # зібрати лише живі грані
        faces = [f for f in self.faces_list if f.alive]
        # центроїд усіх точок як «внутрішня» точка
        O = centroid(self.P)

        # підрахунок кратностей ребер
        edge_count: dict[tuple[int,int], int] = {}
        bad_orient: list[int] = []

        for fid, f in enumerate(self.faces_list):
            if not f.alive:
                continue
            a, b, c = f.v
            if orient3d(self.P[a], self.P[b], self.P[c], O) >= 0:
                bad_orient.append(fid)
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edge_count[key] = edge_count.get(key, 0) + 1

        bad_edges = [(e, k) for e, k in edge_count.items() if k != 2]

        return {
            "faces": len(faces),
            "unique_vertices": len({i for f in faces for i in f.v}),
            "bad_edges": bad_edges,          # ребра з не-2 інцидентними гранями
            "bad_orient_faces": bad_orient,  # ідентифікатори граней з неправильною орієнтацією
        }

    def to_off(self) -> str:
        """
        OFF-представлення поточної оболонки (тільки живі грані).
        Повертає текстовий вміст OFF (v/f/e — e ставимо 0).
        """
        faces = [f for f in self.faces_list if f.alive]
        used = sorted({i for f in faces for i in f.v})
        remap = {old: new for new, old in enumerate(used)}
        # вершини
        lines = ["OFF", f"{len(used)} {len(faces)} 0"]
        for i in used:
            p = self.P[i]
            lines.append(f"{p.x} {p.y} {p.z}")
        # грані
        for f in faces:
            a, b, c = (remap[i] for i in f.v)
            lines.append(f"3 {a} {b} {c}")
        return "\n".join(lines)
