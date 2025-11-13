# cg3d/mesh.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from random import shuffle

from .geom import Pt, centroid, EPS
from .predicates import orient3d, insphere

FaceKey = Tuple[int, int, int]  # відсортована трійка вершин грані

@dataclass
class Tet:
    """
    Тетраедр у сітці.
    v[i] — вершина, протилежна грані i. Отже, грань i містить три вершини v[(i+1)%4], v[(i+2)%4], v[(i+3)%4].
    nbr[i] — сусід через грань i (індекс тетра, або -1 якщо межа).
    """
    v: Tuple[int, int, int, int]
    nbr: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])
    alive: bool = True

    def face_vertices(self, i: int) -> Tuple[int, int, int]:
        a, b, c, d = self.v
        if i == 0: return (b, c, d)
        if i == 1: return (a, c, d)
        if i == 2: return (a, b, d)
        return (a, b, c)

class TetMesh:
    """
    Мінімальна структура 3D тетра-сітки:
      - points: список Pt
      - tets: масив Tet
      - facemap: sorted(face) -> [(tet_id, local_face_idx), ...]
    """
    def __init__(self, points: List[Pt]):
        self.points: List[Pt] = points[:]  # глобальна таблиця вершин
        self.tets: List[Tet] = []
        self.facemap: Dict[FaceKey, List[Tuple[int, int]]] = {}

    def add_tet(self, v0: int, v1: int, v2: int, v3: int) -> int:
        tid = len(self.tets)
        t = Tet((v0, v1, v2, v3))
        # зорієнтуємо тетраедр так, щоб orient3d(v0,v1,v2,v3) > 0 (позитивна орієнтація)
        if orient3d(self.points[v0], self.points[v1], self.points[v2], self.points[v3]) < 0:
            # поміняємо дві вершини місцями (парна перестановка для зміни знаку)
            t.v = (v0, v2, v1, v3)
        self.tets.append(t)
        # зареєструвати грані у facemap
        for i in range(4):
            a, b, c = t.face_vertices(i)
            key = tuple(sorted((a, b, c)))
            self.facemap.setdefault(key, []).append((tid, i))
        return tid

    def link(self, ta: int, fa: int, tb: int, fb: int) -> None:
        self.tets[ta].nbr[fa] = tb
        self.tets[tb].nbr[fb] = ta

    def remove_tet(self, tid: int) -> None:
        """Позначити тетру як мертву й прибрати її грані з facemap."""
        if tid < 0 or tid >= len(self.tets) or not self.tets[tid].alive:
            return
        t = self.tets[tid]
        t.alive = False
        for i in range(4):
            a, b, c = t.face_vertices(i)
            key = tuple(sorted((a, b, c)))
            lst = self.facemap.get(key, [])
            if lst:
                self.facemap[key] = [(tt, ff) for (tt, ff) in lst if tt != tid]

    # ---------- корисні операції ----------
    def neighbors(self, tid: int) -> List[int]:
        return self.tets[tid].nbr[:]

    def extract_boundary_faces(self) -> List[Tuple[int, int, int]]:
        """Повертає всі граничні трикутники (face має рівно 1 інцидентний тет)."""
        out: List[Tuple[int, int, int]] = []
        for key, lst in self.facemap.items():
            alive_lst = [(t,f) for (t,f) in lst if self.tets[t].alive]
            if len(alive_lst) == 1:
                out.append(key)  # key вже відсортована трійка індексів вершин
        return out

    def remove_tets_touching(self, verts: Set[int]) -> None:
        """Прибрати всі тетри, що мають будь-яку з вершин у множині verts (для видалення «супер-тетра»)."""
        for tid, t in enumerate(self.tets):
            if not t.alive: continue
            if any(v in verts for v in t.v):
                self.remove_tet(tid)

    # ---------- locate (скелет) ----------
    def locate(self, p: Pt, start_tid: Optional[int] = None) -> Optional[int]:
        """
        Знаходить тетру, що містить точку p, простим «walking».
        Повертає tid або None (поза сіткою). Скелет без хитрих прискорень.
        """
        # вибір старту
        cur = start_tid
        if cur is None:
            # знайдемо будь-яку живу тетру (на практиці тут тримаємо "останню успішну")
            for i, t in enumerate(self.tets):
                if t.alive:
                    cur = i
                    break
        if cur is None:
            return None

        # ідемо поки виходимо назовні через якусь грань
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            t = self.tets[cur]
            if not t.alive:
                break
            inside_all = True
            # для кожної грані: нормаль «назовні» визначаємо як orient3d(face, opposite) > 0
            for fi in range(4):
                a, b, c = t.face_vertices(fi)
                opp = t.v[fi]
                # напрям нормалі від (a,b,c) у бік "зовні" відносно opp:
                # «всередині» якщо orient3d(a,b,c,p) та orient3d(a,b,c,opp) мають протилежні знаки
                sign_opp = orient3d(self.points[a], self.points[b], self.points[c], self.points[opp])
                sign_p = orient3d(self.points[a], self.points[b], self.points[c], p)
                if sign_opp * sign_p >= 0:  # p по «зовнішній» стороні або на межі — переходимо до сусіда
                    nb = t.nbr[fi]
                    if nb == -1:
                        return None  # пішли назовні
                    cur = nb
                    inside_all = False
                    break
            if inside_all:
                return cur
        return None


    # ---------- валідація сітки ----------
    def validate(self) -> dict:
        """
        Швидка перевірка коректності тетра-сітки:
          - орієнтація кожної живої тетри позитивна;
          - гранична грань має рівно 1 інцидентний живий тет, внутрішня — рівно 2;
          - сусідства симетричні (дзеркальні посилання).
        Повертає словник з діагностикою.
        """
        bad_orientation: list[int] = []
        bad_face_multiplicity: list[tuple[tuple[int,int,int], int]] = []
        bad_neighbors: list[tuple[int,int,str]] = []

        # 0) зберемо лише живі тетри
        alive_tets = [i for i, t in enumerate(self.tets) if t.alive]

        # 1) перевірка орієнтації
        for tid in alive_tets:
            t = self.tets[tid]
            a, b, c, d = t.v
            if orient3d(self.points[a], self.points[b], self.points[c], self.points[d]) <= 0:
                bad_orientation.append(tid)

        # 2) кратність граней (через facemap)
        #    facemap уже містить інцидентні (tet_id, face_idx); рахуємо лише живі
        for key, lst in self.facemap.items():
            alive_inc = [(tt, ff) for (tt, ff) in lst if 0 <= tt < len(self.tets) and self.tets[tt].alive]
            k = len(alive_inc)
            if k not in (1, 2):
                bad_face_multiplicity.append((key, k))

        # 3) симетрія сусідств
        for tid in alive_tets:
            t = self.tets[tid]
            for fi in range(4):
                nb = t.nbr[fi]
                a, b, c = t.face_vertices(fi)
                key = tuple(sorted((a, b, c)))
                if nb == -1:
                    # це має бути гранична грань → у facemap рівно 1 живий інцидент
                    inc = [(tt, ff) for (tt, ff) in self.facemap.get(key, []) if self.tets[tt].alive]
                    if len(inc) != 1:
                        bad_neighbors.append((tid, fi, "boundary_face_inconsistent"))
                    continue
                # має бути живий сусід
                if not (0 <= nb < len(self.tets)) or not self.tets[nb].alive:
                    bad_neighbors.append((tid, fi, "dead_or_invalid_neighbor"))
                    continue
                # у сусіда має бути та сама грань і зворотне посилання
                nb_t = self.tets[nb]
                found = False
                for fj in range(4):
                    aa, bb, cc = nb_t.face_vertices(fj)
                    if tuple(sorted((aa, bb, cc))) == key and nb_t.nbr[fj] == tid:
                        found = True
                        break
                if not found:
                    bad_neighbors.append((tid, fi, f"no_backlink_to_{nb}"))

        return {
            "tets_alive": len(alive_tets),
            "bad_orientation": bad_orientation,                # список tid із неправильною орієнтацією
            "bad_face_multiplicity": bad_face_multiplicity,    # [(face_key, count!=1/2), ...]
            "bad_neighbors": bad_neighbors,                    # [(tid, fi, reason), ...]
        }

    # ---------- OFF-експорт граничної поверхні ----------
    def boundary_off(self) -> str:
        """
        Повертає OFF для граничної поверхні сітки (faces з кратністю 1).
        """
        # 1) зібрати boundary faces
        bfaces = self.extract_boundary_faces()  # список відсортованих трійок індексів
        used = sorted({v for tri in bfaces for v in tri})
        remap = {old: i for i, old in enumerate(used)}

        # 2) вершини
        lines = ["OFF", f"{len(used)} {len(bfaces)} 0"]
        for vi in used:
            p = self.points[vi]
            lines.append(f"{p.x} {p.y} {p.z}")

        # 3) грані
        for tri in bfaces:
            a, b, c = (remap[tri[0]], remap[tri[1]], remap[tri[2]])
            lines.append(f"3 {a} {b} {c}")

        return "\n".join(lines)

    def write_boundary_off(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.boundary_off())


class Delaunay3D:
    """
    Скелет інкрементальної 3D Делоне з порожньою сферою (Bowyer–Watson).
    Без «обмежень» (PLC) — додамо на етапі CDT.
    """
    def __init__(self, points: List[Pt], eps: float = EPS):
        self.eps = eps
        self.mesh = TetMesh(points)
        self.super_verts: Tuple[int, int, int, int] | None = None
        self._seed_tid: Optional[int] = None  # останній знайдений, для locate-walk

    # ---- супер-тетра ----
    def _build_super_tetra(self, scale: float = 1000.0) -> Tuple[int, int, int, int]:
        xs = [p.x for p in self.mesh.points]
        ys = [p.y for p in self.mesh.points]
        zs = [p.z for p in self.mesh.points]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        cz = (min(zs) + max(zs)) * 0.5
        dx = (max(xs) - min(xs)) or 1.0
        dy = (max(ys) - min(ys)) or 1.0
        dz = (max(zs) - min(zs)) or 1.0
        R = scale * max(dx, dy, dz)

        # 4 вершини великого тетраедра навколо всіх точок
        a = Pt(cx - R, cy - R, cz - R)
        b = Pt(cx + R, cy + R, cz - R)
        c = Pt(cx + R, cy - R, cz + R)
        d = Pt(cx - R, cy + R, cz + R)
        ia = len(self.mesh.points); self.mesh.points.append(a)
        ib = len(self.mesh.points); self.mesh.points.append(b)
        ic = len(self.mesh.points); self.mesh.points.append(c)
        id = len(self.mesh.points); self.mesh.points.append(d)
        self.mesh.add_tet(ia, ib, ic, id)
        self.super_verts = (ia, ib, ic, id)
        self._seed_tid = 0
        return self.super_verts

    # ---- вставка однієї точки ----
    def insert(self, p_idx: int) -> None:
        p = self.mesh.points[p_idx]
        # 1) locate
        start_tid = self._seed_tid
        tid = self.mesh.locate(p, start_tid)
        if tid is None:
            # поза поточною сіткою (таке майже не має траплятися з адекватним супер-тетрою)
            # fallback: пошук першої живої
            for i, t in enumerate(self.mesh.tets):
                if t.alive:
                    tid = i; break
            if tid is None:
                return

        # 2) знайти cavity: тетри, у яких p всередині circumsphere
        cavity: Set[int] = set()
        stack = [tid]
        while stack:
            cur = stack.pop()
            if cur in cavity: continue
            t = self.mesh.tets[cur]
            if not t.alive: continue
            a, b, c, d = t.v
            if insphere(self.mesh.points[a], self.mesh.points[b], self.mesh.points[c], self.mesh.points[d], p) > self.eps:
                cavity.add(cur)
                # розширюємо по сусідах
                for nb in t.nbr:
                    if nb != -1 and nb not in cavity:
                        stack.append(nb)

        if not cavity:
            # p уже «зовні» всіх сфер — нічого не робимо
            return

        # 3) зібрати boundary faces (грані cavity, з іншого боку яких тетра не в cavity)
        boundary_faces: List[Tuple[FaceKey, Tuple[int,int,int]]] = []  # (key, (a,b,c))
        for ct in cavity:
            t = self.mesh.tets[ct]
            for fi in range(4):
                a, b, c = t.face_vertices(fi)
                key = tuple(sorted((a, b, c)))
                # «інцидентні» тетри у facemap
                inc = [tt for (tt, ff) in self.mesh.facemap.get(key, []) if self.mesh.tets[tt].alive]
                # якщо серед «активних» інцидентів є тетра НЕ з cavity — це гранична грань
                if any(tt not in cavity for tt in inc):
                    boundary_faces.append((key, (a, b, c)))

        # 4) видалити cavity (позначити мертвими і почистити facemap)
        for ct in cavity:
            self.mesh.remove_tet(ct)

        # 5) пришити нові тетри (p_idx + кожна boundary face)
        new_tets: List[int] = []
        # тимчасова мапа для зшивки нових між собою: (sorted(face_with_p)) -> (tid, local_face_idx)
        stitch_map: Dict[FaceKey, Tuple[int, int]] = {}
        for key, (a, b, c) in boundary_faces:
            # сформуємо новий тет: (a, b, c, p)
            tid_new = self.mesh.add_tet(a, b, c, p_idx)
            new_tets.append(tid_new)

            # зв'язок із «зовнішнім» тетром, який по той бік грані (a,b,c)
            inc = self.mesh.facemap.get(tuple(sorted((a, b, c))), [])
            # серед інцидентів шукаємо активний не-новий (бо наша грань вже записана)
            ext = None
            ext_fi = None
            for (tt, ff) in inc:
                if tt != tid_new and self.mesh.tets[tt].alive:
                    ext = tt; ext_fi = ff; break
            if ext is not None:
                # місцева грань у новому тетрі, протилежна вершині p_idx, — це face index 3
                self.mesh.link(tid_new, 3, ext, ext_fi)

            # зшивка нових між собою по гранях, що містять p
            for face in ((b, c, p_idx), (a, p_idx, c), (a, b, p_idx)):
                fkey = tuple(sorted(face))
                # локальні індекси граней нової тетри, що містять p_idx:
                # у нашому Tet(face index 0..3): face 3 — (a,b,c); решта три — містять p_idx.
                # Потрібно знайти локальний індекс грані у tid_new, відповідної 'face':
                local = _local_face_index_for_vertices(self.mesh.tets[tid_new], face)
                prev = stitch_map.get(fkey)
                if prev is None:
                    stitch_map[fkey] = (tid_new, local)
                else:
                    other_tid, other_local = prev
                    self.mesh.link(tid_new, local, other_tid, other_local)

        # 6) оновити seed для локалізації наступної точки
        if new_tets:
            self._seed_tid = new_tets[0]

    def build(self, insert_order: Optional[List[int]] = None) -> None:
        """Побудувати Делоне-тетраедралізацію для поточних points (окрім супер-вершин)."""
        super_vs = self._build_super_tetra()
        n = len(self.mesh.points)
        verts = list(range(n))
        # не вставляємо супер-вершини
        for s in super_vs:
            verts.remove(s)
        if insert_order is None:
            shuffle(verts)
        else:
            verts = insert_order[:]
        for vi in verts:
            self.insert(vi)

    def remove_super_tetra(self) -> None:
        """Прибрати усі тетри, що торкаються супер-вершин."""
        if not self.super_verts:
            return
        self.mesh.remove_tets_touching(set(self.super_verts))

# ---------- утиліти ----------
def _local_face_index_for_vertices(t: Tet, face: Tuple[int, int, int]) -> int:
    target = set(face)
    for i in range(4):
        if set(t.face_vertices(i)) == target:
            return i
    raise ValueError("face not found in tet")
