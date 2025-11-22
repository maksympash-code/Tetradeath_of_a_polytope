# examples/main.py
from __future__ import annotations

from cg3d.pipeline import tetrahedralize_convex
from cg3d.mesh import TetMesh
from cg3d.hull import ConvexHull3D


def write_off_from_faces(path: str, pts, faces) -> None:
    """
    OFF для трикутної поверхні, заданої вершинами pts (Pt) і списком граней faces.
    faces — список (i,j,k) з індексами у pts.
    """
    used = sorted({i for tri in faces for i in tri})
    remap = {old: new for new, old in enumerate(used)}

    lines = []
    lines.append("OFF")
    lines.append(f"{len(used)} {len(faces)} 0")

    # вершини
    for i in used:
        p = pts[i]
        lines.append(f"{p.x} {p.y} {p.z}")

    # грані
    for (a, b, c) in faces:
        aa, bb, cc = remap[a], remap[b], remap[c]
        lines.append(f"3 {aa} {bb} {cc}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_mesh_from_tets(pts, tets):
    """
    Збирає TetMesh з готових тетраедрів (індекси у pts).
    1) add_tet(...)
    2) за facemap відновлює сусідів.
    """
    mesh = TetMesh(pts)
    # 1) додаємо всі тетри
    for (i0, i1, i2, i3) in tets:
        mesh.add_tet(i0, i1, i2, i3)

    # 2) зшиваємо сусідів через facemap
    for key, lst in mesh.facemap.items():
        alive = [(t, f) for (t, f) in lst if mesh.tets[t].alive]
        if len(alive) == 2:
            (t1, f1), (t2, f2) = alive
            mesh.link(t1, f1, t2, f2)
    return mesh


def main():
    # --- 1) Вхідні дані ---
    # Можеш змінити на читання з файлу / рандом
    points = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (0.5, 0.5, 0.5),
        (0.2, 0.8, 0.3),
        (0.8, 0.2, 0.7),
    ]

    # --- 2) Пайплайн: оболонка + Делоне ---
    pts, surface, tets = tetrahedralize_convex(points, backend="scipy")

    print(f"Вершини:          {len(pts)}")
    print(f"Граней оболонки:  {len(surface)}")
    print(f"Тетраедрів:       {len(tets)}")

    # --- 3) hull.off — опукла оболонка ---
    # Варіант 1: через наш ConvexHull3D.to_off():
    hull = ConvexHull3D(pts)
    with open("hull.off", "w", encoding="utf-8") as f:
        f.write(hull.to_off())
    print("hull.off записано (опукла оболонка).")

    # (Альтернатива: write_off_from_faces("hull.off", pts, surface))

    # --- 4) Збираємо TetMesh з тетраедрів Делоне ---
    mesh = build_mesh_from_tets(pts, tets)

    # --- 5) Валідація тетра-сітки ---
    report = mesh.validate()
    print("VALIDATION:", report)

    # --- 6) boundary.off — гранична поверхня тетра-сітки ---
    mesh.write_boundary_off("boundary.off")
    print("boundary.off записано (граничні трикутники сітки).")

    # --- 7) volume.vtk — повна тетра-сітка ---
    mesh.write_vtk_unstructured("volume.vtk")
    print("volume.vtk записано (вся тетра-сітка для ParaView/MeshLab).")


if __name__ == "__main__":
    main()
