# examples/demo_cdt.py
from cg3d.geom import unique_points
from cg3d.mesh import Delaunay3D

if __name__ == "__main__":
    # куб + внутрішні точки
    raw = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
        (0.5,0.5,0.5), (0.2,0.8,0.3), (0.8,0.2,0.7)
    ]
    pts = unique_points(raw)

    d3 = Delaunay3D(pts)
    d3.build()                    # вставляє всі, крім супер-вершин
    d3.remove_super_tetra()       # прибирає тетри, що торкаються супер-вершин

    # простенька статистика
    alive_tets = [t for t in d3.mesh.tets if t.alive]
    print("tets:", len(alive_tets))
    print("boundary faces:", len(d3.mesh.extract_boundary_faces()))
