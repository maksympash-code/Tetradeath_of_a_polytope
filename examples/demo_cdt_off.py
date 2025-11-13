# examples/demo_cdt_off.py
from cg3d.geom import unique_points
from cg3d.mesh import Delaunay3D

if __name__ == "__main__":
    # куб + кілька внутрішніх
    raw = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
        (0.5,0.5,0.5), (0.2,0.8,0.3), (0.8,0.2,0.7)
    ]
    pts = unique_points(raw)

    d3 = Delaunay3D(pts)
    d3.build()
    d3.remove_super_tetra()

    report = d3.mesh.validate()
    print("VALIDATION:", report)

    d3.mesh.write_boundary_off("cdt_boundary.off")
    print("Wrote cdt_boundary.off — відкривай у MeshLab/ParaView.")
