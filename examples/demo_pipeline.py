# examples/demo_pipeline.py
from cg3d.pipeline import tetrahedralize_convex

if __name__ == "__main__":
    cube = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
        (0.5,0.5,0.5), (0.2,0.8,0.3), (0.8,0.2,0.7)
    ]

    pts, surface, tets = tetrahedralize_convex(cube, backend="internal")  # або "scipy"
    print("Vertices:", len(pts))
    print("Surface triangles:", len(surface))
    print("Tets:", len(tets))
