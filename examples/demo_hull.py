from cg3d.geom import unique_points, Pt
from cg3d.hull import ConvexHull3D

if __name__ == "__main__":
    raw = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
        (0.5,0.5,0.5), (0.2,0.8,0.3), (0.8,0.2,0.7)
    ]
    pts = unique_points(raw)
    hull = ConvexHull3D(pts)
    print("Faces (triangles as vertex indices):")
    for tri in hull.faces():
        print(tri)

    print(hull.validate())
    open("hull.off", "w", encoding="utf-8").write(hull.to_off())
