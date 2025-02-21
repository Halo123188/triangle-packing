import torch

def triangles_overlap_torch(tri1, tri2, tol=1e-6):
    """
    Check if two triangles overlap 
    """
    def get_edges(tri):
        edges = [tri[(i+1) % 3] - tri[i] for i in range(3)]
        return torch.stack(edges, dim=0)  # (3, 2)
    
    def perpendicular(v):
        return torch.stack([-v[1], v[0]])
    
    def project_triangle(tri, axis):
        axis = axis / torch.norm(axis)
        projections = torch.matmul(tri, axis)
        return projections.min(), projections.max()
    
    edges1 = get_edges(tri1)
    edges2 = get_edges(tri2)
    axes = []
    for edge in edges1:
        axes.append(perpendicular(edge))
    for edge in edges2:
        axes.append(perpendicular(edge))
    
    for axis in axes:
        min1, max1 = project_triangle(tri1, axis)
        min2, max2 = project_triangle(tri2, axis)
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True


def constraints_satisfied(triangles, goal_aabb, tol=1e-10):
    """
    Check that a set of triangles satisfies constraints:
    """
    lower = goal_aabb[0]
    upper = goal_aabb[1]
    
    # Boundary Constraint: 
    for label, tri in triangles.items():
        # print("test: ", tri)
        # print("test: ", lower)
        # print("test: ", upper)
        if not ((tri >= lower - tol).all() and (tri <= upper + tol).all()):
            print(f"Triangle {label} violates boundary constraints.")
            return False

    # Collision Constraint: 
    labels = list(triangles.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            tri1 = triangles[labels[i]]
            tri2 = triangles[labels[j]]
            if triangles_overlap_torch(tri1, tri2, tol=tol):
                print(f"Triangles {labels[i]} and {labels[j]} overlap.")
                return False

    return True