import torch

def transform_vertices_2d(local_v, xy_rot): #transform local vertices to global vertices
    x, y, theta = xy_rot[..., 0], xy_rot[..., 1], xy_rot[..., 2]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    R = torch.stack([torch.stack([cos_t, -sin_t], dim=-1),
                     torch.stack([sin_t,  cos_t], dim=-1)], dim=-2)

    # print("R's Shape: ", R.shape)
    # print("local's Shape: ", local.shape)

    rotated = torch.matmul(local_v, R)
    xy_trans = torch.stack([x, y], dim=-1).unsqueeze(1)

    # print("Rotated Shape: ", rotated.shape)
    # print("XY Shape: ", xy_trans.shape)

    global_vertices = rotated + xy_trans
    return global_vertices

def boundary_cost(global_v, goal_aabb, tolerance = 1e-3): #calculate the boundary cost
    #gloval_v (512, 3, 2)
    xmin , ymin = goal_aabb[0, 0], goal_aabb[0, 1]
    xmax , ymax = goal_aabb[1, 0], goal_aabb[1, 1]

    left = torch.relu(xmin + tolerance - global_v[..., 0]) #(512, 3)
    right = torch.relu(global_v[..., 0] - xmax + tolerance)
    down = torch.relu(ymin + tolerance - global_v[..., 1])
    up = torch.relu(global_v[..., 1] - ymax + tolerance)

    cost_per_vertex = left + right + down + up

    cost = cost_per_vertex.sum(dim = -1) #(512,)

    return cost

def ensure_ccw(tri): # tri.shape = (512, 3, 2) Ensure the triangles are CCW
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]

    area = 0.5 * ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - 
                  (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))

    swap = area < 0
    tri_fixed = tri.clone()
    tri_fixed[swap, 1, :], tri_fixed[swap, 2, :] = \
        tri_fixed[swap, 2, :].clone(), tri_fixed[swap, 1, :].clone()
    
    return swap

def sdf_triangle_edge(tri_target, tri_source, eps=1e-6):
    """
    Compute the SDF at the vertices with respect to the triangles in tri_source.
    """
    tri_target_next = torch.roll(tri_target, shifts=-1, dims=1)  # (N, 3, 2)
    e = tri_target_next - tri_target                              
    n = torch.stack([e[..., 1], -e[..., 0]], dim=-1)              # (N, 3, 2)
    edge_len = torch.norm(e, dim=-1, keepdim=True) + eps          # (N, 3, 1)
    n_norm = n / edge_len                                         # (N, 3, 2)
    
    tri_target_edge = tri_target.unsqueeze(2)                     # (N, 3, 1, 2)
    tri_source_edge = tri_source.unsqueeze(1)                     # (N, 1, 3, 2)
    diff = tri_source_edge - tri_target_edge                      # (N, 3, 1, 2)
    n_norm_exp = n_norm.unsqueeze(2)                              # (N, 3, 1, 2)
    d = torch.sum(diff * n_norm_exp, dim=-1)                       # (N, 3, 3)

    sdf_vert, _ = torch.max(d, dim=1)                              # (N, 3)
    return sdf_vert

def sdf_overlap_cost(tri1, tri2, eps=1e-6):
    """
    Compute an SDF cost between two triangles. 
    """
    sdf1 = sdf_triangle_edge(tri2, tri1, eps)  # SDF of tri1 vertices wrt tri2
    sdf2 = sdf_triangle_edge(tri1, tri2, eps)  # SDF of tri2 vertices wrt tri1
    penalty1 = torch.relu(-sdf1)  # (N, 3)
    penalty2 = torch.relu(-sdf2)  # (N, 3)
    cost = penalty1.sum(dim=1) + penalty2.sum(dim=1)  # (N,)
    return cost

def triangle_circle(tri):
    """
    Approximate a triangle by a circle.
    """
    centroid = tri.mean(dim=1)  # (N, 2)
    dists = torch.norm(tri - centroid.unsqueeze(1), dim=-1)  # (N, 3)
    radius_min, _ = dists.min(dim=1)  # (N,)
    radius_max, _ = dists.max(dim=1)
    top2, _ = dists.topk(2, dim=1)
    radius_second = top2[:, 1]
    return centroid, radius_min

def triangle_incircle(tri):
    """
    Compute the incircle (incenter and inradius) of each triangle.
    """
    A = tri[:, 0]  # (N, 2)
    B = tri[:, 1]
    C = tri[:, 2]
    
    a = torch.norm(B - C, dim=1)  # (N,)
    b = torch.norm(A - C, dim=1)  # (N,)
    c = torch.norm(A - B, dim=1)  # (N,)
    s = (a + b + c) / 2.0  # (N,)
    
    area = 0.5 * torch.abs(
        A[:, 0] * (B[:, 1] - C[:, 1]) +
        B[:, 0] * (C[:, 1] - A[:, 1]) +
        C[:, 0] * (A[:, 1] - B[:, 1])
    )  # (N,)
    
    inradius = area / s  # (N,)
    incenter = (a.unsqueeze(1) * A + b.unsqueeze(1) * B + c.unsqueeze(1) * C) / (a + b + c).unsqueeze(1)
    
    return incenter, inradius

def circle_overlap_cost(tri1, tri2):
    """
    Compute the overlap between two triangles approximated by circles.
    """
    c1, r1 = triangle_circle(tri1)
    c2, r2 = triangle_circle(tri2)
    dist_centroids = torch.norm(c1 - c2, dim=-1)
    cost = torch.relu((r1 + r2) - dist_centroids)
    return cost

def incircle_overlap_cost(tri1, tri2):
    """
    Compute the overlap between two triangles approximated by incircles.
    """
    c1, r1 = triangle_incircle(tri1)
    c2, r2 = triangle_incircle(tri2)
    dist_centroids = torch.norm(c1 - c2, dim=-1)
    cost = torch.relu((r1 + r2) - dist_centroids)
    return cost

def is_obtuse_batch(tri):
    """
    Determine whether the triangle is obtuse or not
    """

    #vertices
    p0 = tri[0, 0, :]  # (N, 2)
    p1 = tri[0, 1, :]
    p2 = tri[0, 2, :]

    # Compute squared lengths of each side
    d01 = ((p1 - p0) ** 2).sum()  # (N,)
    d12 = ((p2 - p1) ** 2).sum()  # (N,)
    d20 = ((p0 - p2) ** 2).sum()  # (N,)

    sides_sq = torch.tensor([d01, d12, d20], dtype=torch.float32)
    sides_sq_sorted, _ = sides_sq.sort()

    a2, b2, c2 = sides_sq_sorted 
    # c^2 > a^2 + b^2 => obtuse
    obtuse_mask = c2 > (a2 + b2)  # (N,)
    return obtuse_mask

def combined_circle_overlap_cost(tri1, tri2):
    """
    If the triangle is obtuse, use the approximated circle; if the triangle is not obtuse, use the incircle. 
    By doing this, ensure the circles approximate most of the triangles, but doesn't over-represent
    """
    if is_obtuse_batch(tri1).any():
        c1, r1 = triangle_circle(tri1)
    else:
        c1, r1 = triangle_incircle(tri1)
    if is_obtuse_batch(tri2).any():
        c2, r2 = triangle_circle(tri2)
    else:
        c2, r2 = triangle_incircle(tri2)
    
    dist = torch.norm(c1 - c2, dim=-1)
    cost = torch.relu((r1 + r2) - dist)
    return cost

def combined_overlap_loss(tri1, tri2, lambda_circle=0.5, lambda_SDF=1.0, lambda_incircle=0.5, lambda_combined_circle=0.0, eps=1e-6):
    """
    Combine all the possible cost functions
    """
    L_SDF = sdf_overlap_cost(tri1, tri2, eps)  # (N,)
    L_circle = circle_overlap_cost(tri1, tri2)   # (N,)
    L_incircle = incircle_overlap_cost(tri1, tri2)
    L_combined_circle = combined_circle_overlap_cost(tri1, tri2)
    return L_SDF * lambda_SDF + lambda_circle * L_circle + lambda_incircle * L_incircle +  lambda_combined_circle * L_combined_circle ** 2

def overlap_cost_all(triangles, particles_dict):
    """
    Sum overlap cost for all pairs of triangles
    """
    triangle_labels = list(triangles.keys())
    n_part = list(particles_dict.values())[0].shape[0]  # number of particles

    global_verts_list = []
    for tlabel in triangle_labels:
        local_verts = triangles[tlabel]         # (3, 2)
        xy_rot = particles_dict[tlabel]         # (num_particles, 3)
        gverts = transform_vertices_2d(local_verts, xy_rot)  # (num_particles, 3, 2)
        global_verts_list.append(gverts)

    total_overlap = torch.zeros(n_part, device=global_verts_list[0].device)
    for i in range(len(triangle_labels)):
        for j in range(i+1, len(triangle_labels)):
            c = combined_overlap_loss(global_verts_list[i], global_verts_list[j])
            total_overlap += c

    return total_overlap

def compactness_cost(global_vertices, goal_aabb):
    """
    Compute a cost that encourages triangles to be packed more compactly towards the center.
    """
    goal_center = (goal_aabb[0] + goal_aabb[1]) / 2

    centers = torch.mean(global_vertices, dim=1)  # (N, 2)

    box_diagonal = torch.norm(goal_aabb[1] - goal_aabb[0])
    distances = torch.norm(centers - goal_center, dim=1) / box_diagonal
    
    return distances
