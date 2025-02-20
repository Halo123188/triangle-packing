import torch

def transform_vertices_2d(local_v, xy_rot):
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

def boundary_cost(global_v, goal_aabb, tolerance = 1e-3):
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

def pairwise_overlap_cost(global_v1, global_v2, tolerance = 1e-3):
    c1 = global_v1.mean(dim=1)
    c2 = global_v2.mean(dim=1)

    inside_c1_in_2 = point_in_triangle_cost(c1, global_v1)  # (num_particles,)
    inside_c2_in_1 = point_in_triangle_cost(c2, global_v2)  # (num_particles,)

    # Sum them up as a cost. If zero => likely no overlap in this simple heuristic
    # If > 0 => overlap or near-overlap is suspected
    cost = inside_c1_in_2 + inside_c2_in_1
    return cost

def point_in_triangle_cost(points, tri_vertices):
    A = tri_vertices[:, 0, :]  # shape (num_particles, 2)
    B = tri_vertices[:, 1, :]
    C = tri_vertices[:, 2, :]
    P = points                 # shape (num_particles, 2)

    # Vector cross product helper in 2D
    def cross_2d(u, v):
        # cross_2d((ux,uy), (vx,vy)) = ux*vy - uy*vx
        return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]

    # Edges AP, AB
    AP = P - A
    AB = B - A
    crossAB = cross_2d(AB, AP)  # sign of cross product

    # Edges BP, BC
    BP = P - B
    BC = C - B
    crossBC = cross_2d(BC, BP)

    # Edges CP, CA
    CP = P - C
    CA = A - C
    crossCA = cross_2d(CA, CP)

    # If P is inside the triangle, crossAB, crossBC, crossCA should 
    # all be either >=0 or <=0 (assuming no degeneracies)
    # We'll do a quick sign check:
    def is_positive(x): return x >= 0
    all_pos = is_positive(crossAB) & is_positive(crossBC) & is_positive(crossCA)
    all_neg = (crossAB <= 0) & (crossBC <= 0) & (crossCA <= 0)
    inside  = all_pos | all_neg

    # Convert boolean to float cost. If inside => cost>0, else cost=0
    # You can scale how strongly you penalize being inside by multiplying
    cost = inside.float()  # shape (num_particles,)
    return cost

def overlap_cost_all(triangles, particles_dict):
    """
    Sum overlap cost for all pairs of triangles, for each particle in parallel.
    For convenience, we flatten them so that the result is shape (num_particles,).
    But note each triangle might have a distinct set of 'particles' in the dictionary. 
    Typically we'd use the same number of particles for all. 
    """
    # We'll gather the global coords for each triangle in a list
    # Let's assume each triangle has the same num_particles for simplicity
    triangle_labels = list(triangles.keys())
    n_part = list(particles_dict.values())[0].shape[0]  # number of particles

    global_verts_list = []
    for tlabel in triangle_labels:
        local_verts = triangles[tlabel]         # (3, 2)
        xy_rot = particles_dict[tlabel]         # (num_particles, 3)
        gverts = transform_vertices_2d(local_verts, xy_rot)  # (num_particles, 3, 2)
        global_verts_list.append(gverts)

    # Now compute pairwise overlap costs
    total_overlap = torch.zeros(n_part, device=global_verts_list[0].device)
    for i in range(len(triangle_labels)):
        for j in range(i+1, len(triangle_labels)):
            c = pairwise_overlap_cost(global_verts_list[i], global_verts_list[j])
            total_overlap += c

    return total_overlap