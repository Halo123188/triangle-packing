import json
import os.path
import time
from datetime import datetime
from cost_functions import transform_vertices_2d, boundary_cost, overlap_cost_all, ensure_ccw, compactness_cost
from check_valid import constraints_satisfied

import rerun as rr
import torch
import numpy

assets_dir = os.path.join(os.path.dirname(__file__), "assets")


def load_env(num_triangles: int, env_idx: int) -> dict:
    """
    Load the environment. You don't need to modify this function.
    """
    tri_env_path = os.path.join(assets_dir, f"{num_triangles}_triangles.jsonl")
    if not os.path.exists(tri_env_path):
        raise FileNotFoundError(f"File {tri_env_path} does not exist")

    with open(tri_env_path, "r") as f:
        for i, line in enumerate(f):
            if i == env_idx:
                break
        if i != env_idx:
            raise IndexError(f"Trial index {env_idx} out of range for {tri_env_path}")

    # Check there are only num_triangles + 1 objects in the environment
    env = json.loads(line)
    assert (
        len(env) == num_triangles + 1
    ), f"Expected {num_triangles + 1} objects, got {len(env)}"
    print(f"Loaded num_triangles={num_triangles} and idx={env_idx} from {tri_env_path}")
    return env


def visualize_env(env: dict):
    """
    Visualize environment in rerun. You don't need to modify this function.
    """
    for label, obj in env.items():
        shape = obj["shape"]

        if shape == "box":
            extents = obj["extents"]
            half_sizes = [extents[0] / 2, extents[1] / 2, extents[2] / 2]
            centroid = obj["centroid"]

            rr.log(
                f"world/{label}",
                rr.Boxes3D(
                    half_sizes=half_sizes,
                    centers=centroid,
                ),
            )
        elif shape == "arbitrary_triangle":
            vertices = obj["vertices"]
            faces = obj["faces"]
            rgb = obj["color"]

            vertices_3d = [v + [0.0] for v in vertices]
            rr.log(
                f"world/{label}",
                rr.Mesh3D(
                    vertex_positions=vertices_3d,
                    triangle_indices=faces,
                    vertex_colors=[rgb],
                ),
            )
        else:
            raise ValueError(f"Unknown shape {shape}")


def get_goal_aabb(env: dict) -> torch.Tensor:
    """Get goal axis-aligned bounding box (AABB) from environment. You don't need to modify this function."""
    # We only care about x and y so take the first two elements
    goal_extents = torch.tensor(env["goal"]["extents"][:2])
    goal_centroid = torch.tensor(env["goal"]["centroid"][:2])

    # goal_aabb is a 2x2 tensor, where the first row is the min corner and the second row is the max corner
    goal_aabb = torch.stack(
        [
            goal_centroid - goal_extents / 2,
            goal_centroid + goal_extents / 2,
        ]
    )
    return goal_aabb



def optimize(
    num_triangles: int,
    env_idx: int,
    num_particles: int,
    visualize: bool = True,
    device: str = "cpu",
) -> float:
    """
    Solve the triangle packing problem. Returns the time required to find a satisfying particle.
    """
    env = load_env(num_triangles, env_idx)
    max_steps = 500
    lr = 0.1
    tolerance = 1e-3
    boundary_weight = 1.0
    compactness_weight = 0.0
    collision_weight = 1.0

    if visualize:
        recording_id = datetime.now().isoformat().split(".")[0]
        rr.init("triangle_world", recording_id=recording_id, spawn=True)
        visualize_env(env)

    goal_aabb = get_goal_aabb(env).to(device)
    print(f"Goal AABB: {goal_aabb}")

    triangles = {
        label: torch.tensor(obj["vertices"], device=device)
        for label, obj in env.items()
        if label != "goal"
    }

    # Randomly sample xy positions and rotations for each triangle
    # These form the particles to be optimized
    particles = {}
    for triangle in triangles:
        xy = torch.rand(num_particles, 2, device=device)
        xy = xy * (goal_aabb[1] - goal_aabb[0]) + goal_aabb[0]
        rot = torch.rand(num_particles, 1, device=device) * 2 * torch.pi
        xy_rot = torch.cat((xy, rot), dim=1)
        xy_rot.requires_grad_(True)
        particles[triangle] = xy_rot
    
    param_list = [particles[t] for t in particles] # a list of 3 tensor (512, 3)
    optimizer = torch.optim.Adam(param_list, lr=lr)


    # Your code starts here. Feel free to write any additional methods you need.
    # You should track the time required to find a satisfying particle along with other metrics you think are relevant
    start_time = time.perf_counter()

    for step in range(max_steps):
        print(step, '-' * 10)
        optimizer.zero_grad()
        total_cost = torch.zeros(num_particles, device=device)


        # Process each triangle
        for tlabel, local_verts in triangles.items():
            xy_rot = particles[tlabel]
            global_verts = transform_vertices_2d(local_verts, xy_rot)  # shape (512, 3, 2)

            # Boundary cost
            cost_in = boundary_cost(global_verts, goal_aabb)
            total_cost += cost_in * boundary_weight
            assert (cost_in >= 0).all().item(), "Some elements in cost_in are not greater than 0"

            # Compactness cost
            compact_cost = compactness_cost(global_verts, goal_aabb)
            total_cost += compact_cost * compactness_weight
            assert (compact_cost >= 0).all().item(), "Some elements in compact_cost are not greater than 0"

            # # Area coverage cost
            # area_cost = area_coverage_cost(global_verts, goal_aabb)
            # total_cost += area_cost * area_weight
            # assert (area_cost >= 0).all().item(), "Some elements in area_cost are not greater than 0"

        # Overlap cost
        overlap_c = overlap_cost_all(triangles, particles)
        assert (overlap_c >= 0).all().item(), "Some elements in overlap_c are not greater than 0"
        total_cost += overlap_c * collision_weight

        mean_cost = total_cost.mean()
        mean_cost.backward()
        optimizer.step()

        min_cost_val, min_idx = total_cost.min(dim=0)
        # for tlabel, local_verts in triangles.items():
        #     print("Gradients:", particles[tlabel].grad)
        print("Result_cost: ", min_cost_val.item())
        print("Overlap_c: ", overlap_c[min_idx])

        if visualize:
            best_idx = min_idx
            for triangle, vertices in triangles.items():
                xy_rot_best = particles[triangle][best_idx].detach().clone()
                # TODO: transform the triangle vertices by the rotation and xy translation
                global_verts_best = transform_vertices_2d(vertices, xy_rot_best.unsqueeze(0))
                v2d = global_verts_best[0]

                print("Result_pos: ", v2d)

                vertices_3d = torch.cat(
                    (v2d, torch.zeros_like(v2d[:, :1])), dim=1
                )
                        #vertices_3d += 0.25  # offset for sake of this demo
                rr.log(
                    f"world/{triangle}",
                    rr.Mesh3D(
                        vertex_positions=vertices_3d.cpu().tolist(), triangle_indices=[[0, 1, 2]]
                    ),
                )
        
        if min_cost_val.item() < tolerance:
            transformed_triangles = {}
            for label, vertices in triangles.items():
                xy_rot_best = particles[label][min_idx].detach().clone()
                global_verts_best = transform_vertices_2d(vertices, xy_rot_best.unsqueeze(0))  # (1, 3, 2)
                transformed_triangles[label] = global_verts_best[0]  # (3, 2)

        # Check that the constraints are satisfied for this candidate.
            if constraints_satisfied(transformed_triangles, goal_aabb, tol=1e-3):
                time_to_solution = time.perf_counter() - start_time
                print("All consitions satisfied")
                return time_to_solution
            else:
                print("Candidate solution does not satisfy constraints.")

    # No solution found
    return float("inf")


if __name__ == "__main__":
    # Use device="cpu" if you don't have a GPU
    duration = optimize(num_triangles=3, env_idx=0, num_particles=512, device="cpu")
    print("Time to solution:", duration)
