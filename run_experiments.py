from optimize import optimize


def run_experiments():
    num_runs_per_env = 5
    env_idxs = list(range(10))
    all_triangles = [3, 4, 5]

    results = []
    for num_triangles in all_triangles:
        for env_idx in env_idxs:
            time_to_solutions = []
            for run_idx in range(num_runs_per_env):
                time_to_solution = optimize(
                    num_triangles, env_idx, num_particles=512, visualize=False
                )
                time_to_solutions.append(time_to_solution)

            results.append(
                {
                    "num_triangles": num_triangles,
                    "env_idx": env_idx,
                    "time_to_solutions": time_to_solutions,
                }
            )

    # TODO: collect results into a table


if __name__ == "__main__":
    run_experiments()
