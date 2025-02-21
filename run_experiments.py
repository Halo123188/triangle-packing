from optimize import optimize


def run_experiments():
    num_runs_per_env = 5
    env_idxs = list(range(10))
    all_triangles = [3]

    results = []
    for num_triangles in all_triangles:
        for env_idx in env_idxs:
            time_to_solutions = []
            for run_idx in range(num_runs_per_env):
                time_to_solution = optimize(
                    num_triangles, env_idx, num_particles=512, visualize=True
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
    rows = []
    for r in results:
        times = np.array(r["time_to_solutions"])
        mean_time = times.mean()
        std_time = times.std(ddof=1)
        ci = 1.96 * std_time / np.sqrt(num_runs_per_env)
        rows.append({
            "Triangles": r["num_triangles"],
            "Environment": r["env_idx"],
            "Time to Solution (s)": f"{mean_time:.3f} ± {ci:.3f}"
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_experiments()
