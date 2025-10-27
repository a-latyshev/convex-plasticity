import json
import numpy as np

solvers = ["CLARABEL", "MOSEK", "SCS"]
n_processes = [8, 4, 2, 1]
mesh_sizes = [0.3, 0.06, 0.025, 0.01]
compiled_options = [True, False]

# solvers = ["MOSEK"]
# n_processes = [5, 6]
# mesh_sizes = [0.3]

params = []

for h in mesh_sizes:
    for n in n_processes:
        for solver in solvers:
            for compiled in compiled_options:
                    params.append({
                        "solver": solver, "compiled": compiled, "mesh_size": h, "n_processes": n, "patch_size_max": True,
                    })

with open("parameters.json", "w") as f:
    json.dump(params, f, indent=4)

