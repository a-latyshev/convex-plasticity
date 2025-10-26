from mpi4py import MPI

import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import json

import os, sys
path_to_src = os.path.join(os.path.dirname(__file__), "./src/")
sys.path.append(path_to_src)
from convex_plasticity import solve_convex_plasticity
import argparse

def solve_convex_plasticity_generic(varying_parameters):
    params = {
        # mesh
        "mesh_size": 0.3,
        # convex solver
        "solver": "CLARABEL", "compiled": False, "patch_size_max": True, "patch_size": None,
        **varying_parameters # update default parameters
    }

    solve_convex_plasticity(params)


parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str, default="1", help="json file with parameters")
parser.add_argument("--jobid", type=str, default="1", help="jobid")
parser.add_argument("--patch-size", type=int, default=3, help="number of quadratures per patch")
parser.add_argument("--patch-size-max", type=bool, default=True, help="if True take max patch size")
parser.add_argument("--solver", type=str, default="CLARABEL", help="convex solver name (e.g. CLARABEL, MOSEK)")
parser.add_argument("--compiled", type=bool, default=False, help="whether to use cvxpygen")
parser.add_argument("--h", type=float, default=0.3, help="mesh size parameter")
args, _ = parser.parse_known_args()

param_file = args.param_file

if not os.path.isfile(param_file):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Parameter file '{param_file}' not found. Using default parameters.", flush=True)
    param_set = {
        # mesh
        "mesh_size": args.h,
        # convex solver
        "solver": args.solver, "compiled": args.compiled, "patch_size_max": args.patch_size_max, "patch_size": args.patch_size,
    }
else:
    with open(param_file, 'r') as f:
        param_set = json.load(f)

param_set_ = {**param_set, "jobid": args.jobid}
if MPI.COMM_WORLD.rank == 0:
    print(f"Running simulation with parameters: \n{param_set_}", flush=True)

solve_convex_plasticity_generic(param_set_)
