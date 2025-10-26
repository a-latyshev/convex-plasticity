#!/bin/bash -l

# Generate JSON-file with parameters
python parameters_gnu_parallel.py

# Extract parameters from JSON file
PARAM_FILE="parameters.json"

# Create a directory for individual parameter files

# JOB_ID=$(uuidgen)
JOB_ID="$(date +%Y%m%d%H%M%S)-$$"
PARAMS_DIR="logs/${JOB_ID}/params"
mkdir -p "$PARAMS_DIR"

# Split the big JSON into many small JSON files (one per parameter set)
python3 -c '
import json, os, sys
with open(sys.argv[1]) as f:
    params = json.load(f)
for i, param in enumerate(params):
    fname = os.path.join(sys.argv[2], f"param_{i:04d}.json")
    with open(fname, "w") as out:
        json.dump(param, out)
' "$PARAM_FILE" "$PARAMS_DIR"

# Create a list of all parameter files
PARAMS_LIST_FILE="logs/${JOB_ID}/params_list.txt"
find "$PARAMS_DIR" -name 'param_*.json' | sort > "$PARAMS_LIST_FILE"

# Define the parallel job log file
PARALLEL_JOBLOG_FILE="logs/${JOB_ID}/${JOB_ID}-parallel.out"

# Run parallel tasks
echo "Starting parallel..."

# Path to the installed cargo to compile CARLABEL 
# TODO: adjust if needed
export PATH="$PATH:/root/.cargo/bin"
parallel --delay 0.2 --jobs 8 \
    --joblog ${PARALLEL_JOBLOG_FILE} \
    "NUM_PROC=\$(python3 -c 'import json; print(json.load(open(\"{}\"))[\"n_processes\"])'); \
     mpirun -n \$NUM_PROC python3 run_convex_plasticity.py --param_file {} --jobid JOB_ID > logs/${JOB_ID}/job_{#}.out 2>&1" \
    :::: ${PARAMS_LIST_FILE}

echo "Finished."
