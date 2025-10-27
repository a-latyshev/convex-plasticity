# Convex-plasticity v2.0 - coupling with external operators

Author: Andrey Latyshev

This is an attempt to revitalize the `convex-plasticity` project via
`dolfinx-external-operator`.

The current work is based on the tutorial of [von Mises
plasticity](https://a-latyshev.github.io/dolfinx-external-operator/demo/demo_plasticity_von_mises.html)
within external operators.

## Workflow

**Summary - 27/10/2025**

* Any run (simple or with GNU Parallel) will dump input parameters and output
  data as `dict` via `pickle` in `output/`. This makes it easy to upload
  all output files in one place.
* See `convex_plasticity_analysis.ipynb` for post-processing.
* Tested on `dolfinx:v0.10.0`

**File structure**

`./external-operators/`:

- `demo_convex_plasticity.py` is a temporary playground, outdated. See
  `convex_plasticity.py`
- `src/convex_plasticity.py` is a main source file.
-  `demo_plasticity_von_mises_pure_ufl.py` is a highly efficient implementation
   of von Mises.
- `run_benchmarks.sh` is an outdated way to run scripts.
- `utilities.py`
- `notebooks/convex_plasticity_analysis.ipynb` is a main post-processing file.
- `run_convex_plasticity.py` is a wrapper to run `convex_plasticity.py` with
  default input parameters.
- `run_gnu_parallel.sh` is a principle script to interact with GNU Parallel.
  - It generates all possible combinations of parameters from
    `parameters_gnu_parallel.py`,
  - makes directories for logs,
  - runs multiple jobs; job per parameter set.
- `parameters_gnu_parallel.py` creates a JSON file with parameter sets of interest.

### Simple run

```shell
cd external-operators
mpirun -n 1 python run_convex_plasticity.py --h 0.3 --solver CLARABEL --compiled False --patch-size-max
```

Simple run supports the following flags:

* Path to JSON with input parameters (for GNU Parallel): `--param_file
  "params.json"`.
* Job ID: `--jobid 123`.
* Size of patch in quadratures: `--patch-size 10`.
* Use max patch size, meaning all quadratures on a process are used to solve the
convex problem in one pass: `--patch-size-max`.
* Convex solver supported by CVXPY: `--solver CLARABEL`.
* Compile with CVXPYGen: `--compiled`.
* Mesh size: `--h 0.3`.

Test compilation of CLARABEL with CVXPYGen
```shell
cd external-operators
export PATH="$PATH:/root/.cargo/bin"
mpirun -n 1 python run_convex_plasticity.py --h 0.3 --solver CLARABEL --compiled
```

### GNU parallel

Massive automatization of multiple runs.

1. Set up parameter sets in `parameters_gnu_parallel.py`.
2. Run GNU parallel script:
```shell
cd external-operators
bash run_gnu_parallel.sh
```
3. Study failed runs in `logs/<jobid>`.
4. Collect all `pkl` files with pandas (see `convex_plasticity_analysis.ipynb`)
   in the output folder (`external-operators/output/`, by default).

**Tips**
* In `run_gnu_parallel.sh`, there is `parallel --delay 0.2 --jobs 8 ...`.
  `--jobs 8` means number of jobs that will be run in parallel. For example, if you run something like
```shell
parallel --delay 0.2 --jobs 8 \
  "mpirun -n 16 python3 run_convex_plasticity.py --param_file {} --jobid ${JOB_ID} > logs/${JOB_ID}/job_{#}.out 2>&1" \
  :::: ${PARAMS_LIST_FILE}
``` 
then 16x8 = 128 CPUs will be occupied simultaneously. Make sure that you have
that many.
* Current design likes producing many small files (logs, compilation, data).
  That's why at some point, the user has to erase them manually.

## Installation

*!!Install in a container only!!*

Main requirements

```shell 
pip install cvxpy pandas mosek clarabel
```

MOSEK

```shell
mkdir ~/mosek
cp mosek.lic ~/mosek/mosek.lic
```

**dolfinx-external-operator**

```shell
git clone https://github.com/a-latyshev/dolfinx-external-operator.git
cd dolfinx-external-operator
pip install .
```

**GNU Parallel**

```shell
apt update
apt install parallel locales
```

**CVXPYGen**

```shell
pip install cvxpygen
```

Auto-installations (?):
 - Julia

To enable CLARABEL:
```shell
apt update
apt install cargo
cargo install cbindgen --locked
export PATH="$PATH:/root/.cargo/bin"
apt install rustup
rustup update nightly
```

**cvxpylayers**

```shell
pip install cvxpylayers jax==0.5.3
```
