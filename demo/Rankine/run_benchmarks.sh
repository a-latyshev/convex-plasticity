# mpirun -n 4 python demo_plasticity_rankine.py --h 0.025 --solver CLARABEL --patch-size-max True
# mpirun -n 16 python demo_plasticity_rankine.py --h 0.01 --solver CLARABEL --patch-size-max True
# mpirun -n 8 python demo_plasticity_rankine.py --h 0.01 --solver CLARABEL --patch-size-max True
# mpirun -n 4 python demo_plasticity_rankine.py --h 0.01 --solver CLARABEL --patch-size-max True

# mpirun -n 1 python demo_plasticity_rankine.py --h 0.025 --solver MOSEK --patch-size-max True
# mpirun -n 4 python demo_plasticity_rankine.py --h 0.025 --solver MOSEK --patch-size-max True
# mpirun -n 8 python demo_plasticity_rankine.py --h 0.025 --solver MOSEK --patch-size-max True

# mpirun -n 16 python demo_plasticity_rankine.py --h 0.3 --solver MOSEK --patch-size-max True
# mpirun -n 16 python demo_plasticity_rankine.py --h 0.06 --solver MOSEK --patch-size-max True
# mpirun -n 16 python demo_plasticity_rankine.py --h 0.025 --solver MOSEK --patch-size-max True
# mpirun -n 16 python demo_plasticity_rankine.py --h 0.01 --solver MOSEK --patch-size-max True


mpirun -n 4 python demo_plasticity_rankine.py --h 0.3 --solver MOSEK --patch-size-max True
mpirun -n 4 python demo_plasticity_rankine.py --h 0.06 --solver MOSEK --patch-size-max True
mpirun -n 4 python demo_plasticity_rankine.py --h 0.025 --solver MOSEK --patch-size-max True
mpirun -n 4 python demo_plasticity_rankine.py --h 0.01 --solver MOSEK --patch-size-max True

# h = array([0.3  , 0.06 , 0.01 , 0.025])