import convex_return_mapping as crm # there is a conflict in the oder of imported modules

from mpi4py import MPI

import plasticity_framework as pf
# import sys
# sys.path.append("../")
# import fenicsx_support as fs

import matplotlib.pyplot as plt
import numpy as np

import logging

Pa_dim = 70e3
E = 70e3 / Pa_dim #[-]
nu = 0.3 #[-]
sig0 = 250 / Pa_dim #[-]
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus

vonMises = crm.vonMises(sig0, H)
material_vM = crm.Material(crm.IsotropicElasticity(E, nu), vonMises)

alpha = 0

DruckerPrager = crm.DruckerPrager(sig0, alpha, H)
material_DP = crm.Material(crm.IsotropicElasticity(E, nu), DruckerPrager)

logger_file = logging.getLogger('analysis_file')
logger_file.setLevel(logging.DEBUG)
# logger_file.setLevel(logging.INFO)
logger_file.setLevel(pf.LOG_INFO_STAR)
fh = logging.FileHandler("log/plasticity_analysis_ENPC.log", mode='a')
logger_file.addHandler(fh)

patch_sizes = [250]
conic_solvers = ['MOSEK']

time_medium_mesh = []
for conic_solver in conic_solvers:
    time = []
    for size in patch_sizes:
        logger_file.log(pf.LOG_INFO_STAR, f'conic solver = {conic_solver}, patch size = {size}')
        plasticity_convex = pf.ConvexPlasticity(material_DP, logger=logger_file,  solver='SNESQN', mesh_name="thick_cylinder.msh", patch_size=size, conic_solver=conic_solver, tol_conic_solver=1e-13)
        _, _, T, _, _, _ = plasticity_convex.solve()
        time.append(T)
        
    time_medium_mesh.append(time)

    #30000 GiB for 500 ECOS!