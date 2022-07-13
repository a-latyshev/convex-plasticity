"""The module provides a minimal functionality to replace return-mapping procedure of plasticity problems by resolving an additional convex optimization problem.
 
The module is based on `cvxpy` python library for convex optimization problems. It helps to replace a return step in the return-mapping approach, which is commonly used to solve plasticity problems in FEM codes. 

It contains essential classes to describe the elasto-plastic 2D materials behavior with isotropic hardening : 
: failure criteria classes as a future constraint for minimization problems
: particular constitutive laws such that `IsotropicElasticity`
: the abstract material class `Material`
: the class `ReturnMapping` realizing the return step via convex optimization technics 

    Example:

    vonMises = crm.vonMises(sig0, H)
    elastoplastic_material = crm.Material(crm.IsotropicElasticity(E, nu), vonMises)
    return_mapping = ReturnMapping(elastoplastic_material)

Note. We work here with the Voight notation, which can be described as follows 
    |sigma_xx| = |lambda + 2mu, lambda, lambda, 0| |epsilon_xx|
    |sigma_yy| = |lambda, lambda + 2mu, lambda, 0| |epsilon_yy|
    |sigma_zz| = |lambda, lambda, lambda + 2mu, 0| |epsilon_zz|
    |sigma_xy| = |0, 0 ,0 , 2mu|                   |epsilon_xy|

    So o
    .. math::
"""

import cvxpy as cp
import numpy as np

class vonMises:
    def __init__(self, sigma0, hardening):
        self.sig0 = sigma0
        self.H = hardening

    def criterion(self, sig, p):
        dev = np.array([[2/3., -1/3., -1/3., 0],
                        [-1/3., 2/3., -1/3., 0],
                        [-1/3., -1/3., 2/3., 0],
                        [0, 0, 0, 1.]])
        s = dev @ sig
        return [np.sqrt(3/2)*cp.norm(s) <= self.sig0 + p * self.H]

class Rankine:
    def __init__(self):
        self.fc = cp.Parameter()
        self.ft = cp.Parameter()

    def criterion(self, sig):
        Sig = cp.bmat([[sig[0], sig[3]/np.sqrt(2), 0],
                      [sig[3]/np.sqrt(2), sig[1], 0],
                      [0, 0, sig[2]]])
        return [cp.lambda_max(Sig) <= self.ft, cp.lambda_min(Sig) >= -self.fc]


class IsotropicElasticity:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.lambda_ = E*nu/(1+nu)/(1-2*nu)
        self.mu_ = E/2/(1+nu)        

    def C(self):
        l, m = self.lambda_, self.mu_
        return np.array([[l+2*m, l, l, 0],
                         [l, l+2*m, l, 0],
                         [l, l, l+2*m, 0],
                         [0, 0, 0, 2*m]])


class Material:
    def __init__(self, constitutive_law, yield_criterion, plane_stress=False):
        self.C = constitutive_law.C()
        self.criterion = yield_criterion.criterion
        self.plane_stress = plane_stress
                            
class ReturnMapping:
    def __init__(self, material, solver=cp.SCS):
        self.material = material
        self.deps = cp.Parameter((4,))
        self.sig_old = cp.Parameter((4,))
        self.sig_elas = self.sig_old + self.material.C @ self.deps
        self.sig = cp.Variable((4,))
        
        self.p_old = cp.Parameter(nonneg=True)
        self.p = cp.Variable(nonneg=True)

        self.sig_old.value = np.zeros((4,))
        self.deps.value = np.zeros((4,))
        self.p_old.value = 0

        target_expression = cp.quad_form(self.sig - self.sig_elas, np.linalg.inv(self.material.C)) + vonMises.H * cp.square(self.p - self.p_old)

        self.constrains = material.criterion(self.sig, self.p) 

        if self.material.plane_stress:
            self.constrains.append(self.sig[2] == 0)

        self.opt_problem = cp.Problem(cp.Minimize(target_expression), self.constrains)
        self.solver = solver
        
    def solve(self, **kwargs):
        self.opt_problem.solve(solver=self.solver, requires_grad=True, **kwargs)

        self.C_tang = np.zeros((4, 4))
        for i in range(4):
            z = np.zeros((4,))
            z[i] = 1
            self.deps.delta = z
            self.opt_problem.derivative()
            self.C_tang[i, :] = self.sig.delta 
    
# from petsc4py import PETSc

# def to_Voight_vector(sigma):
#     sigma_vec = np.zeros(6, dtype=PETSc.ScalarType)
#     for i in range(3):
#         for j in range(3):
#             sigma_vec[Voight_indexes[i,j]] = sigma[i,j]
#     return sigma_vec

# def to_Voight_matrix(C):
#     C_matrix = np.zeros((6, 6), dtype=PETSc.ScalarType)
    
#     for i in range(3):
#         for j in range(3):
#             for k in range(3):
#                 for l in range(3):
#                     C_matrix[Voight_indexes[i,j], Voight_indexes[k,l]] = C[i,j,k,l]

#     return C_matrix