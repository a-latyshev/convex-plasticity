"""The module provides a minimal functionality to replace return-mapping procedure of plasticity problems by resolving an additional convex optimization problem.
 
The module is based on `cvxpy` python library for convex optimization problems. It helps to replace a return step in the return-mapping approach, which is commonly used to solve plasticity problems in FEM codes. 

It contains essential classes to describe the elasto-plastic 2D materials behavior with isotropic hardening : 
- yield criteria classes as a future constraint for minimization problems
- particular constitutive laws such that `IsotropicElasticity`
- the abstract material class `Material`
- the class `ReturnMapping` realizing the return step via convex optimization technics 

    Example::
        vonMises = crm.vonMises(sig0, H)
        
        elastoplastic_material = crm.Material(crm.IsotropicElasticity(E, nu), vonMises)

        return_mapping = ReturnMapping(elastoplastic_material)
    
Remark. We work here with the Voigt notation, which can be described as follows:
    |sigma_xx| = |lambda + 2mu, lambda, lambda, 0| |epsilon_xx|
    |sigma_yy| = |lambda, lambda + 2mu, lambda, 0| |epsilon_yy|
    |sigma_zz| = |lambda, lambda, lambda + 2mu, 0| |epsilon_zz|
    |sigma_xy| = |0, 0 ,0 , 2mu|                   |epsilon_xy|

    So taking into account the symmetry of main tensors we must to multiply the `xy` components of Voigt strain and stress vectors by the root of 2 to have appropriate results after the inner product of such expressions as 
   
    .. math::
        \varepsilon : \varepsilon, s:s, \sigma:\varepsilon  
"""

import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import block_diag
from dolfinx import common

class YieldCriterion(ABC):
    @abstractmethod
    def criterion(self):
        pass

class vonMises(YieldCriterion):
    """Represents the von Mises yield criterion for elastoplastic materials with the isotropic hardening.

    Attributes:
        sig0: An uniaxial strength [Pa].
        H: A modulus of isotropic hardening [Pa].
    """
    def __init__(self, sigma0:np.float64, hardening:np.float64):
        """Inits vonMises criterion."""
        self.sig0 = sigma0
        self.H = hardening

    def criterion(self, sig:cp.expressions.variable.Variable, p:cp.expressions.variable.Variable):
        """Creates a constraint for convex optimization problem in the form of von Mises criterion.
        
        Args:
            sig: A cvxpy variable of 4-dimensional Voigt vector of stresses
            p: A cvxpy variable of cumulated equivalent plastic strain
        
        Returns:    
            A list with the condition of von Mises yield criterion
        """
        N = p.size
        dev = np.array([[2/3., -1/3., -1/3., 0],
                        [-1/3., 2/3., -1/3., 0],
                        [-1/3., -1/3., 2/3., 0],
                        [0, 0, 0, 1.]])

        sig0 = np.repeat(self.sig0, N)
        s = dev @ sig

        return [np.sqrt(3/2)*cp.norm(s, axis=0) <= sig0 + p * self.H]

class DruckerPrager(YieldCriterion):
    def __init__(self, sigma0: np.float64, alpha: np.float64, hardening: np.float64):
        self.sig0 = sigma0
        self.alpha = alpha 
        self.H = hardening

    def criterion(self, sig: cp.expressions.variable.Variable, p: cp.expressions.variable.Variable):
        N = p.size
        dev = np.array([[2/3., -1/3., -1/3., 0],
                        [-1/3., 2/3., -1/3., 0],
                        [-1/3., -1/3., 2/3., 0],
                        [0, 0, 0, 1.]])
        tr = np.array([1, 1, 1, 0])
        s = dev @ sig
        sig0 = np.repeat(self.sig0, N)
        alpha = self.alpha
        sig_m = tr @ sig 
        return [np.sqrt(3/2)*cp.norm(s, axis=0) + alpha * sig_m <= sig0 + p * self.H]
    

class Rankine(YieldCriterion):
    def __init__(self, ft: np.float64, fc: np.float64, hardening: np.float64):
        self.fc = ft
        self.ft = fc
        self.H = hardening

    def criterion(self, sig: cp.expressions.variable.Variable, p: cp.expressions.variable.Variable):
        N = p.size

        ft = np.repeat(self.ft, N)
        fc = np.repeat(self.fc, N)

        sigma_max = []
        sigma_min = []
        for i in range(N):
            SIG = cp.bmat([[sig[0,i], sig[3,i]/np.sqrt(2), 0],
                           [sig[3,i]/np.sqrt(2), sig[1,i], 0],
                           [0, 0, sig[2,i]]])
            sigma_max.append(cp.lambda_max(SIG))
            sigma_min.append(cp.lambda_min(SIG))

        return [cp.hstack(sigma_max) <= ft + p * self.H, cp.hstack(sigma_min) >= -fc - p * self.H]


class IsotropicElasticity:
    """A constitutive law of isotropic elasticity.
    
    Attributes: 
        E: Young's modulus [Pa].
        nu: Poisson coefficient [-].   
        lambda: Lame's first parameter [Pa].
        mu: shear modulus [Pa] .
    """
    def __init__(self, E, nu):
        """Inits an  IsotropicElasticity class."""
        self.E = E
        self.nu = nu
        self.lambda_ = E*nu/(1+nu)/(1-2*nu)
        self.mu_ = E/2/(1+nu)        

    def C(self):
        """Returns a 4x4 Voigt elastic tensor."""
        l, m = self.lambda_, self.mu_
        return np.array([[l+2*m, l, l, 0],
                         [l, l+2*m, l, 0],
                         [l, l, l+2*m, 0],
                         [0, 0, 0, 2*m]])

class Material:
    """An abstract 2D material class.
    
    Attributes:
        C: A 4x4 Voigt elastic tensor.
        yield_criterion: A yield criterion.
        plane_stress: A boolean flag showing whether we consider a plane stress problem or not.    
    """
    def __init__(self, constitutive_law, yield_criterion:YieldCriterion, plane_stress: bool = False):
        """Inits Material class."""
        self.C = constitutive_law.C()
        self.constitutive_law = constitutive_law
        self.yield_criterion = yield_criterion
        self.plane_stress = plane_stress
                            
class ReturnMapping:
    """An implementation of return-mapping procedure via convex problems solving.

    Attributes:
        deps:
        sig_old:
        sig:
        p_old:
        p:
        C_tang:
        e:
        opt_problem:
        solver:
    """
    def __init__(self, material:Material, N:int, solver=cp.SCS):
        """Inits ReturnMapping class.
        
        Args:
            material: An appropriate material.
            solver: A convex optimization solver
            
        Note:
            We use here `cp.SCS` as it allows to calculate the derivatives of target variables.
        """
        self.N = N
        self.deps = cp.Parameter((4, N), name='deps')
        self.sig_old = cp.Parameter((4, N), name='sig_old')
        sig_elas = self.sig_old + material.C @ self.deps
        self.sig = cp.Variable((4, N), name='sig')
        
        self.p_old = cp.Parameter((N,), nonneg=True, name='p_old')
        self.p = cp.Variable((N,),nonneg=True, name='p')

        self.sig_old.value = np.zeros((4, N))
        self.deps.value = np.zeros((4, N))
        self.p_old.value = np.zeros((N,))
        self.C_tang = np.zeros((N, 4, 4))

        S = np.linalg.inv(material.C)
        delta_sig = self.sig - sig_elas
        # energy = []
        # for i in range(N):
        #     energy.append(cp.quad_form(delta_sig[:, i], S))
        # target_expression = cp.sum(cp.hstack(energy)) + material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)
        
        # energy = cp.sum(cp.diag(delta_sig.T @ S_sparsed @ delta_sig))
        
        S_sparsed = block_diag([S for _ in range(N)])
        delta_sig_vector = cp.reshape(delta_sig, (N*4))

        elastic_energy = cp.quad_form(delta_sig_vector, S_sparsed, assume_PSD=True)
        # target_expression = 0.5*elastic_energy + 0.5*material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)
        D = material.yield_criterion.H * np.eye(N)
        target_expression = 0.5*elastic_energy + 0.5*cp.quad_form(self.p - self.p_old, D)

        constrains = material.yield_criterion.criterion(self.sig, self.p) 

        if material.plane_stress:
            constrains.append(self.sig[2] == 0) #TO MODIFY!

        self.opt_problem = cp.Problem(cp.Minimize(target_expression), constrains)
        self.solver = solver
    
    def solve(self, **kwargs):
        """Solves a minimization problem and calculates the derivative of `sig` variable.
        
        Args:
            **kwargs: additional solver attributes, such as tolerance, etc.
        """
        self.opt_problem.solve(solver=self.solver, requires_grad=False, ignore_dpp=False, **kwargs)
        
    def solve_and_derivate(self, **kwargs):
        """Solves a minimization problem and calculates the derivative of `sig` variable.
        
        Args:
            **kwargs: additional solver attributes, such as tolerance, etc.
        """

        with common.Timer() as t: 
            self.opt_problem.solve(solver=self.solver, requires_grad=True, **kwargs)
            self.convex_solving_time = t.elapsed()[0] 
        
        with common.Timer() as t: 
            for i in range(4):
                for j in range(self.N):
                    e = np.zeros((4, self.N))
                    e[i, j] = 1
                    self.deps.delta = e
                    self.opt_problem.derivative()
                    self.C_tang[j, :, i] = self.sig.delta[:, j] 
            
            self.differentiation_time = t.elapsed()[0] # time.time() - start
    
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