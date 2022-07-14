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
        dev = np.array([[2/3., -1/3., -1/3., 0],
                        [-1/3., 2/3., -1/3., 0],
                        [-1/3., -1/3., 2/3., 0],
                        [0, 0, 0, 1.]])
        s = dev @ sig
        return [np.sqrt(3/2)*cp.norm(s) <= self.sig0 + p * self.H]

class Rankine(YieldCriterion):
    def __init__(self):
        self.fc = cp.Parameter()
        self.ft = cp.Parameter()

    def criterion(self, sig:cp.expressions.variable.Variable):
        Sig = cp.bmat([[sig[0], sig[3]/np.sqrt(2), 0],
                      [sig[3]/np.sqrt(2), sig[1], 0],
                      [0, 0, sig[2]]])
        return [cp.lambda_max(Sig) <= self.ft, cp.lambda_min(Sig) >= -self.fc]


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
    def __init__(self, material:Material, solver=cp.SCS):
        """Inits ReturnMapping class.
        
        Args:
            material: An appropriate material.
            solver: A convex optimization solver
            
        Note:
            We use here `cp.SCS` as it allows to calculate the derivatives of target variables.
        """
        material = material
        self.deps = cp.Parameter((4,))
        self.sig_old = cp.Parameter((4,))
        sig_elas = self.sig_old + material.C @ self.deps
        self.sig = cp.Variable((4,))
        
        self.p_old = cp.Parameter(nonneg=True)
        self.p = cp.Variable(nonneg=True)

        self.sig_old.value = np.zeros((4,))
        self.deps.value = np.zeros((4,))
        self.p_old.value = 0
        self.C_tang = np.zeros((4, 4))
        self.e = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        target_expression = cp.quad_form(self.sig - sig_elas, np.linalg.inv(material.C)) + material.yield_criterion.H * cp.square(self.p - self.p_old)

        constrains = material.yield_criterion.criterion(self.sig, self.p) 

        if material.plane_stress:
            constrains.append(self.sig[2] == 0)

        self.opt_problem = cp.Problem(cp.Minimize(target_expression), constrains)
        self.solver = solver
        
    def solve(self, **kwargs):
        """Solves a minimization problem and calculates the derivative of `sig` variable.
        
        Args:
            **kwargs: additional solver attributes, such as tolerance, etc.
        """
        self.opt_problem.solve(solver=self.solver, requires_grad=True, **kwargs)
        for i in range(4):
            self.deps.delta = self.e[i]
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