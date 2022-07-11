import cvxpy as cp
import numpy as np

class vonMises:
    def __init__(self, sigma0, hardening):
        self.sig0 = cp.Parameter(nonneg=True, value=sigma0)
        # self.sig0.value = sigma0

        self.H = cp.Parameter(nonneg=True, value=hardening)
        # self.H.value = hardening

    def criterion(self, sig, p, theta):
        dev = np.array([[2/3., -1/3., -1/3., 0],
                        [-1/3., 2/3., -1/3., 0],
                        [-1/3., -1/3., 2/3., 0],
                        [0, 0, 0, 1.]])
        s = dev @ sig
        return [np.sqrt(3/2)*cp.norm(s) <= self.sig0 + theta]

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
    
    # def C(self):
    #     I = np.eye(3)
    #     J4 = 1./3. * np.tensordot(I, I, axes=0)
    #     I4 = np.einsum('ij,kl->ikjl', I, I)
    #     K4 = DEV = I4 - J4
    #     C_elas = (3*self.lambda_ + 2*self.mu_)*J4 + 2*self.mu_*K4
    #     return C_elas

    # Voight notation
    # |sigma_xx| = |lambda + 2mu, lambda, lambda, 0| |epsilon_xx|
    # |sigma_yy| = |lambda, lambda + 2mu, lambda, 0| |epsilon_yy|
    # |sigma_zz| = |lambda, lambda, lambda + 2mu, 0| |epsilon_zz|
    # |sigma_xy| = |0, 0 ,0 , mu|                    |2epsilon_xy|
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
    def __init__(self, mat, solver=cp.SCS):
        self.mat = mat
        self.deps = cp.Parameter((4,))
        self.sig_old = cp.Parameter((4,))
        self.sig_elas = self.sig_old + self.mat.C @ self.deps
    
        self.sig = cp.Variable((4, ))
        obj = cp.quad_form(self.sig - self.sig_elas, np.linalg.inv(self.mat.C))
        
        self.cons = mat.criterion(self.sig)
        if self.mat.plane_stress:
            self.cons.append(self.sig[2] == 0)
        self.prob = cp.Problem(cp.Minimize(obj), self.cons)
        self.solver = solver
        
    def solve(self, **kwargs):
        self.prob.solve(solver=self.solver, requires_grad=True, **kwargs)
        self.sig_old.value = self.sig.value
        self.C_tang = np.zeros((4, 4))
        for i in range(4):
            z = np.zeros((4,))
            z[i] = 1
            self.deps.delta = z
            self.prob.derivative()
            self.C_tang[i, :] = self.sig.delta
    
from petsc4py import PETSc

def to_Voight_vector(sigma):
    sigma_vec = np.zeros(6, dtype=PETSc.ScalarType)
    for i in range(3):
        for j in range(3):
            sigma_vec[Voight_indexes[i,j]] = sigma[i,j]
    return sigma_vec

def to_Voight_matrix(C):
    C_matrix = np.zeros((6, 6), dtype=PETSc.ScalarType)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C_matrix[Voight_indexes[i,j], Voight_indexes[k,l]] = C[i,j,k,l]

    return C_matrix