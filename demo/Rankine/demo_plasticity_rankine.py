# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dolfinx-env (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plasticity of von Mises
#
# ## Implementation
#
# ### Preamble

# %%
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numba
import numpy as np
from demo_plasticity_von_mises_pure_ufl import plasticity_von_mises_pure_ufl
from solvers import PETScNonlinearProblem, PETScNonlinearSolver
from utilities import build_cylinder_quarter, find_cell_by_point

import basix
import ufl
from dolfinx import fem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# %% [markdown]
# Here we define geometrical and material parameters of the problem as well as some useful constants.

# %%
R_e, R_i = 1.3, 1.0  # external/internal radius
E0 = 70e3 
E, nu = 70e3 / E0, 0.3  # elastic parameters
E_tangent = E / 100.0  # tangent modulus
H = E * E_tangent / (E - E_tangent)  # hardening modulus
sigma_0 = 250.0 / E0  # yield strength
sigt = 250.0 / E0  # tensile strength
sigc = 250.0 / E0  # compression strength

lmbda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu)
mu = E / 2.0 / (1.0 + nu)
# stiffness matrix
C_elas = np.array(
    [
        [lmbda + 2.0 * mu, lmbda, lmbda, 0.0],
        [lmbda, lmbda + 2.0 * mu, lmbda, 0.0],
        [lmbda, lmbda, lmbda + 2.0 * mu, 0.0],
        [0.0, 0.0, 0.0, 2.0 * mu],
    ],
    dtype=PETSc.ScalarType,
)

deviatoric = np.eye(4, dtype=PETSc.ScalarType)
deviatoric[:3, :3] -= np.full((3, 3), 1.0 / 3.0, dtype=PETSc.ScalarType)

# %%
mesh, facet_tags, facet_tags_labels = build_cylinder_quarter()

# %%
k_u = 2
V = fem.functionspace(mesh, ("Lagrange", k_u, (mesh.geometry.dim,)))
# Boundary conditions
bottom_facets = facet_tags.find(facet_tags_labels["Lx"])
left_facets = facet_tags.find(facet_tags_labels["Ly"])

bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, bottom_facets)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, left_facets)

sym_bottom = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), bottom_dofs_y, V.sub(1))
sym_left = fem.dirichletbc(np.array(0.0, dtype=PETSc.ScalarType), left_dofs_x, V.sub(0))

bcs = [sym_bottom, sym_left]


def epsilon(v):
    grad_v = ufl.grad(v)
    return ufl.as_vector([grad_v[0, 0], grad_v[1, 1], 0, np.sqrt(2.0) * 0.5 * (grad_v[0, 1] + grad_v[1, 0])])


k_stress = 2 * (k_u - 1)
ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": k_stress, "quadrature_scheme": "default"},
)

Du = fem.Function(V, name="displacement_increment")
S_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress, value_shape=(4,))
S = fem.functionspace(mesh, S_element)
sigma = FEMExternalOperator(epsilon(Du), function_space=S)

n = ufl.FacetNormal(mesh)
loading = fem.Constant(mesh, PETSc.ScalarType(0.0))

v = ufl.TestFunction(V)
F = ufl.inner(sigma, epsilon(v)) * dx - ufl.inner(loading * -n, v) * ds(facet_tags_labels["inner"])

# Internal state
P_element = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=k_stress)
P = fem.functionspace(mesh, P_element)

p = fem.Function(P, name="cumulative_plastic_strain")
p_n = fem.Function(P, name="cumulative_plastic_strain_n")
dp = fem.Function(P, name="incremental_plastic_strain")
sigma_n = fem.Function(S, name="stress_n")


# %% [markdown]
# ### Defining the external operator
#

# %%
import cvxpy as cp
from scipy.sparse import block_diag

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
    def __init__(self, constitutive_law, yield_criterion, plane_stress: bool = False):
        """Inits Material class."""
        self.C = constitutive_law.C()
        self.constitutive_law = constitutive_law
        self.yield_criterion = yield_criterion
        self.plane_stress = plane_stress


class vonMises():
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

class Rankine():
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

class ConvexPlasticity:
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
        """Inits ConvexPlasticity class.
        
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
            constrains.append(self.sig[2] == 0) #TODO: MODIFY!

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


# %%
rankine = Rankine(sigt, sigc, H)
von_mises = vonMises(sigma_0, H)
material = Material(IsotropicElasticity(E, nu), von_mises)

patch_size = 3
return_mapping = ConvexPlasticity(material, patch_size, 'CLARABEL')
tol = 1.0e-13
scs_params = {'eps': tol, 'eps_abs': tol, 'eps_rel': tol}
conic_solver_params = {}

# %% [markdown]
# Now nothing stops us from defining the implementation of the external operator
# derivative (the tangent tensor $\boldsymbol{C}_\text{tang}$) in the
# function `C_tang_impl`. It returns global values of the derivative, stress
# tensor and the cumulative plastic increment.

# %%
stress_dim = 4
sigma_n.x.array[:] = 0.0
p_n.x.array[:] = 0.0

# %%
num_quadrature_points = int(sigma_n.x.array.size / stress_dim)

# deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
# sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
# p_ = p.x.array.reshape((num_cells, num_quadrature_points))

# _, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)
N_patches = int(num_quadrature_points / patch_size)
residue_size = num_quadrature_points % patch_size
p_vals = np.empty_like(p.x.array)
# p_values = p.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))
p_values = p_vals[:num_quadrature_points - residue_size].reshape((-1, patch_size))
p_old_values = p_n.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))
# deps_values = deps.x.array[:4*(num_quadrature_points -
# residue_size)].reshape((-1, patch_size, 4))

sigma_vals = np.empty_like(sigma.ref_coefficient.x.array)
sig_values = sigma_vals[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
sig_old_values = sigma_n.x.array[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))

if residue_size != 0:
    return_mapping_residue = ConvexPlasticity(material, residue_size, 'CLARABEL')
    # p_values_residue = p.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
    p_values_residue = p_vals[num_quadrature_points - residue_size:].reshape((1, residue_size))
    p_old_values_residue = p_n.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
    deps_values_residue = deps.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
    sig_values_residue = sigma_vals[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
    sig_old_values_residue = sigma_n.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))

def sigma_impl(deps):
    deps_values = deps[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    # if residue_size != 0:
        # sig_values_residue = sig.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        # sig_old_values_residue = sig_old.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))

    for q in range(N_patches):
        return_mapping.deps.value[:] = deps_values[q,:].T
        return_mapping.sig_old.value[:] = sig_old_values[q,:].T
        return_mapping.p_old.value = p_old_values[q,:]
        
        return_mapping.solve(**conic_solver_params)

        sig_values[q,:] = return_mapping.sig.value[:].T
        p_values[q,:] = return_mapping.p.value

    if residue_size != 0: #how to improve ?
        deps_values_residue = deps[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        return_mapping_residue.deps.value[:] = deps_values_residue[0,:].T
        return_mapping_residue.sig_old.value[:] = sig_old_values_residue[0,:].T
        return_mapping_residue.p_old.value = p_old_values_residue[0,:]
        
        return_mapping_residue.solve(**conic_solver_params)

        sig_values_residue[0,:] = return_mapping_residue.sig.value[:].T
        p_values_residue[0,:] = return_mapping_residue.p.value
    return sigma_vals.reshape(-1), p_vals.reshape(-1)

global_size = int(sigma.ref_coefficient.x.array.size / 4.0)
C_elas_ = np.empty((global_size, 4, 4), dtype=PETSc.ScalarType)
for i in range(global_size):
    C_elas_[i] = C_elas

def C_tang_impl(deps):
    # num_cells, num_quadrature_points, _ = deps.shape
   
    # deps_ = deps.reshape((num_cells, num_quadrature_points, 4))
    # sigma_n_ = sigma_n.x.array.reshape((num_cells, num_quadrature_points, 4))
    # p_ = p.x.array.reshape((num_cells, num_quadrature_points))

    # C_tang_, sigma_, dp_ = return_mapping(deps_, sigma_n_, p_)

    return C_elas_.reshape(-1)

# %% [markdown]
# It is worth noting that at the time of the derivative evaluation, we compute the
# values of the external operator as well. Thus, there is no need for a separate
# implementation of the operator $\boldsymbol{\sigma}$. We will reuse the output
# of the `C_tang_impl` to update values of the external operator further in the
# Newton loop.

# %%
def sigma_external(derivatives):
    if derivatives == (0,):
        return sigma_impl
    if derivatives == (1,):
        return C_tang_impl
    else:
        return NotImplementedError


sigma.external_function = sigma_external

# %% [markdown]
# ```{note}
# The framework allows implementations of external operators and its derivatives
# to return additional outputs. In our example, alongside with the values of the
# derivative, the function `C_tang_impl` returns, the values of the stress tensor
# and the cumulative plastic increment. Both additional outputs may be reused by
# the user afterwards in the Newton loop.
# ```

# %% [markdown]
# ### Form manipulations
#
# As in the previous tutorials before solving the problem we need to perform
# some transformation of both linear and bilinear forms.

# %%
u_hat = ufl.TrialFunction(V)
J = ufl.derivative(F, Du, u_hat)
J_expanded = ufl.algorithms.expand_derivatives(J)

F_replaced, F_external_operators = replace_external_operators(F)
J_replaced, J_external_operators = replace_external_operators(J_expanded)

F_form = fem.form(F_replaced)
J_form = fem.form(J_replaced)

# %% [markdown]
# ```{note}
#  We remind that in the code above we replace `FEMExternalOperator` objects by
#  their `fem.Function` representatives, the coefficients which are allocated
#  during the call of the `FEMExternalOperator` constructor. The access to these
#  coefficients may be carried out through the field `ref_coefficient` of an
#  `FEMExternalOperator` object. For example, the following code returns the
#  finite coefficient associated with the tangent matrix
#  `C_tang = J_external_operators[0].ref_coefficient`.
# ```

# %% [markdown]
# ### Solving the problem
#
# Once we prepared the forms containing external operators, we can defind the
# nonlinear problem and its solver. Here we modified the original DOLFINx
# `NonlinearProblem` and called it `NonlinearProblemWithCallback` to let the
# solver evaluate external operators at each iteration. For this matter we define
# the function `constitutive_update` with external operators evaluations and
# update of the internal variable `dp`.

# %%
def constitutive_update():
    evaluated_operands = evaluate_operands(F_external_operators)
    ((_, p_vals),) = evaluate_external_operators(F_external_operators, evaluated_operands)
    _ = evaluate_external_operators(J_external_operators, evaluated_operands)
    # This avoids having to evaluate the external operators of F.
    # sigma.ref_coefficient.x.array[:] = sigma_new
    p.x.array[:] = p_vals


problem = PETScNonlinearProblem(Du, F_replaced, J_replaced, bcs=bcs, external_callback=constitutive_update)

petsc_options = {
    "snes_type": "qn",
    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
    "snes_qn_m": 100,
    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
    "snes_qn_restart_type": "none", #<powell,periodic,none> 
    "pc_type": "cholesky", # cholesky >> hypre > gamg,sor 
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "snes_atol": 1.0e-8,
    "snes_rtol": 1.0e-8,
    "snes_max_it": 100,
    "snes_monitor": "",
}


solver = PETScNonlinearSolver(mesh.comm, problem, petsc_options=petsc_options)  # PETSc.SNES wrapper

# %% [markdown]
# Now we are ready to solve the problem.

# %%
u = fem.Function(V, name="displacement")

x_point = np.array([[R_i, 0, 0]])
cells, points_on_process = find_cell_by_point(mesh, x_point)

q_lim = 2.0 / np.sqrt(3.0) * np.log(R_e / R_i) * sigma_0
num_increments = 20
load_steps = np.linspace(0, 1.1, num_increments, endpoint=True) ** 0.5
loadings = q_lim * load_steps
results = np.zeros((num_increments, 2))

eps = np.finfo(PETSc.ScalarType).eps

for i, loading_v in enumerate(loadings):
    if MPI.COMM_WORLD.rank == 0:
        print(f"Load increment #{i}, load: {loading_v:.3f}")

    loading.value = loading_v
    Du.x.array[:] = eps

    iters, _ = solver.solve(Du)
    print(f"\tInner Newton iterations: {iters}")

    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()

    # p.x.petsc_vec.axpy(1.0, dp.x.petsc_vec)
    p_n.x.array[:] = p.x.array
    sigma_n.x.array[:] = sigma.ref_coefficient.x.array

    if len(points_on_process) > 0:
        results[i, :] = (u.eval(points_on_process, cells)[0], loading.value / q_lim)

# %% [markdown]
# ### Post-processing
#
# In order to verify the correctness of obtained results, we perform their
# comparison against a "pure UFl" implementation. Thanks to simplicity of the von
# Mises model we can express stress tensor and tangent moduli analytically within
# the variational setting and so in UFL. Such a performant implementation is
# presented by the function `plasticity_von_mises_pure_ufl`.

# %%
results_pure_ufl = plasticity_von_mises_pure_ufl(verbose=True)

# %% [markdown]
# Here below we plot the displacement of the inner boundary of the cylinder
# $u_x(R_i, 0)$ with respect to the applied pressure in the von Mises model with
# isotropic hardening. The plastic deformations are reached at the pressure
# $q_{\lim}$ equal to the analytical collapse load for perfect plasticity.

# %%
if len(points_on_process) > 0:
    plt.plot(results_pure_ufl[:, 0], results_pure_ufl[:, 1], "o-", label="pure UFL")
    plt.plot(results[:, 0], results[:, 1], "*-", label="dolfinx-external-operator (CVXPY)")
    plt.xlabel(r"Displacement of inner boundary $u_x$ at $(R_i, 0)$ [mm]")
    plt.ylabel(r"Applied pressure $q/q_{\text{lim}}$ [-]")
    plt.legend()
    plt.grid()
    plt.savefig("output.png")
    plt.show()

# %% [markdown]
# ## References
#
# ```{bibliography}
# :filter: docname in docnames
# ```


