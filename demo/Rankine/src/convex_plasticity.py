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

import pickle
from datetime import datetime
import os, psutil
from dolfinx import common

import sys
sys.path.append("./")
from cvxpygen import cpg
import cvxpy as cp
from scipy.sparse import block_diag

# import jax
# jax.config.update("jax_enable_x64", True)
# from cvxpylayers.jax import CvxpyLayer

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes

def solve_convex_plasticity(params=None):
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

    h = params["mesh_size"]
    mesh, facet_tags, facet_tags_labels = build_cylinder_quarter(lc=h)

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
        """An implementation of return-mapping procedure via convex problems solving."""
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

            self.sig_numpy = np.empty((4, N))
            self.p_numpy = np.empty((N,))

            self.sig_old.value = np.zeros((4, N))
            self.deps.value = np.zeros((4, N))
            self.p_old.value = np.zeros((N,))
            self.C_tang = np.zeros((N, 4, 4))

            S = np.linalg.inv(material.C)
            L = np.linalg.cholesky(np.linalg.inv(material.C))
            delta_sig = self.sig - sig_elas
            elastic_energy = cp.sum_squares(L @ delta_sig)
            # target_expression = 0.5*elastic_energy + 0.5*material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)
            D = material.yield_criterion.H * np.eye(N)
            target_expression = 0.5 * elastic_energy + 0.5 * material.yield_criterion.H * cp.sum_squares(self.p - self.p_old)

            constrains = material.yield_criterion.criterion(self.sig, self.p) 

            if material.plane_stress:
                constrains.append(self.sig[2] == 0) #TODO: MODIFY!

            self.opt_problem = cp.Problem(cp.Minimize(target_expression), constrains)
            self.solver = solver

            # self.sig.value[:] = 0.0
            # self.p.value[:] = 0.0
            # TODO: Try to differentiate with cvxpylayers
            # self.cvxpylayer = CvxpyLayer(self.opt_problem, parameters=[self.deps, self.sig_old, self.p_old], variables=[self.sig, self.p])
            # self.dxxcvxpylayer = jax.grad(lambda deps, sig_old, p_old: self.cvxpylayer(deps, sig_old, p_old)[0][0], argnums=[0])
            # self.dyycvxpylayer = jax.grad(lambda deps, sig_old, p_old: self.cvxpylayer(deps, sig_old, p_old)[0][1], argnums=[0])
            # self.dzzcvxpylayer = jax.grad(lambda deps, sig_old, p_old: self.cvxpylayer(deps, sig_old, p_old)[0][2], argnums=[0])
            # self.dxycvxpylayer = jax.grad(lambda deps, sig_old, p_old: self.cvxpylayer(deps, sig_old, p_old)[0][3], argnums=[0])

        def solve(self, **kwargs):
            """Solves a minimization problem and calculates the derivative of `sig` variable.
            
            Args:
                **kwargs: additional solver attributes, such as tolerance, etc.
            """
            self.opt_problem.solve(solver=self.solver, requires_grad=False, ignore_dpp=False, warm_start=True, **kwargs)
            
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

        # def solve_and_derivate(self, **kwargs):
        #     """Solves a minimization problem and calculates the derivative of `sig` variable.
            
        #     Args:
        #         **kwargs: additional solver attributes, such as tolerance, etc.
        #     """
        #     # self.opt_problem.solve(solver=self.solver, requires_grad=False, ignore_dpp=False, **kwargs)
        #     solution = self.cvxpylayer(self.deps.value, self.sig_old.value, self.p_old.value)
        #     self.sig_numpy[:] = solution[0]
        #     self.p_numpy[:] = solution[1]

        #     result = self.dxxcvxpylayer(jax.numpy.array(self.deps.value), jax.numpy.array(self.sig_old.value), jax.numpy.array(self.p_old.value))
        #     print(result)

    rankine = Rankine(sigt, sigc, H)
    von_mises = vonMises(sigma_0, H)
    material = Material(IsotropicElasticity(E, nu), von_mises)

    patch_size = params["patch_size"]
    patch_size_max = params["patch_size_max"]
    convex_solver = params["solver"]
    convex_solver_residue = params["solver"]
    if patch_size_max:
        # patch_size = MPI.COMM_WORLD.allreduce(P.dofmap.index_map.size_local, op=MPI.MIN)
        patch_size = P.dofmap.index_map.size_local
    return_mapping = ConvexPlasticity(material, patch_size, convex_solver)
    # tol = 1.0e-13
    # scs_params = {'eps': tol, 'eps_abs': tol, 'eps_rel': tol}
    conic_solver_params = {}

    timer = common.Timer("DOLFINx_timer")

    compiled = params["compiled"]
    if compiled:
        # TODO: turn off compilation outputs
        timer.start()
        code_dir = f'code_dir_{MPI.COMM_WORLD.rank}'
        sys.path.append(code_dir)
        cpg.generate_code(return_mapping.opt_problem, code_dir=code_dir, solver=convex_solver, prefix=f'{MPI.COMM_WORLD.rank}', gradient=False)
        # TODO: Figure out when gradient=True will work
        # MPI.COMM_WORLD.barrier() # Wait until rank 0 has generated the code
        
        from code_dir.cpg_solver import cpg_solve
        convex_solver = f'CPG_{MPI.COMM_WORLD.rank}'
        return_mapping.opt_problem.register_solve(convex_solver, cpg_solve)
        
        timer.stop()
        compilation_time = timer.elapsed().total_seconds()

    stress_dim = 4
    sigma_n.x.array[:] = 0.0
    p_n.x.array[:] = 0.0
    num_quadrature_points = int(sigma_n.x.array.size / stress_dim)

    N_patches = int(num_quadrature_points / patch_size)
    residue_size = num_quadrature_points % patch_size
    # TODO: enable compilation for non-zero residue size

    if MPI.COMM_WORLD.rank == 0:
        print(f"n_quadratures_local: {P.dofmap.index_map.size_local} n_quadratures_global: {P.dofmap.index_map.size_global} n_processes: {MPI.COMM_WORLD.size} mesh_size: {h} compiled: {compiled} solver: {convex_solver} patch_size: {patch_size} N_patches: {N_patches} residue_size: {residue_size}", flush=True)

    p_vals = np.empty_like(p.x.array)
    # p_values = p.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))
    p_values = p_vals[:num_quadrature_points - residue_size].reshape((-1, patch_size))
    p_old_values = p_n.x.array[:num_quadrature_points - residue_size].reshape((-1, patch_size))

    sigma_vals = np.empty_like(sigma.ref_coefficient.x.array)
    sig_values = sigma_vals[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    sig_old_values = sigma_n.x.array[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))

    if residue_size != 0:
        return_mapping_residue = ConvexPlasticity(material, residue_size, convex_solver_residue)
        # p_values_residue = p.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
        p_values_residue = p_vals[num_quadrature_points - residue_size:].reshape((1, residue_size))
        p_old_values_residue = p_n.x.array[num_quadrature_points - residue_size:].reshape((1, residue_size))
        sig_values_residue = sigma_vals[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        sig_old_values_residue = sigma_n.x.array[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))

    def sigma_impl(deps):
        deps_ = deps.reshape(-1)
        deps_values = deps_[:4*(num_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
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

            # sig_values[q,:] = return_mapping.sig_numpy[:].T
            # p_values[q,:] = return_mapping.p_numpy

        if residue_size != 0: #how to improve ?
            deps_values_residue = deps_[4*(num_quadrature_points - residue_size):].reshape((1, residue_size, 4))
            return_mapping_residue.deps.value[:] = deps_values_residue[0,:].T
            return_mapping_residue.sig_old.value[:] = sig_old_values_residue[0,:].T
            return_mapping_residue.p_old.value = p_old_values_residue[0,:]
            
            return_mapping_residue.solve(**conic_solver_params)

            sig_values_residue[0,:] = return_mapping_residue.sig.value[:].T
            p_values_residue[0,:] = return_mapping_residue.p.value

            # sig_values[0,:] = return_mapping.sig_numpy[:].T
            # p_values[0,:] = return_mapping.p_numpy
            
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

    def sigma_external(derivatives):
        if derivatives == (0,):
            return sigma_impl
        if derivatives == (1,):
            return C_tang_impl
        else:
            return NotImplementedError


    sigma.external_function = sigma_external

    u_hat = ufl.TrialFunction(V)
    J = ufl.derivative(F, Du, u_hat)
    J_expanded = ufl.algorithms.expand_derivatives(J)

    F_replaced, F_external_operators = replace_external_operators(F)
    J_replaced, J_external_operators = replace_external_operators(J_expanded)

    F_form = fem.form(F_replaced)
    J_form = fem.form(J_replaced)

    def constitutive_update():
        evaluated_operands = evaluate_operands(F_external_operators)
        ((_, p_vals),) = evaluate_external_operators(F_external_operators, evaluated_operands)
        _ = evaluate_external_operators(J_external_operators, evaluated_operands)
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
        # "snes_monitor": "",
    }

    solver = PETScNonlinearSolver(mesh.comm, problem, petsc_options=petsc_options)  # PETSc.SNES wrapper

    timer.start()

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
            print(f"Load increment #{i}, load: {loading_v:.3f}", flush=True)

        loading.value = loading_v
        Du.x.array[:] = eps

        iters, _ = solver.solve(Du)
        if MPI.COMM_WORLD.rank == 0:
            print(f"\tInner Newton iterations: {iters}", flush=True)

        u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
        u.x.scatter_forward()

        # p.x.petsc_vec.axpy(1.0, dp.x.petsc_vec)
        p_n.x.array[:] = p.x.array
        sigma_n.x.array[:] = sigma.ref_coefficient.x.array

        if len(points_on_process) > 0:
            results[i, :] = (u.eval(points_on_process, cells)[0], loading.value / q_lim)

    timer.stop()
    total_time = timer.elapsed().total_seconds()
   
    memory_usage = MPI.COMM_WORLD.allreduce(get_memory_usage() / (2 ** 30), op=MPI.SUM)

    if MPI.COMM_WORLD.rank == 0:
        n_processes = MPI.COMM_WORLD.size
        output_data = {
            "total_time": total_time, 
            "memory_usage": memory_usage,
            "n_quadratures_local": P.dofmap.index_map.size_local,
            "n_quadratures_global": P.dofmap.index_map.size_global,
            "n_processes": n_processes,
            "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }
        if compiled:
            output_data["compilation_time"] = compilation_time
        
        os.makedirs("./output", exist_ok=True)
        filename = (
            f"output_-{convex_solver}_-{patch_size}_-{n_processes}_{h}_{compiled}.pkl"
        )
        output_data["output_file"] = os.path.join("./output", filename)
        print(output_data, flush=True)
        output_data_to_store = {**params, **output_data}
        with open(output_data["output_file"], "wb") as f:
            f.write(pickle.dumps(output_data_to_store))
