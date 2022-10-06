import convex_return_mapping as crm # there is a conflict in the oder of imported modules
import meshio
import numpy as np

import ufl
from dolfinx import fem, io, common
from mpi4py import MPI
from petsc4py import PETSc

from typing import List, Union, Dict, Optional, Callable

import time

import sys
sys.path.append("../")
import fenicsx_support as fs
import utility_functions as uf

import logging
LOG_INFO_STAR = logging.INFO + 5

SQRT2 = np.sqrt(2.)

class LinearProblem():
    def __init__(
        self,
        dR: ufl.Form,
        R: ufl.Form,
        u: fem.Function,
        bcs: List[fem.dirichletbc] = []
    ):
        self.u = u
        self.bcs = bcs

        V = u.function_space
        domain = V.mesh

        self.R = R
        self.dR = dR
        self.b_form = fem.form(R)
        self.A_form = fem.form(dR)
        self.b = fem.petsc.create_vector(self.b_form)
        self.A = fem.petsc.create_matrix(self.A_form)

        self.comm = domain.comm

        self.solver = self.solver_setup()

    def solver_setup(self) -> PETSc.KSP:
        """Sets the solver parameters."""
        solver = PETSc.KSP().create(self.comm)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(self.A)
        return solver

    def assemble_vector(self) -> None:
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b, self.b_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)

    def assemble_matrix(self) -> None:
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.A_form, bcs=self.bcs)
        self.A.assemble()

    def assemble(self) -> None:
        self.assemble_matrix()
        self.assemble_vector()
    
    def solve (
        self, 
        du: fem.function.Function, 
    ) -> None:
        """Solves the linear system and saves the solution into the vector `du`
        
        Args:
            du: A global vector to be used as a container for the solution of the linear system
        """
        self.solver.solve(self.b, du.vector)

class NonlinearProblem(LinearProblem):
    def __init__(
        self,
        dR: ufl.Form,
        R: ufl.Form,
        u: fem.Function,
        bcs: List[fem.dirichletbc] = [],
        Nitermax: int = 200, 
        tol: float = 1e-8,
        inside_Newton: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(dR, R, u, bcs)
        self.Nitermax = Nitermax
        self.tol = tol
        self.du = fem.Function(self.u.function_space)

        if inside_Newton is not None:
            self.inside_Newton = inside_Newton
        else:
            def dummy_func():
                pass
            self.inside_Newton = dummy_func
        
        if logger is not None:
            self.logger = logger 
        else:
            self.logger = logging.getLogger('nonlinear_solver')

    
    def solve(self) -> int:
        
        self.assemble()

        nRes0 = self.b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
        nRes = nRes0
        niter = 0

        # start = time.time()

        while nRes/nRes0 > self.tol and niter < self.Nitermax:
            
            self.solver.solve(self.b, self.du.vector)
            
            self.u.vector.axpy(1, self.du.vector) # u = u + 1*du
            self.u.x.scatter_forward() 

            start_return_mapping = time.time()

            self.inside_Newton()

            end_return_mapping = time.time()

            self.assemble()

            nRes = self.b.norm()

            niter += 1

            self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Increment: {niter}, norm(Res/Res0) = {nRes/nRes0:.1e}. Time (return mapping) = {end_return_mapping - start_return_mapping:.2f} (s)')
            
        
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)')
        # self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, Iterations = {niter}, Time = {time.time() - start:.2f} (s) \n')


        return niter

class SNESProblem(LinearProblem):
    """
    Problem class compatible with PETSC.SNES solvers.
    """

    def __init__(
        self,
        F_form: ufl.Form,
        u: fem.Function,
        J_form: Optional[ufl.Form] = None,
        bcs: List[fem.dirichletbc] = [],
        petsc_options: Dict[str, Union[str, int, float]] = {},
        prefix: Optional[str] = None,
        inside_Newton: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix
        self.petsc_options = petsc_options

        if J_form is None:
            V = u.function_space
            J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))
        
        super().__init__(J_form, F_form, u, bcs)

        if inside_Newton is not None:
            self.inside_Newton = inside_Newton
        else:
            def dummy_func():
                pass
            self.inside_Newton = dummy_func
        
        if logger is not None:
            self.logger = logger 
        else:
            self.logger = logging.getLogger('SNES_solver')

                
    def set_petsc_options(self) -> None:
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        
        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solver_setup(self) -> PETSc.SNES:
        # Create nonlinear solver
        snes = PETSc.SNES().create(self.comm)

        # Set options
        snes.setOptionsPrefix(self.prefix)
        self.set_petsc_options()        

        snes.setFunction(self.assemble_vector, self.b)
        snes.setJacobian(self.assemble_matrix, self.A)

        snes.setFromOptions()

        return snes

    def assemble_vector(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # x_{k+1} = x_k + dx_k, where dx_k = x ?
        x.copy(self.u.vector) 
        self.u.x.scatter_forward()

        #TODO: SNES makes the iteration #0, where he calculates the b norm. `inside_Newton()` can be omitted in that case
        self.inside_Newton()

        super().assemble_vector()
        

    def assemble_matrix(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        super().assemble_matrix()

    def solve(self) -> int:
    
        # start = time.time()

        self.solver.solve(None, self.u.vector)
    
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  {self.prefix} SNES solver converged in {self.solver.getIterationNumber()} iterations with converged reason {self.solver.getConvergedReason()})')
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)\n')

        self.u.x.scatter_forward()
        return self.solver.getIterationNumber() 

class AbstractPlasticity():
    def __init__(
        self, 
        sig0: np.float64, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
    ):
        if MPI.COMM_WORLD.rank == 0:
            # It works with the msh4 only!!
            msh = meshio.read(mesh_name)

            # Create and save one file for the mesh, and one file for the facets 
            triangle_mesh = fs.create_mesh(msh, "triangle", prune_z=True)
            line_mesh = fs.create_mesh(msh, "line", prune_z=True)
            meshio.write("thick_cylinder.xdmf", triangle_mesh)
            meshio.write("mt.xdmf", line_mesh)
            # print(msh)
        
        with io.XDMFFile(MPI.COMM_WORLD, "thick_cylinder.xdmf", "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")
            ct = xdmf.read_meshtags(self.mesh, name="Grid")

        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim - 1)

        with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
            ft = xdmf.read_meshtags(self.mesh, name="Grid")
        
        deg_u = 2
        self.deg_stress = 2

        self.V = fem.VectorFunctionSpace(self.mesh, ("CG", deg_u))
        self.u = fem.Function(self.V, name="Total_displacement")
        self.Du = fem.Function(self.V, name="Current_increment")
        self.v_ = ufl.TrialFunction(self.V)
        self.u_ = ufl.TestFunction(self.V)
        P0 = fem.FunctionSpace(self.mesh, ("DG", 0))
        self.p_avg = fem.Function(P0, name="Plastic_strain")

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')        
        self.W = fem.FunctionSpace(self.mesh, We)
        self.W0 = fem.FunctionSpace(self.mesh, W0e)

        self.sig = fem.Function(self.W)
        self.p = fem.Function(self.W0)

        left_marker = 3
        down_marker = 1
        left_facets = ft.indices[ft.values == left_marker]
        down_facets = ft.indices[ft.values == down_marker]
        left_dofs = fem.locate_dofs_topological(self.V.sub(0), self.mesh.topology.dim-1, left_facets)
        down_dofs = fem.locate_dofs_topological(self.V.sub(1), self.mesh.topology.dim-1, down_facets)

        self.bcs = [fem.dirichletbc(PETSc.ScalarType(0), left_dofs, self.V.sub(0)), fem.dirichletbc(PETSc.ScalarType(0), down_dofs, self.V.sub(1))]

        Re, Ri = 1.3, 1.   # external/internal radius
        n = ufl.FacetNormal(self.mesh)
        self.q_lim = float(2/np.sqrt(3)*np.log(Re/Ri)*sig0)
        self.loading = fem.Constant(self.mesh, PETSc.ScalarType(0.0 * self.q_lim))

        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=ft)
        self.dx = ufl.Measure(
            "dx",
            domain=self.mesh,
            metadata={"quadrature_degree": self.deg_stress, "quadrature_scheme": "default"},
        )

        # Defining a cell containing (Ri, 0) point, where we calculate a value of u
        x_point = np.array([[Ri, 0, 0]])
        self.cells, self.points_on_proc = fs.find_cell_by_point(self.mesh, x_point)

        self.problem = None
        
        def F_ext(v):
            return -self.loading * ufl.inner(n, v)*self.ds(4)

        self.F_ext = F_ext

        if logger is not None:
            self.logger = logger 
        else:
            self.logger = logging.getLogger('abstract_plasticity')

        num_nodes_global = self.mesh.topology.index_map(self.mesh.topology.dim - 2).size_global
        num_cells_global = self.mesh.topology.index_map(self.mesh.topology.dim).size_global
        num_quadrature_points_global = self.W0.dofmap.index_map.size_global
        num_nodes_local = self.mesh.topology.index_map(self.mesh.topology.dim - 2).size_local
        num_cells_local = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        num_quadrature_points_local = self.W0.dofmap.index_map.size_local

        if MPI.COMM_WORLD.rank == 0:
            self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: mesh = {mesh_name}')
            self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: mesh info: {num_nodes_global} nodes (global), {num_cells_global} cells (global), {num_quadrature_points_global} quadrature_points (Q2 space) (global)')
        self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: mesh info: {num_nodes_local} nodes (local), {num_cells_local} cells (local), {num_quadrature_points_local} quadrature_points (Q2 space) (local)')
        

    def inside_Newton(self) -> None:
        pass

    def after_Newton(self) -> None:
        pass

    def initialize_variables(self) -> None:
        pass

    def solve(self, Nincr: int = 20) -> tuple[List, np.ndarray, float, fem.function.Function, fem.function.Function, fem.function.Function]:
        load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
        results = np.zeros((Nincr+1, 2))
        # xdmf = io.XDMFFile(MPI.COMM_WORLD, "plasticity.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
        # xdmf.write_mesh(mesh)

        self.initialize_variables()

        start = time.time()

        for (i, t) in enumerate(load_steps):
            self.loading.value = t * self.q_lim

            self.Du.x.array[:] = 0

            nRes0 = self.problem.b.norm()
            self.logger.info(f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, norm(nRes0) = {nRes0:.1e}, load = {t * self.q_lim}')

            start_step = time.time()
            niters = self.problem.solve()
            end_step = time.time()

            self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, Iterations = {niters}, Time = {end_step - start_step:.2f} (s)')

            self.u.vector.axpy(1, self.Du.vector) # u = u + 1*Du
            self.u.x.scatter_forward()

            self.after_Newton()
            # fs.project(p, p_avg)
        
            # xdmf.write_function(u, t)
            # xdmf.write_function(p_avg, t)

            if len(self.points_on_proc) > 0:
                results[i+1, :] = (self.u.eval(self.points_on_proc, self.cells)[0], t)

        # xdmf.close()
        # end = time.time()
        total_time = time.time() - start
        self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Time (total) = {total_time:.2f} (s)\n')

        return self.points_on_proc, results, total_time, self.sig, self.p, self.u

class vonMisesPlasticity(AbstractPlasticity):
    def __init__(
        self, 
        material: crm.Material, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
        solver: str = "nonlinear",
    ):
        sig0 = material.yield_criterion.sig0
        mu_ = material.constitutive_law.mu_
        lambda_ = material.constitutive_law.lambda_
        H = material.yield_criterion.H

        super().__init__(sig0, mesh_name, logger)

        self.dp = fem.Function(self.W0) 

        
        self.proj_sig = lambda deps, old_sig, old_p: uf.proj_sig_vonMises(deps, old_sig, old_p, lambda_, mu_, sig0, H)
        
        if solver == 'nonlinear':
            self.sig_old = fem.Function(self.W)
            self.n_elas = fem.Function(self.W)
            self.beta = fem.Function(self.W0)
     
            sigma_tang = lambda e: uf.sigma_tang_vonMises(e, self.n_elas, self.beta, lambda_, mu_, H)

            def inside_Newton():
                deps = uf.eps(self.Du)
                sig_, n_elas_, beta_, self.dp_ = self.proj_sig(deps, self.sig_old, self.p)
                fs.interpolate_quadrature(sig_, self.sig)
                fs.interpolate_quadrature(n_elas_, self.n_elas)
                fs.interpolate_quadrature(beta_, self.beta)
            self.inside_Newton = inside_Newton

            def after_Newton():
                fs.interpolate_quadrature(self.dp_, self.dp)
                self.sig_old.x.array[:] = self.sig.x.array[:]
                self.p.vector.axpy(1, self.dp.vector)
                self.p.x.scatter_forward()
            self.after_Newton = after_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.u.vector.set(0.0)
                self.n_elas.vector.set(0.0)
                self.beta.vector.set(0.0)
            self.initialize_variables = initialize_variables 

            a_Newton = ufl.inner(uf.eps(self.v_), sigma_tang(uf.eps(self.u_)))*self.dx
            res = -ufl.inner(uf.eps(self.u_), uf.as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

            self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax=200, tol=1e-8, inside_Newton=self.inside_Newton, logger=self.logger)

        elif solver == 'SNES' or solver == 'SNESQN':
            deps_p = lambda deps, old_sig, old_p: uf.deps_p_vonMises(deps, old_sig, old_p, lambda_, mu_, sig0, H)
            sigma = lambda eps: uf.sigma(eps, lambda_, mu_)
            deps = uf.eps(self.Du)

            def after_Newton():
                sig_, _, _, dp_ = self.proj_sig(deps, self.sig, self.p)
                fs.interpolate_quadrature(sig_, self.sig)
                fs.interpolate_quadrature(dp_, self.dp)
                self.p.vector.axpy(1, self.dp.vector)
                self.p.x.scatter_forward()
            self.after_Newton = after_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.p.vector.set(0.0)
                self.u.vector.set(0.0)
            self.initialize_variables = initialize_variables 

            residual = ufl.inner(uf.as_3D_tensor(self.sig) + sigma(deps - deps_p(deps, self.sig, self.p)), uf.eps(self.u_))*self.dx - self.F_ext(self.u_)
            J = ufl.derivative(ufl.inner(sigma(deps - deps_p(deps, self.sig, self.p)), uf.eps(self.u_))*self.dx, self.Du, self.v_)

            petsc_options = {}
            if solver == 'SNES':
                petsc_options = {
                    "snes_type": "vinewtonrsls",
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": 1.0e-08,
                    "snes_rtol": 1.0e-08,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    "snes_monitor_cancel": "",
                }
            else:
                petsc_options = {
                    "snes_type": "qn",
                    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
                    "snes_qn_m": 100,
                    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
                    "snes_qn_restart_type": "none", #<powell,periodic,none> 
                    "pc_type": "cholesky", # cholesky >> hypre > gamg,sor ; asm, lu, gas - don't work
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": 1.0e-08,
                    "snes_rtol": 1.0e-08,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    # "snes_monitor": "",
                    "snes_monitor_cancel": "",
                }

            self.problem = SNESProblem(residual, self.Du, J_form=J, bcs=self.bcs, petsc_options=petsc_options, inside_Newton=None, logger=self.logger)
        else:
            raise RuntimeError(f"Solver {solver} doesn't support!")

class DruckerPragerPlasticity(AbstractPlasticity):
    def __init__(
        self, 
        material: crm.Material, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
        solver: str = "nonlinear",
    ):
        sig0 = material.yield_criterion.sig0
        mu_ = material.constitutive_law.mu_
        lambda_ = material.constitutive_law.lambda_
        k = lambda_ + 2*mu_/3 # Bulk modulus 
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        alpha = material.yield_criterion.alpha
        M = (3*mu_ + 9*alpha*alpha*k) + H

        super().__init__(sig0, mesh_name, logger)

        self.dp = fem.Function(self.W0) 

        self.proj_sig = lambda deps, old_sig, old_p: uf.proj_sig(deps, old_sig, old_p, lambda_, mu_, sig0, H, alpha, M, k)
        
        if solver == 'nonlinear':
            self.sig_old = fem.Function(self.W)
            self.n_elas = fem.Function(self.W)
            self.beta = fem.Function(self.W0)
     
            sigma_tang = lambda e: uf.sigma_tang(e, self.n_elas, self.beta, lambda_, mu_, H, alpha, M, k)

            def inside_Newton():
                deps = uf.eps(self.Du)
                sig_, n_elas_, beta_, self.dp_ = self.proj_sig(deps, self.sig_old, self.p)
                fs.interpolate_quadrature(sig_, self.sig)
                fs.interpolate_quadrature(n_elas_, self.n_elas)
                fs.interpolate_quadrature(beta_, self.beta)
            self.inside_Newton = inside_Newton

            def after_Newton():
                fs.interpolate_quadrature(self.dp_, self.dp)
                self.sig_old.x.array[:] = self.sig.x.array[:]
                self.p.vector.axpy(1, self.dp.vector)
                self.p.x.scatter_forward()
            self.after_Newton = after_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.u.vector.set(0.0)
                self.n_elas.vector.set(0.0)
                self.beta.vector.set(0.0)
            self.initialize_variables = initialize_variables 

            a_Newton = ufl.inner(uf.eps(self.v_), sigma_tang(uf.eps(self.u_)))*self.dx
            res = -ufl.inner(uf.eps(self.u_), uf.as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

            self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax=200, tol=1e-8, inside_Newton=self.inside_Newton, logger=self.logger)

        elif solver == 'SNES' or solver == 'SNESQN':
            deps_p = lambda deps, old_sig, old_p: uf.deps_p(deps, old_sig, old_p, lambda_, mu_, sig0, H, alpha, M)
            sigma = lambda eps: uf.sigma(eps, lambda_, mu_)
            deps = uf.eps(self.Du)

            def after_Newton():
                sig_, _, _, dp_ = self.proj_sig(deps, self.sig, self.p)
                fs.interpolate_quadrature(sig_, self.sig)
                fs.interpolate_quadrature(dp_, self.dp)
                self.p.vector.axpy(1, self.dp.vector)
                self.p.x.scatter_forward()
            self.after_Newton = after_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.p.vector.set(0.0)
                self.u.vector.set(0.0)
            self.initialize_variables = initialize_variables 

            residual = ufl.inner(uf.as_3D_tensor(self.sig) + sigma(deps - deps_p(deps, self.sig, self.p)), uf.eps(self.u_))*self.dx - self.F_ext(self.u_)
            J = ufl.derivative(ufl.inner(sigma(deps - deps_p(deps, self.sig, self.p)), uf.eps(self.u_))*self.dx, self.Du, self.v_)

            petsc_options = {}
            if solver == 'SNES':
                petsc_options = {
                    "snes_type": "vinewtonrsls",
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": 1.0e-08,
                    "snes_rtol": 1.0e-08,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    "snes_monitor_cancel": "",
                }
            else:
                petsc_options = {
                    "snes_type": "qn",
                    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
                    "snes_qn_m": 100,
                    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
                    "snes_qn_restart_type": "none", #<powell,periodic,none> 
                    "pc_type": "cholesky", # cholesky >> hypre > gamg,sor ; asm, lu, gas - don't work
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": 1.0e-08,
                    "snes_rtol": 1.0e-08,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    # "snes_monitor": "",
                    "snes_monitor_cancel": "",
                }

            self.problem = SNESProblem(residual, self.Du, J_form=J, bcs=self.bcs, petsc_options=petsc_options, inside_Newton=None, logger=self.logger)
        else:
            raise RuntimeError(f"Solver {solver} doesn't support!")

class ConvexPlasticity(AbstractPlasticity):
    def __init__(
        self, 
        material: crm.Material, 
        patch_size: int = 1, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
        solver: str = "nonlinear",
        conic_solver: str = 'SCS',
        tol_Newton: float = 1e-8,
        tol_conic_solver: float = 1e-13
    ):
        if isinstance(material.yield_criterion, crm.DruckerPrager) or isinstance(material.yield_criterion, crm.vonMises):
            sig0 = material.yield_criterion.sig0
        elif isinstance(material.yield_criterion, crm.Rankine):
            sig0 = material.yield_criterion.ft
        else:
            raise RuntimeError(f"Convex plasticity doesn't support this material, chose another one.")

        mu_ = material.constitutive_law.mu_
        lambda_ = material.constitutive_law.lambda_
        
        super().__init__(sig0, mesh_name, logger)

        if MPI.COMM_WORLD.rank == 0:
            self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Newton solver = {solver}, conic solver = {conic_solver}, patch size = {patch_size}')

        self.sig_old = fem.Function(self.W)
        self.p_old = fem.Function(self.W0) 
        self.deps = fem.Function(self.W, name="deps")

        # tol_convex = 1e-13
        conic_solver_params = {}
        if conic_solver == 'SCS':
            conic_solver_params = {
                'eps': tol_conic_solver, 
                'eps_abs': tol_conic_solver, 
                'eps_rel': tol_conic_solver,
                'eps_infeas': 1e-12,
            }
        elif conic_solver == 'MOSEK':
            conic_solver_params = {'mosek_params': {
                'MSK_DPAR_BASIS_TOL_X': 1.0e-9, 
                'MSK_DPAR_BASIS_TOL_S': 1.0e-9, 
                'MSK_DPAR_INTPNT_TOL_DFEAS': 1.0e-9, 
                'MSK_DPAR_INTPNT_TOL_PFEAS': 1.0e-9,
                'MSK_DPAR_INTPNT_TOL_REL_GAP': 1.0e-9,
                'MSK_DPAR_BASIS_REL_TOL_S': 1.0e-10
            }}
        elif conic_solver == 'ECOS':
            conic_solver_params = {'abstol': tol_conic_solver, 'reltol': tol_conic_solver}
        else:
            raise RuntimeError(f"Convex plasticity supports SCS, MOSEK and ECOS conic solvers.")
        
        deps_Voigt = uf.eps_Voigt(self.Du)
        sigma = lambda eps: uf.sigma(eps, lambda_, mu_)
        
        self.n_quadrature_points = len(self.p.x.array)
        self.N_patches = int(self.n_quadrature_points / patch_size)
        self.residue_size = self.n_quadrature_points % patch_size

        self.return_mapping = crm.ReturnMapping(material, patch_size, conic_solver)
        self.material = material

        self.p_values = self.p.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.p_old_values = self.p_old.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.deps_values = self.deps.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_values = self.sig.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_old_values = self.sig_old.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))

        if self.residue_size != 0:
            self.return_mapping_residue = crm.ReturnMapping(material, self.residue_size, conic_solver)
            
            self.p_values_residue = self.p.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.p_old_values_residue = self.p_old.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.deps_values_residue = self.deps.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_values_residue = self.sig.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_old_values_residue = self.sig_old.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))

        if solver == 'nonlinear':
            def inside_Newton():
                start_return_mapping = time.time()

                fs.interpolate_quadrature(deps_Voigt, self.deps) # eps_xy * sqrt(2.)!
                for q in range(self.N_patches):
                    self.return_mapping.deps.value[:] = self.deps_values[q,:].T
                    self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
                    self.return_mapping.p_old.value = self.p_old_values[q,:]
                    
                    self.return_mapping.solve_and_derivate(verbose=False, **conic_solver_params)

                    self.sig_values[q,:] = self.return_mapping.sig.value[:].T
                    self.p_values[q,:] = self.return_mapping.p.value
                    self.C_tang_values[q,:] = self.return_mapping.C_tang[:]

                if self.residue_size != 0: #how to improve ?
                    self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
                    self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
                    self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
                    
                    self.return_mapping_residue.solve_and_derivate(verbose=False, **conic_solver_params)
                    
                    self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
                    self.p_values_residue[0,:] = self.return_mapping_residue.p.value
                    self.C_tang_values_residue[0,:] = self.return_mapping_residue.C_tang[:]

                self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {time.time() - start_return_mapping:.2f} (s)')
                
            self.inside_Newton = inside_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.p_old.vector.set(0.0)
                self.u.vector.set(0.0)
                for i in range(self.n_quadrature_points):
                    self.C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = self.material.C
            self.initialize_variables = initialize_variables

            WTe = ufl.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, shape=(4, 4), quad_scheme='default')
            WT = fem.FunctionSpace(self.mesh, WTe)
            self.C_tang = fem.Function(WT)
            self.C_tang_values = self.C_tang.x.array[:4*4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4, 4))
            if self.residue_size != 0:
                self.C_tang_values_residue = self.C_tang.x.array[4*4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4, 4))

            a_Newton = ufl.inner(uf.eps_Voigt(self.v_), ufl.dot(self.C_tang, uf.eps_Voigt(self.u_)))*self.dx 
            res = -ufl.inner(uf.eps(self.u_), uf.as_3D_tensor_Voigt(self.sig))*self.dx + self.F_ext(self.u_)
            self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax=200, tol=tol_Newton, inside_Newton=self.inside_Newton, logger=self.logger)

        elif solver == 'SNESQN':
            def inside_Newton():
                start_return_mapping = time.time()
                
                fs.interpolate_quadrature(deps_Voigt, self.deps) # eps_xy * sqrt(2.)!
                for q in range(self.N_patches):
                    self.return_mapping.deps.value[:] = self.deps_values[q,:].T
                    self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
                    self.return_mapping.p_old.value = self.p_old_values[q,:]
                    
                    self.return_mapping.solve(verbose=False, **conic_solver_params)
                    self.sig_values[q,:] = self.return_mapping.sig.value[:].T
                    self.p_values[q,:] = self.return_mapping.p.value

                if self.residue_size != 0: #how to improve ?
                    self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
                    self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
                    self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
                    
                    self.return_mapping_residue.solve(verbose=False, **conic_solver_params)
                    self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
                    self.p_values_residue[0,:] = self.return_mapping_residue.p.value

                self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {time.time() - start_return_mapping:.2f} (s)')
                
            self.inside_Newton = inside_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.p_old.vector.set(0.0)
                self.u.vector.set(0.0)
            self.initialize_variables = initialize_variables

            residual = ufl.inner(uf.as_3D_tensor_Voigt(self.sig), uf.eps(self.u_))*self.dx - self.F_ext(self.u_)
            J = ufl.derivative(ufl.inner(sigma(uf.eps(self.Du)), uf.eps(self.u_))*self.dx, self.Du, self.v_)

            petsc_options = {
                "snes_type": "qn",
                "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
                "snes_qn_m": 100,
                "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
                "snes_qn_restart_type": "none", #<powell,periodic,none> 
                "pc_type": "cholesky", # cholesky >> hypre > gamg, sor ; asm, lu, gas - don't work
                "snes_linesearch_type": "basic",
                "ksp_type": "preonly",
                "pc_factor_mat_solver_type": "mumps",
                "snes_atol": tol_Newton,
                "snes_rtol": tol_Newton,
                "snes_stol": 0.0,
                "snes_max_it": 500,
                # "snes_monitor": "",
                "snes_monitor_cancel": "",
            }
            self.problem = SNESProblem(residual, self.Du, J, self.bcs, petsc_options=petsc_options, inside_Newton=self.inside_Newton, logger=self.logger)
        else:
            raise RuntimeError(f"Solver {solver} doesn't support!")

    def after_Newton(self):
        self.p_old.x.array[:] = self.p.x.array        
        self.sig_old.x.array[:] = self.sig.x.array

class ConvexAnalyticalPlasticity(AbstractPlasticity):
    def __init__(
        self, 
        material: crm.Material, 
        patch_size: int = 1, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
        solver: str = "nonlinear",
        conic_solver: str = "SCS",
        tol_Newton: float = 1e-8,
        tol_conic_solver: float = 1e-13
    ):
        mu_ = material.constitutive_law.mu_
        lambda_ = material.constitutive_law.lambda_
        H = material.yield_criterion.H

        deps_p = None
        if isinstance(material.yield_criterion, crm.DruckerPrager):
            alpha = material.yield_criterion.alpha
            k = lambda_ + 2*mu_/3 # Bulk modulus 
            sig0 = material.yield_criterion.sig0
            M = (3*mu_ + 9*alpha*alpha*k) + H
            
            deps_p = lambda deps, old_sig, old_p: uf.deps_p_convex(deps, old_sig, old_p, lambda_, mu_, sig0, H, alpha, M)
            # deps_p = lambda deps, old_sig, old_p: uf.deps_p_vonMises(deps, old_sig, old_p, lambda_, mu_, sig0, H)

        # elif isinstance(material.yield_criterion, crm.vonMises):
        #     print('vonMises')
        #     sig0 = material.yield_criterion.sig0
        #     def deps_p_vonMises(deps, old_sig, old_p):
        #         sig_n = uf.as_3D_tensor_Voigt(old_sig)
        #         sig_elas = sig_n + sigma(deps)
        #         s = ufl.dev(sig_elas)
        #         sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        #         f_elas = sig_eq - sig0 - H*old_p
        #         deps_p = ufl.conditional(f_elas > 0, 3./2. * f_elas/(3*mu_+H) * s/sig_eq , 0*ufl.Identity(3))  # sig_eq is equal to 0 on the first iteration
        #         # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
        #         return deps_p
        #     deps_p = deps_p_vonMises

        else:
            raise RuntimeError(f"Convex plasticity supports Drucker-Prager materials, chose another one.")

        super().__init__(sig0, mesh_name, logger)
    
        if MPI.COMM_WORLD.rank == 0:
            self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Newton solver = {solver}, conic solver = {conic_solver}, patch size = {patch_size}')
        
        self.sig_old = fem.Function(self.W)
        self.p_old = fem.Function(self.W0) 
        self.deps = fem.Function(self.W, name="deps")

        conic_solver_params = {}
        if conic_solver == 'SCS':
            conic_solver_params = {'eps': tol_conic_solver, 'eps_abs': tol_conic_solver, 'eps_rel': tol_conic_solver}
        elif conic_solver == 'MOSEK':
            conic_solver_params = {'mosek_params': {
                'MSK_DPAR_BASIS_TOL_X': 1.0e-9, 
                'MSK_DPAR_BASIS_TOL_S': 1.0e-9, 
                'MSK_DPAR_INTPNT_TOL_DFEAS': 1.0e-9, 
                'MSK_DPAR_INTPNT_TOL_PFEAS': 1.0e-9,
                'MSK_DPAR_INTPNT_TOL_REL_GAP': 1.0e-9,
                'MSK_DPAR_BASIS_REL_TOL_S': 1.0e-10
            }}
        elif conic_solver == 'ECOS':
            conic_solver_params = {'abstol': tol_conic_solver, 'reltol': tol_conic_solver}
        else:
            raise RuntimeError(f"Convex plasticity supports SCS, MOSEK and ECOS conic solvers.")

        deps_Voigt = uf.eps_Voigt(self.Du)
        sigma = lambda eps: uf.sigma(eps, lambda_, mu_)

        self.n_quadrature_points = len(self.p.x.array)
        self.N_patches = int(self.n_quadrature_points / patch_size)
        self.residue_size = self.n_quadrature_points % patch_size

        self.return_mapping = crm.ReturnMapping(material, patch_size, conic_solver)
        self.material = material

        self.p_values = self.p.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.p_old_values = self.p_old.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.deps_values = self.deps.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_values = self.sig.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_old_values = self.sig_old.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))

        if self.residue_size != 0:
            self.return_mapping_residue = crm.ReturnMapping(material, self.residue_size, conic_solver)
            
            self.p_values_residue = self.p.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.p_old_values_residue = self.p_old.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.deps_values_residue = self.deps.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_values_residue = self.sig.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_old_values_residue = self.sig_old.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))

        if solver == 'nonlinear':
            def inside_Newton():
                start_return_mapping = time.time()

                fs.interpolate_quadrature(deps_Voigt, self.deps) # eps_xy * sqrt(2.)!
                for q in range(self.N_patches):
                    self.return_mapping.deps.value[:] = self.deps_values[q,:].T
                    self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
                    self.return_mapping.p_old.value = self.p_old_values[q,:]
                    
                    self.return_mapping.solve_and_derivate(verbose=False, **conic_solver_params)

                    self.sig_values[q,:] = self.return_mapping.sig.value[:].T
                    self.p_values[q,:] = self.return_mapping.p.value
                    self.C_tang_values[q,:] = self.return_mapping.C_tang[:]

                if self.residue_size != 0: #how to improve ?
                    self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
                    self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
                    self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
                    
                    self.return_mapping_residue.solve_and_derivate(verbose=False, **conic_solver_params)
                    
                    self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
                    self.p_values_residue[0,:] = self.return_mapping_residue.p.value
                    self.C_tang_values_residue[0,:] = self.return_mapping_residue.C_tang[:]

                self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {time.time() - start_return_mapping:.2f} (s)')
                
            self.inside_Newton = inside_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.p_old.vector.set(0.0)
                self.u.vector.set(0.0)
                for i in range(self.n_quadrature_points):
                    self.C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = self.material.C
            self.initialize_variables = initialize_variables

            WTe = ufl.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, shape=(4, 4), quad_scheme='default')
            WT = fem.FunctionSpace(self.mesh, WTe)
            self.C_tang = fem.Function(WT)
            self.C_tang_values = self.C_tang.x.array[:4*4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4, 4))
            if self.residue_size != 0:
                self.C_tang_values_residue = self.C_tang.x.array[4*4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4, 4))

            a_Newton = ufl.inner(uf.eps_Voigt(self.v_), ufl.dot(self.C_tang, uf.eps_Voigt(self.u_)))*self.dx 
            res = -ufl.inner(uf.eps(self.u_), uf.as_3D_tensor_Voigt(self.sig))*self.dx + self.F_ext(self.u_)
            self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax=200, tol=1e-8, inside_Newton=self.inside_Newton, logger=self.logger)

        elif solver == 'SNES' or solver == 'SNESQN':
            def inside_Newton():
                start_return_mapping = time.time()

                fs.interpolate_quadrature(deps_Voigt, self.deps) # eps_xy * sqrt(2.)!

                for q in range(self.N_patches):
                    self.return_mapping.deps.value[:] = self.deps_values[q,:].T
                    self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
                    self.return_mapping.p_old.value = self.p_old_values[q,:]
                    
                    self.return_mapping.solve(verbose=False, **conic_solver_params)
                    self.sig_values[q,:] = self.return_mapping.sig.value[:].T
                    self.p_values[q,:] = self.return_mapping.p.value

                if self.residue_size != 0: #how to improve ?
                    self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
                    self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
                    self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
                    
                    self.return_mapping_residue.solve(verbose=False, **conic_solver_params)
                    self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
                    self.p_values_residue[0,:] = self.return_mapping_residue.p.value

                self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {time.time() - start_return_mapping:.2f} (s)')
                
            self.inside_Newton = inside_Newton

            def initialize_variables():
                self.sig.vector.set(0.0)
                self.sig_old.vector.set(0.0)
                self.p.vector.set(0.0)
                self.p_old.vector.set(0.0)
                self.u.vector.set(0.0)
            self.initialize_variables = initialize_variables

            residual = ufl.inner(uf.as_3D_tensor_Voigt(self.sig_old) + sigma(uf.eps(self.Du) - deps_p(uf.eps(self.Du), self.sig_old, self.p_old)), uf.eps(self.u_))*self.dx - self.F_ext(self.u_)
            J = ufl.derivative(ufl.inner(sigma(uf.eps(self.Du)), uf.eps(self.u_))*self.dx, self.Du, self.v_)

            J = ufl.derivative(ufl.inner(sigma(uf.eps(self.Du) - deps_p(uf.eps(self.Du), self.sig_old, self.p_old)), uf.eps(self.u_))*self.dx, self.Du, self.v_)
            # J = ufl.derivative(ufl.inner(sigma(uf.eps(self.Du) - deps_p(uf.eps(self.Du), self.sig_old, self.p_old)), uf.eps(self.u_))*self.dx, self.Du, self.v_)

            petsc_options = {}
            if solver == 'SNES':
                petsc_options = {
                    "snes_type": "vinewtonrsls",
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": tol_Newton,
                    "snes_rtol": tol_Newton,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    # "snes_monitor": "",
                    "snes_monitor_cancel": "",
                }
            else:
                petsc_options = {
                    "snes_type": "qn",
                    "snes_qn_type": "lbfgs", #lbfgs broyden, badbroyden
                    "snes_qn_m": 100,
                    "snes_qn_scale_type": "jacobian", #<diagonal,none,scalar,jacobian> 	
                    "snes_qn_restart_type": "none", #<powell,periodic,none> 
                    "pc_type": "cholesky", # cholesky >> hypre > gamg, sor ; asm, lu, gas - don't work
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_factor_mat_solver_type": "mumps",
                    "snes_atol": tol_Newton,
                    "snes_rtol": tol_Newton,
                    "snes_stol": 0.0,
                    "snes_max_it": 500,
                    # "snes_monitor": "",
                    "snes_monitor_cancel": "",
                }
            self.problem = SNESProblem(residual, self.Du, J, self.bcs, petsc_options=petsc_options, inside_Newton=self.inside_Newton, logger=self.logger)
        else:
            raise RuntimeError(f"Solver {solver} doesn't support!")

    def after_Newton(self):
        self.p_old.x.array[:] = self.p.x.array        
        self.sig_old.x.array[:] = self.sig.x.array
