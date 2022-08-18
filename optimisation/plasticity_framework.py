
# from ctypes import Union
# from yaml import load
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
    ):
        super().__init__(dR, R, u, bcs)
        self.Nitermax = Nitermax
        self.tol = tol
        self.du = fem.Function(self.u.function_space)
    
    def solve(self, inside_Newton: Callable, logger: logging.Logger) -> None:
        
        self.assemble_vector()

        nRes0 = self.b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
        nRes = nRes0
        niter = 0

        start = time.time()

        while nRes/nRes0 > self.tol and niter < self.Nitermax:
            
            self.solver.solve(self.b, self.du.vector)
            
            self.u.vector.axpy(1, self.du.vector) # u = u + 1*du
            self.u.x.scatter_forward() 

            start_return_mapping = time.time()

            inside_Newton()

            end_return_mapping = time.time()

            self.assemble()

            nRes = self.b.norm()

            niter += 1

            logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Increment: {niter}, norm(Res/Res0) = {nRes/nRes0:.1e}. Time (return mapping) = {end_return_mapping - start_return_mapping:.2f} (s)')
        
        logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)\n')

class SNESProblem(LinearProblem):
    """
    Problem class compatible with PETSC.SNES solvers.
    """

    def __init__(
        self,
        F_form: ufl.Form,
        u: fem.Function,
        inside_Newton: Callable,
        bcs: List[fem.dirichletbc] = [],
        J_form: Optional[ufl.Form] = None,
        petsc_options: Dict[str, Union[str, int, float]] = {},
        prefix: Optional[str] = None,
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

        self.inside_Newton = inside_Newton
        self.du = fem.Function(self.u.function_space)
                
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

        # self.du.set(0.0)

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # x_{k+1} = x_k + dx_k, where dx_k = x ?
        x.copy(self.u.vector) 
        self.u.x.scatter_forward()

        # self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # self.u.vector.axpy(1, self.du.vector) # u = u + 1*x
        # self.u.x.scatter_forward()

        print('x norm', x.norm())
        print('b norm', self.b.norm())

        super().assemble_vector()

        self.inside_Newton()

        print('b norm', self.b.norm())

        super().assemble_vector()

        # super().assemble_matrix()

        print('b norm', self.b.norm())



    def assemble_matrix(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """

        # self.inside_Newton()

        # print('A norm', self.A.norm()) #error!

        super().assemble_matrix()

        print('A norm', self.A.norm())

    def solve(self, inside_Newton: Callable, logger: logging.Logger) -> None:
    
        start = time.time()

        self.solver.solve(None, self.u.vector)
    
        logger.debug(f'rank#{MPI.COMM_WORLD.rank}  {self.prefix} SNES solver converged in {self.solver.getIterationNumber()} iterations with converged reason {self.solver.getConvergedReason()})')
        logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)\n')

        self.u.x.scatter_forward()

        # self.inside_Newton()
    

class AbstractPlasticity():
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh", logger: logging.Logger = logging.getLogger()):
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

        left_marker = 3
        down_marker = 1
        left_facets = ft.indices[ft.values == left_marker]
        down_facets = ft.indices[ft.values == down_marker]
        left_dofs = fem.locate_dofs_topological(self.V.sub(0), self.mesh.topology.dim-1, left_facets)
        down_dofs = fem.locate_dofs_topological(self.V.sub(1), self.mesh.topology.dim-1, down_facets)

        self.bcs = [fem.dirichletbc(PETSc.ScalarType(0), left_dofs, self.V.sub(0)), fem.dirichletbc(PETSc.ScalarType(0), down_dofs, self.V.sub(1))]

        Re, Ri = 1.3, 1.   # external/internal radius
        n = ufl.FacetNormal(self.mesh)
        self.q_lim = float(2/np.sqrt(3)*np.log(Re/Ri)*material.yield_criterion.sig0)
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
        
        self.logger = logger

    def inside_Newton(self):
        pass

    def after_Newton(self):
        pass

    def initialize_variables(self):
        pass

    def eps(self, v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0],
                              [e[0, 1], e[1, 1], 0],
                              [0, 0, 0]])

    def solve(self, Nincr: int = 20):
        load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
        results = np.zeros((Nincr+1, 2))
        load_steps = load_steps
        # xdmf = io.XDMFFile(MPI.COMM_WORLD, "plasticity.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
        # xdmf.write_mesh(mesh)

        self.initialize_variables()
        self.problem.assemble_matrix()

        return_mapping_times = np.zeros((len(load_steps)))

        start = time.time()

        for (i, t) in enumerate(load_steps):
            return_mapping_times_tmp = []
            self.loading.value = t * self.q_lim

            self.Du.x.array[:] = 0

            # if MPI.COMM_WORLD.rank == 0:
            #     print(f"\nnRes0 , {nRes0} \n Increment: {str(i+1)}, load = {t * self.q_lim}")
            nRes0 = self.problem.b.norm()
            self.logger.info(f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, norm(nRes0) = {nRes0:.1e}, load = {t * self.q_lim}')

            self.problem.solve(self.inside_Newton, self.logger)

            self.u.vector.axpy(1, self.Du.vector) # u = u + 1*Du
            self.u.x.scatter_forward()

            self.after_Newton()
            # fs.project(p, p_avg)
        
            # xdmf.write_function(u, t)
            # xdmf.write_function(p_avg, t)

            # return_mapping_times[i] = np.mean(return_mapping_times_tmp)
            # print(f'rank#{MPI.COMM_WORLD.rank}: Time (mean return mapping) = {return_mapping_times[i]:.3f} (s)')

            if len(self.points_on_proc) > 0:
                results[i+1, :] = (self.u.eval(self.points_on_proc, self.cells)[0], t)

        # xdmf.close()
        # end = time.time()
        # self.logger.info(f'\n rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {np.mean(return_mapping_times):.3f} (s)')
        self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Time (total) = {time.time() - start:.2f} (s)')

        return self.points_on_proc, results

class vonMisesPlasticity(AbstractPlasticity):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh", logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')
        
        W = fem.FunctionSpace(self.mesh, We)
        W0 = fem.FunctionSpace(self.mesh, W0e)

        self.sig = fem.Function(W)
        self.sig_old = fem.Function(W)
        self.p = fem.Function(W0, name="Cumulative_plastic_strain")
        self.dp = fem.Function(W0) 
        self.n_elas = fem.Function(W)
        self.beta = fem.Function(W0)

        mu = material.constitutive_law.mu_
        lmbda = material.constitutive_law.lambda_
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        
        def sigma(eps_el):
            return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3], 0],
                                  [X[3], X[1], 0],
                                  [0, 0, X[2]]])

        ppos = lambda x: (x + ufl.sqrt(x**2))/2.
        
        def proj_sig(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            f_elas = sig_eq - sig0 - H*old_p
            dp = ppos(f_elas)/(3*mu+H)
            n_elas = s/sig_eq*ppos(f_elas)/f_elas
            beta = 3*mu*dp/sig_eq
            new_sig = sig_elas-beta*s
            return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
                ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
                beta, dp       

        self.proj_sig = proj_sig

        def sigma_tang(e):
            N_elas = as_3D_tensor(self.n_elas)
            return sigma(e) - 3*mu*(3*mu/(3*mu+H)-self.beta)*ufl.inner(N_elas, e)*N_elas - 2*mu*self.beta*ufl.dev(e) 

        a_Newton = ufl.inner(self.eps(self.v_), sigma_tang(self.eps(self.u_)))*self.dx
        res = -ufl.inner(self.eps(self.u_), as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

        self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax = 200, tol = 1e-8)

    def inside_Newton(self):
        deps = self.eps(self.Du)
        sig_, n_elas_, beta_, self.dp_ = self.proj_sig(deps, self.sig_old, self.p)

        fs.interpolate_quadrature(sig_, self.sig)
        fs.interpolate_quadrature(n_elas_, self.n_elas)
        fs.interpolate_quadrature(beta_, self.beta)

    def after_Newton(self):
        fs.interpolate_quadrature(self.dp_, self.dp)
        self.sig_old.x.array[:] = self.sig.x.array[:]
        self.p.vector.axpy(1, self.dp.vector)
        self.p.x.scatter_forward()
        # print('p after copy', np.max(p.x.array), np.min(p.x.array), p.vector.norm())

    def initialize_variables(self):
        self.sig.vector.set(0.0)
        self.sig_old.vector.set(0.0)
        self.p.vector.set(0.0)
        self.u.vector.set(0.0)
        self.n_elas.vector.set(0.0)
        self.beta.vector.set(0.0)

class DruckerPragerPlasticity(vonMisesPlasticity):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh", logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)
        
        mu = material.constitutive_law.mu_
        lmbda = material.constitutive_law.lambda_
        k = lmbda + 2*mu/3 # Bulk modulus 
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        alpha = material.yield_criterion.alpha
        M = (3*mu + 9*alpha*alpha*k) + H
        TPV = np.finfo(PETSc.ScalarType).eps # très petite value 

        def sigma(eps_el):
            return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3], 0],
                                [X[3], X[1], 0],
                                [0, 0, X[2]]])

        ppos = lambda x: (x + ufl.sqrt(x**2))/2.
        def proj_sig(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            tr_sig = ufl.tr(sig_elas)
            f_elas = sig_eq + alpha*tr_sig - sig0 - H*old_p
            dp = ppos(f_elas)/M
            n_elas = s/sig_eq*ppos(f_elas)/f_elas
            beta = 3*mu*dp/sig_eq 
            new_sig = sig_elas - (beta*s + dp * alpha*3*k * ufl.Identity(3))
            return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
                ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
                beta, dp       

        self.proj_sig = proj_sig

        def sigma_tang(e):
            N_elas = as_3D_tensor(self.n_elas)
            return sigma(e) - ( 3*mu*(3*mu/M - self.beta) * ufl.inner(N_elas, e) * N_elas + 9*mu*k*alpha/M * ( ufl.inner(N_elas, e) * ufl.Identity(3) + ufl.tr(e) * N_elas ) + ( 9*k*k*alpha*alpha/M * ufl.tr(e) * ufl.Identity(3)) * self.beta / (self.beta + TPV) + 2*mu*self.beta*ufl.dev(e) )

        a_Newton = ufl.inner(self.eps(self.v_), sigma_tang(self.eps(self.u_)))*self.dx
        res = -ufl.inner(self.eps(self.u_), as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

        self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs, Nitermax = 200, tol = 1e-8)

class ConvexPlasticity(AbstractPlasticity):
    def __init__(self, material: crm.Material, patch_size: int = 1, mesh_name: str = "thick_cylinder.msh", logger: logging.Logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')
        WTe = ufl.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, shape=(4, 4), quad_scheme='default')
        
        W = fem.FunctionSpace(self.mesh, We)
        W0 = fem.FunctionSpace(self.mesh, W0e)
        WT = fem.FunctionSpace(self.mesh, WTe)

        self.sig = fem.Function(W)
        self.sig_old = fem.Function(W)
        self.p = fem.Function(W0, name="Cumulative_plastic_strain")
        self.p_old = fem.Function(W0) 

        self.C_tang = fem.Function(WT)
        self.deps = fem.Function(W, name="deps")
        
        def eps_vec(v):
            e = ufl.sym(ufl.grad(v))
            return ufl.as_vector([e[0, 0], e[1, 1], 0, SQRT2 * e[0, 1]])

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3] / SQRT2, 0],
                                [X[3] / SQRT2, X[1], 0],
                                [0, 0, X[2]]])       

        self.eps_vec = eps_vec

        self.n_quadrature_points = len(self.C_tang.x.array.reshape((-1, 4, 4)))
        for i in range(self.n_quadrature_points):
            self.C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = material.C

        self.N_patches = int(self.n_quadrature_points / patch_size)
        self.residue_size = self.n_quadrature_points % patch_size

        self.return_mapping = crm.ReturnMapping(material, patch_size)
        self.material = material

        self.p_values = self.p.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.p_old_values = self.p_old.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.deps_values = self.deps.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_values = self.sig.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_old_values = self.sig_old.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.C_tang_values = self.C_tang.x.array[:4*4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4, 4))

        if self.residue_size != 0:
            self.return_mapping_residue = crm.ReturnMapping(material, self.residue_size)
            
            self.p_values_residue = self.p.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.p_old_values_residue = self.p_old.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.deps_values_residue = self.deps.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_values_residue = self.sig.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_old_values_residue = self.sig_old.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.C_tang_values_residue = self.C_tang.x.array[4*4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4, 4))

        a_Newton = ufl.inner(eps_vec(self.v_), ufl.dot(self.C_tang, eps_vec(self.u_)))*self.dx 
        res = -ufl.inner(self.eps(self.u_), as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

        self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs)

    def inside_Newton(self):
        tol = 1e-13
        fs.interpolate_quadrature(self.eps_vec(self.Du), self.deps) # eps_xy * sqrt(2.)!
        for q in range(self.N_patches):
            self.return_mapping.deps.value[:] = self.deps_values[q,:].T
            self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
            self.return_mapping.p_old.value = self.p_old_values[q,:]
            
            self.return_mapping.solve(derivation=True, verbose=False, eps=tol, eps_abs=tol, eps_rel=tol) #, alpha=1, scale=5.

            self.sig_values[q,:] = self.return_mapping.sig.value[:].T
            self.p_values[q,:] = self.return_mapping.p.value
            self.C_tang_values[q,:] = self.return_mapping.C_tang[:]

        if self.residue_size != 0: #how to improve ?
            self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
            self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
            self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
            
            self.return_mapping_residue.solve(derivation=True, verbose=False, eps=tol, eps_abs=tol, eps_rel=tol) #, alpha=1, scale=5.
            
            self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
            self.p_values_residue[0,:] = self.return_mapping_residue.p.value
            self.C_tang_values_residue[0,:] = self.return_mapping_residue.C_tang[:]

    def after_Newton(self):
        self.p_old.x.array[:] = self.p.x.array        
        self.sig_old.x.array[:] = self.sig.x.array

    def initialize_variables(self):
        self.sig.vector.set(0.0)
        self.sig_old.vector.set(0.0)
        self.p.vector.set(0.0)
        self.p_old.vector.set(0.0)
        self.u.vector.set(0.0)

        for i in range(self.n_quadrature_points):
            self.C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = self.material.C

class ConvexPlasticitySNES(ConvexPlasticity):
    def __init__(self, material: crm.Material, patch_size: int = 1, mesh_name: str = "thick_cylinder.msh", logger: logging.Logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')
        WTe = ufl.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, shape=(4, 4), quad_scheme='default')
        
        W = fem.FunctionSpace(self.mesh, We)
        W0 = fem.FunctionSpace(self.mesh, W0e)
        WT = fem.FunctionSpace(self.mesh, WTe)

        self.sig = fem.Function(W)
        self.sig_old = fem.Function(W)
        self.p = fem.Function(W0, name="Cumulative_plastic_strain")
        self.p_old = fem.Function(W0) 

        self.C_tang = fem.Function(WT)
        self.deps = fem.Function(W, name="deps")
        
        def eps_vec(v):
            e = ufl.sym(ufl.grad(v))
            return ufl.as_vector([e[0, 0], e[1, 1], 0, SQRT2 * e[0, 1]])

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3] / SQRT2, 0],
                                [X[3] / SQRT2, X[1], 0],
                                [0, 0, X[2]]])       

        self.eps_vec = eps_vec

        self.n_quadrature_points = len(self.C_tang.x.array.reshape((-1, 4, 4)))
        for i in range(self.n_quadrature_points):
            self.C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = material.C

        self.N_patches = int(self.n_quadrature_points / patch_size)
        self.residue_size = self.n_quadrature_points % patch_size

        self.return_mapping = crm.ReturnMapping(material, patch_size)
        self.material = material

        self.p_values = self.p.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.p_old_values = self.p_old.x.array[:self.n_quadrature_points - self.residue_size].reshape((-1, patch_size))
        self.deps_values = self.deps.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_values = self.sig.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.sig_old_values = self.sig_old.x.array[:4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4))
        self.C_tang_values = self.C_tang.x.array[:4*4*(self.n_quadrature_points - self.residue_size)].reshape((-1, patch_size, 4, 4))

        if self.residue_size != 0:
            self.return_mapping_residue = crm.ReturnMapping(material, self.residue_size)
            
            self.p_values_residue = self.p.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.p_old_values_residue = self.p_old.x.array[self.n_quadrature_points - self.residue_size:].reshape((1, self.residue_size))
            self.deps_values_residue = self.deps.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_values_residue = self.sig.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.sig_old_values_residue = self.sig_old.x.array[4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4))
            self.C_tang_values_residue = self.C_tang.x.array[4*4*(self.n_quadrature_points - self.residue_size):].reshape((1, self.residue_size, 4, 4))

        a_Newton = ufl.inner(eps_vec(self.v_), ufl.dot(self.C_tang, eps_vec(self.u_)))*self.dx 
        res = -ufl.inner(self.eps(self.u_), as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

        self.problem = NonlinearProblem(a_Newton, res, self.Du, self.bcs)

        residual = ufl.inner(as_3D_tensor(self.sig) + sigma(self.eps(self.Du) - deps_p(self.eps(self.Du), self.sig, self.p)), self.eps(self.u_))*self.dx - self.F_ext(self.u_)

        petsc_options = {
            "snes_type": "vinewtonrsls",
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_atol": 1.0e-08,
            "snes_rtol": 1.0e-09,
            "snes_stol": 0.0,
            "snes_max_it": 500,
            "snes_monitor_cancel": "",
        }

        self.problem = SNESProblem(residual, self.Du, self.bcs, petsc_options = petsc_options)


    def inside_Newton(self):
        tol = 1e-13
        fs.interpolate_quadrature(self.eps_vec(self.Du), self.deps) # eps_xy * sqrt(2.)!
        for q in range(self.N_patches):
            self.return_mapping.deps.value[:] = self.deps_values[q,:].T
            self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
            self.return_mapping.p_old.value = self.p_old_values[q,:]
            
            self.return_mapping.solve(derivation=True, verbose=False, eps=tol, eps_abs=tol, eps_rel=tol) #, alpha=1, scale=5.

            self.sig_values[q,:] = self.return_mapping.sig.value[:].T
            self.p_values[q,:] = self.return_mapping.p.value
            self.C_tang_values[q,:] = self.return_mapping.C_tang[:]

        if self.residue_size != 0: #how to improve ?
            self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
            self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
            self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
            
            self.return_mapping_residue.solve(derivation=True, verbose=False, eps=tol, eps_abs=tol, eps_rel=tol) #, alpha=1, scale=5.
            
            self.sig_values_residue[0,:] = self.return_mapping_residue.sig.value[:].T
            self.p_values_residue[0,:] = self.return_mapping_residue.p.value
            self.C_tang_values_residue[0,:] = self.return_mapping_residue.C_tang[:]

class vonMisesPlasticitySNES(AbstractPlasticity):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh", logger: logging.Logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')
        
        W = fem.FunctionSpace(self.mesh, We)
        W0 = fem.FunctionSpace(self.mesh, W0e)

        self.sig = fem.Function(W)
        self.p = fem.Function(W0, name="Cumulative_plastic_strain")
        self.dp = fem.Function(W0) 

        mu = material.constitutive_law.mu_
        lmbda = material.constitutive_law.lambda_
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        
        def sigma(eps_el):
            return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3], 0],
                                  [X[3], X[1], 0],
                                  [0, 0, X[2]]])

        ppos = lambda x: (x + ufl.sqrt(x**2))/2.
        
        def proj_sig(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            f_elas = sig_eq - sig0 - H*old_p
            dp = ppos(f_elas)/(3*mu+H)
            beta = 3*mu*dp/sig_eq
            new_sig = sig_elas-beta*s
            return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
            dp       

        self.proj_sig = proj_sig

        def deps_p(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            f_elas = sig_eq - sig0 - H*old_p
            dp_sig_eq = ufl.conditional(f_elas > 0, f_elas/(3*mu+H)/sig_eq, 0) # sig_eq is equal to 0 on the first iteration
            # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
            return 3./2. * dp_sig_eq * s 

        residual = ufl.inner(as_3D_tensor(self.sig) + sigma(self.eps(self.Du) - deps_p(self.eps(self.Du), self.sig, self.p)), self.eps(self.u_))*self.dx - self.F_ext(self.u_)

        petsc_options = {
            "snes_type": "vinewtonrsls",
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_atol": 1.0e-08,
            "snes_rtol": 1.0e-09,
            "snes_stol": 0.0,
            "snes_max_it": 500,
            "snes_monitor_cancel": "",
        }

        self.problem = SNESProblem(residual, self.Du, self.bcs, petsc_options=petsc_options)

    def after_Newton(self):
        sig_, dp_ = self.proj_sig(self.eps(self.Du), self.sig, self.p)
        fs.interpolate_quadrature(sig_, self.sig)
        fs.interpolate_quadrature(dp_, self.dp)
        self.p.vector.axpy(1, self.dp.vector)
        self.p.x.scatter_forward()        

    def initialize_variables(self):
        self.sig.vector.set(0.0)
        self.p.vector.set(0.0)
        self.u.vector.set(0.0)
        
    def solve(self, Nincr: int = 20):
        load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
        results = np.zeros((Nincr+1, 2))
        load_steps = load_steps

        self.initialize_variables()
        
        start = time.time()

        for (i, t) in enumerate(load_steps):
            self.loading.value = t * self.q_lim

            nRes0 = self.problem.b.norm() 
            self.logger.info(f'rank#{MPI.COMM_WORLD.rank}: Step: {str(i+1)}, norm(nRes0) = {nRes0:.1e}, load = {t * self.q_lim}')
            self.problem.solve(self.logger)
            
            self.u.vector.axpy(1, self.Du.vector) # u = u + 1*Du
            self.u.x.scatter_forward()

            self.after_Newton()
            
            if len(self.points_on_proc) > 0:
                results[i+1, :] = (self.u.eval(self.points_on_proc, self.cells)[0], t)

        self.logger.log(LOG_INFO_STAR, f'rank#{MPI.COMM_WORLD.rank}: Time (total) = {time.time() - start:.2f} (s)')

        return self.points_on_proc, results

class DruckerPragerPlasticitySNES(vonMisesPlasticitySNES):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh", logger: logging.Logger = logging.getLogger()):
        super().__init__(material, mesh_name, logger)

        We = ufl.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, quad_scheme='default')
        
        W = fem.FunctionSpace(self.mesh, We)
        W0 = fem.FunctionSpace(self.mesh, W0e)

        self.sig = fem.Function(W)
        self.p = fem.Function(W0, name="Cumulative_plastic_strain")
        self.dp = fem.Function(W0) 

        mu = material.constitutive_law.mu_
        lmbda = material.constitutive_law.lambda_
        k = lmbda + 2*mu/3 # Bulk modulus 
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        alpha = material.yield_criterion.alpha
        M = (3*mu + 9*alpha*alpha*k)  + H
        TPV = np.finfo(PETSc.ScalarType).eps # très petite value 

        def sigma(eps_el):
            return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

        def as_3D_tensor(X):
            return ufl.as_tensor([[X[0], X[3], 0],
                                  [X[3], X[1], 0],
                                  [0, 0, X[2]]])

        ppos = lambda x: (x + ufl.sqrt(x**2))/2.
        def proj_sig(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            tr_sig = ufl.tr(sig_elas)
            f_elas = sig_eq + alpha*tr_sig - sig0 - H*old_p
            dp = ppos(f_elas)/M
            beta = 3*mu*dp/sig_eq 
            new_sig = sig_elas - (beta*s + dp * alpha*3*k * ufl.Identity(3)) 
            return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
                dp       

        self.proj_sig = proj_sig

        def deps_p(deps, old_sig, old_p):
            sig_n = as_3D_tensor(old_sig)
            sig_elas = sig_n + sigma(deps)
            s = ufl.dev(sig_elas)
            sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
            tr_sig = ufl.tr(sig_elas)
            f_elas = sig_eq + alpha*tr_sig - sig0 - H*old_p

            deps_p = ufl.conditional(f_elas > 0, f_elas/M  * (3./2. * s/sig_eq + alpha * ufl.Identity(3)), 0 * ufl.Identity(3))
            # dp_sig_eq = ufl.conditional(f_elas > 0, f_elas/M/sig_eq, 0) # sig_eq is equal to 0 on the first iteration
            # dp = ppos(f_elas)/(3*mu+H) # this approach doesn't work with ufl.derivate
            return deps_p
            # return dp_sig_eq * (3./2.  * s + alpha * sig_eq * ufl.Identity(3)) 
            # return (3./2. * dp_sig_eq * s + alpha * dp_sig_eq * sig_eq * ufl.Identity(3)) 

        residual = ufl.inner(as_3D_tensor(self.sig) + sigma(self.eps(self.Du) - deps_p(self.eps(self.Du), self.sig, self.p)), self.eps(self.u_))*self.dx - self.F_ext(self.u_)

        petsc_options = {
            "snes_type": "vinewtonrsls",
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_atol": 1.0e-08,
            "snes_rtol": 1.0e-09,
            "snes_stol": 0.0,
            "snes_max_it": 500,
            "snes_monitor_cancel": "",
        }

        self.problem = SNESProblem(residual, self.Du, self.bcs, petsc_options = petsc_options)
