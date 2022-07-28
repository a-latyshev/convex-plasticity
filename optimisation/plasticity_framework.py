
from yaml import load
import convex_return_mapping as crm # there is a conflict in the oder of imported modules
import meshio
import numpy as np

import ufl
from dolfinx import fem, io, common
from mpi4py import MPI
from petsc4py import PETSc

import typing

import time

import sys
sys.path.append("../")
import fenicsx_support as fs

SQRT2 = np.sqrt(2.)

class StandardProblem():
    def __init__(self,
                 dR: ufl.Form,
                 R: ufl.Form,
                 u: fem.Function,
                 bcs: typing.List[fem.dirichletbc] = []
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

    def assemble_vector(self):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b, self.b_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)

    def assemble_matrix(self):
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.A_form, bcs=self.bcs)
        self.A.assemble()

    def assemble(self):
        self.assemble_matrix()
        self.assemble_vector()
    
    def solve(self, 
              du: fem.function.Function, 
    ):
        """Solves the linear system and saves the solution into the vector `du`
        
        Args:
            du: A global vector to be used as a container for the solution of the linear system
        """
        self.solver.solve(self.b, du.vector)

class AbstractPlasticity():
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh"):
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
        self.du = fem.Function(self.V, name="Iteration_correction")
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

    def solve(self, Nitermax: int = 200, tol: float = 1e-8, Nincr:int = 20):
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

            self.problem.assemble_vector()

            nRes0 = self.problem.b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
            nRes = nRes0
            self.Du.x.array[:] = 0

            if MPI.COMM_WORLD.rank == 0:
                print(f"\nnRes0 , {nRes0} \n Increment: {str(i+1)}, load = {t * self.q_lim}")
            niter = 0

            while nRes/nRes0 > tol and niter < Nitermax:
                self.problem.solve(self.du)
                # print('du', np.max(self.du.x.array), np.min(self.du.x.array), self.du.vector.norm())

                self.Du.vector.axpy(1, self.du.vector) # Du = Du + 1*du
                self.Du.x.scatter_forward() 

                start_interpolate = time.time()

                self.inside_Newton()

                return_mapping_times_tmp.append(time.time() - start_interpolate)

                self.problem.assemble()

                nRes = self.problem.b.norm() 

                if MPI.COMM_WORLD.rank == 0:
                    print(f"    Residual: {nRes}, {nRes/nRes0}")
                niter += 1
            self.u.vector.axpy(1, self.Du.vector) # u = u + 1*Du
            self.u.x.scatter_forward()

            print('u', np.max(self.u.x.array), np.min(self.u.x.array), self.u.vector.norm())

            self.after_Newton()
            # fs.project(p, p_avg)
        
            # xdmf.write_function(u, t)
            # xdmf.write_function(p_avg, t)

            return_mapping_times[i] = np.mean(return_mapping_times_tmp)
            print(f'rank#{MPI.COMM_WORLD.rank}: Time (mean return mapping) = {return_mapping_times[i]:.3f} (s)')

            if len(self.points_on_proc) > 0:
                results[i+1, :] = (self.u.eval(self.points_on_proc, self.cells)[0], t)

        # xdmf.close()
        # end = time.time()
        print(f'\n rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {np.mean(return_mapping_times):.3f} (s)')
        print(f'rank#{MPI.COMM_WORLD.rank}: Time = {time.time() - start:.3f} (s)')

        return self.points_on_proc, results

class StandardPlasticity(AbstractPlasticity):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh"):
        super().__init__(material, mesh_name)

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

        self.problem = StandardProblem(a_Newton, res, self.Du, self.bcs)

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

class StandardDPPlasticity(StandardPlasticity):
    def __init__(self, material: crm.Material, mesh_name: str = "thick_cylinder.msh"):
        super().__init__(material, mesh_name)
        
        mu = material.constitutive_law.mu_
        lmbda = material.constitutive_law.lambda_
        k = lmbda + 2*mu/3 # Bulk modulus 
        sig0 = material.yield_criterion.sig0
        H = material.yield_criterion.H
        alpha = material.yield_criterion.alpha
        M = (3*mu + 9*alpha*alpha*k) / np.sqrt(1 + 2*alpha*alpha) + H
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
            new_sig = sig_elas - (beta*s + dp * alpha*3*k * ufl.Identity(3)) / np.sqrt(1 + 2*alpha*alpha)
            return ufl.as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
                ufl.as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
                beta, dp       

        self.proj_sig = proj_sig

        def sigma_tang(e):
            N_elas = as_3D_tensor(self.n_elas)
            return sigma(e) - ( 3*mu*(3*mu/M - self.beta) * ufl.inner(N_elas, e) * N_elas + 9*mu*k*alpha/M * ( ufl.inner(N_elas, e) * ufl.Identity(3) + ufl.tr(e) * N_elas ) + ( 27*k*k*alpha*alpha/M * ufl.tr(e) * ufl.Identity(3)) * self.beta / (self.beta + TPV) + 2*mu*self.beta*ufl.dev(e) ) / np.sqrt(1 + 2*alpha*alpha)

        a_Newton = ufl.inner(self.eps(self.v_), sigma_tang(self.eps(self.u_)))*self.dx
        res = -ufl.inner(self.eps(self.u_), as_3D_tensor(self.sig))*self.dx + self.F_ext(self.u_)

        self.problem = StandardProblem(a_Newton, res, self.Du, self.bcs)

class ConvexPlasticity(AbstractPlasticity):
    def __init__(self, material: crm.Material, patch_size: int = 1, mesh_name: str = "thick_cylinder.msh"):
        super().__init__(material, mesh_name)

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

        self.problem = StandardProblem(a_Newton, res, self.Du, self.bcs)

    def inside_Newton(self):
        fs.interpolate_quadrature(self.eps_vec(self.Du), self.deps) # eps_xy * sqrt(2.)!
        for q in range(self.N_patches):
            self.return_mapping.deps.value[:] = self.deps_values[q,:].T
            self.return_mapping.sig_old.value[:] = self.sig_old_values[q,:].T
            self.return_mapping.p_old.value = self.p_old_values[q,:]
            
            self.return_mapping.solve(derivation=True, verbose=False, eps=1e-13) #, alpha=1, scale=5.

            self.sig_values[q,:] = self.return_mapping.sig.value[:].T
            self.p_values[q,:] = self.return_mapping.p.value
            self.C_tang_values[q,:] = self.return_mapping.C_tang[:]

        if self.residue_size != 0: #how to improve ?
            self.return_mapping_residue.deps.value[:] = self.deps_values_residue[0,:].T
            self.return_mapping_residue.sig_old.value[:] = self.sig_old_values_residue[0,:].T
            self.return_mapping_residue.p_old.value = self.p_old_values_residue[0,:]
            
            self.return_mapping_residue.solve(derivation=True, verbose=False, eps=1e-13) #, alpha=1, scale=5.
            
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
