
import convex_return_mapping as crm # there is a conflict in the oder of imported modules
import meshio
import numpy as np

import ufl
from dolfinx import fem, io, common
from mpi4py import MPI
from petsc4py import PETSc

import typing

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
        fem.petsc.assemble_matrix(self.A, self.A_form, bcs=bcs)
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


class PlasticityFramework():
    def __init__(self, material:crm.Material, mesh_name:str = "thick_cylinder.msh"):
        if MPI.COMM_WORLD.rank == 0:
            # It works with the msh4 only!!
            msh = meshio.read(mesh_name)

            # Create and save one file for the mesh, and one file for the facets 
            triangle_mesh = fs.create_mesh(msh, "triangle", prune_z=True)
            line_mesh = fs.create_mesh(msh, "line", prune_z=True)
            meshio.write("thick_cylinder.xdmf", triangle_mesh)
            meshio.write("mt.xdmf", line_mesh)
            print(msh)
        
        with io.XDMFFile(MPI.COMM_WORLD, "thick_cylinder.xdmf", "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")
            ct = xdmf.read_meshtags(self.mesh, name="Grid")

        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim - 1)

        with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
            ft = xdmf.read_meshtags(self.mesh, name="Grid")

        Re, Ri = 1.3, 1.   # external/internal radius

        deg_u = 2
        deg_stress = 2

        self.V = fem.VectorFunctionSpace(self.mesh, ("CG", deg_u))

        left_marker = 3
        down_marker = 1
        left_facets = ft.indices[ft.values == left_marker]
        down_facets = ft.indices[ft.values == down_marker]
        left_dofs = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim-1, left_facets)
        down_dofs = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim-1, down_facets)

        self.bcs = [fem.dirichletbc(PETSc.ScalarType(0), left_dofs, V.sub(0)), fem.dirichletbc(PETSc.ScalarType(0), down_dofs, V.sub(1))]

        n = ufl.FacetNormal(mesh)
        q_lim = float(2/np.sqrt(3)*np.log(Re/Ri)*material.yield_criterion.sig0)
        loading = fem.Constant(mesh, PETSc.ScalarType(0.0 * q_lim))


        ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
        dx = ufl.Measure(
            "dx",
            domain=mesh,
            metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
        )

        def F_ext(v):
            return -loading * ufl.inner(n, v)*ds(4)


        We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
        W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
        WTe = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, shape=(4, 4), quad_scheme='default')

        W = fem.FunctionSpace(mesh, We)
        W0 = fem.FunctionSpace(mesh, W0e)
        WT = fem.FunctionSpace(mesh, WTe)

        sig = fem.Function(W)
        sig_old = fem.Function(W)
        p = fem.Function(W0, name="Cumulative_plastic_strain")
        p_old = fem.Function(W0, name="Cumulative_plastic_strain")
        u = fem.Function(V, name="Total_displacement")
        du = fem.Function(V, name="Iteration_correction")
        Du = fem.Function(V, name="Current_increment")
        v = ufl.TrialFunction(V)
        u_ = ufl.TestFunction(V)
        C_tang = fem.Function(WT)

        deps = fem.Function(W, name="deps")

        P0 = fem.FunctionSpace(mesh, ("DG", 0))
        p_avg = fem.Function(P0, name="Plastic_strain")


        def eps(v):
            e = ufl.sym(ufl.grad(v))
            return ufl.as_tensor([[e[0, 0], e[0, 1], 0],
                                [e[0, 1], e[1, 1], 0],
                                [0, 0, 0]])

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

        def sigma_tang(e):
            N_elas = as_3D_tensor(n_elas)
            return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*ufl.inner(N_elas, e)*N_elas - 2*mu*beta*ufl.dev(e) 

        a_Newton = ufl.inner(eps(v), sigma_tang(eps(u_)))*dx
        res = -ufl.inner(eps(u_), as_3D_tensor(sig))*dx + F_ext(u_)

        self.my_problem = StandardProblem(a_Newton, res, Du, bcs)

    def solve(self, Nitermax: int = 200, tol: float = 1e-8, Nincr:int = 20):

        load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
        results = np.zeros((Nincr+1, 2))
        load_steps = load_steps
        # xdmf = io.XDMFFile(MPI.COMM_WORLD, "plasticity.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
        # xdmf.write_mesh(mesh)

        sig.vector.set(0.0)
        sig_old.vector.set(0.0)
        p.vector.set(0.0)
        u.vector.set(0.0)
        n_elas.vector.set(0.0)
        beta.vector.set(0.0)

        self.my_problem.assemble_matrix()

        return_mapping_times = np.zeros((len(load_steps)))

        start = time.time()

        for (i, t) in enumerate(load_steps):
            return_mapping_times_tmp = []
            loading.value = t * q_lim

            my_problem.assemble_vector()

            nRes0 = my_problem.b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
            nRes = nRes0
            Du.x.array[:] = 0

            if MPI.COMM_WORLD.rank == 0:
                print(f"\nnRes0 , {nRes0} \n Increment: {str(i+1)}, load = {t * q_lim}")
            niter = 0


            while nRes/nRes0 > tol and niter < Nitermax:
                my_problem.solve(du)
                # print('du', np.max(du.x.array), np.min(du.x.array), du.vector.norm())

                Du.vector.axpy(1, du.vector) # Du = Du + 1*du
                Du.x.scatter_forward() 

                start_interpolate = time.time()

                deps = eps(Du)
                sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)

                fs.interpolate_quadrature(sig_, sig)
                fs.interpolate_quadrature(n_elas_, n_elas)
                fs.interpolate_quadrature(beta_, beta)
                # fs.interpolate_quadrature(dp_, dp)

                return_mapping_times_tmp.append(time.time() - start_interpolate)

                my_problem.assemble()

                nRes = my_problem.b.norm() 

                if MPI.COMM_WORLD.rank == 0:
                    print(f"    Residual: {nRes}")
                niter += 1
            u.vector.axpy(1, Du.vector) # u = u + 1*Du
            u.x.scatter_forward()

            fs.interpolate_quadrature(dp_, dp)
            p.vector.axpy(1, dp.vector)
            p.x.scatter_forward()
            print('p after copy', np.max(p.x.array), np.min(p.x.array), p.vector.norm())
            
            sig_old.x.array[:] = sig.x.array[:]

            # fs.project(p, p_avg)
            
            # xdmf.write_function(u, t)
            # xdmf.write_function(p_avg, t)

            return_mapping_times[i] = np.mean(return_mapping_times_tmp)
            print(f'rank#{MPI.COMM_WORLD.rank}: Time (mean return mapping) = {return_mapping_times[i]:.3f} (s)')

            if len(points_on_proc) > 0:
                results[i+1, :] = (u.eval(points_on_proc, cells)[0], t)

        # xdmf.close()
        # end = time.time()
        print(f'\n rank#{MPI.COMM_WORLD.rank}: Time (return mapping) = {np.mean(return_mapping_times):.3f} (s)')
        print(f'rank#{MPI.COMM_WORLD.rank}: Time = {time.time() - start:.3f} (s)')


def solve_convex_plasticity_interpolation(sig0, material, patch_size=3):

    with io.XDMFFile(MPI.COMM_WORLD, "thick_cylinder.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    # elastic parameters
    # sig0_dim = 70e3 #[Pa]
    # Ri_dim = 1.0 #[m]

    # E = 70e3 / Pa_dim #[-]
    # nu = 0.3 #[-]
    # lmbda = E*nu/(1+nu)/(1-2*nu)
    # mu = fem.Constant(mesh, PETSc.ScalarType(E/2./(1+nu)))

    # sig0 = 250 / Pa_dim #[-]
    # Et = E/100.  # tangent modulus
    # H = E*Et/(E-Et)  # hardening modulus

    Re, Ri = 1.3, 1.   # external/internal radius


    deg_u = 2
    deg_stress = 2
    V = fem.VectorFunctionSpace(mesh, ("CG", deg_u))
    We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
    W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    WTe = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, shape=(4, 4), quad_scheme='default')

    W = fem.FunctionSpace(mesh, We)
    W0 = fem.FunctionSpace(mesh, W0e)
    WT = fem.FunctionSpace(mesh, WTe)


    sig = fem.Function(W)
    sig_old = fem.Function(W)
    p = fem.Function(W0, name="Cumulative_plastic_strain")
    p_old = fem.Function(W0, name="Cumulative_plastic_strain")
    u = fem.Function(V, name="Total_displacement")
    du = fem.Function(V, name="Iteration_correction")
    Du = fem.Function(V, name="Current_increment")
    v = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)
    C_tang = fem.Function(WT)

    deps = fem.Function(W, name="deps")

    P0 = fem.FunctionSpace(mesh, ("DG", 0))
    p_avg = fem.Function(P0, name="Plastic_strain")


    left_marker = 3
    down_marker = 1
    left_facets = ft.indices[ft.values == left_marker]
    down_facets = ft.indices[ft.values == down_marker]
    left_dofs = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim-1, left_facets)
    down_dofs = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim-1, down_facets)

    bcs = [fem.dirichletbc(PETSc.ScalarType(0), left_dofs, V.sub(0)), fem.dirichletbc(PETSc.ScalarType(0), down_dofs, V.sub(1))]

    n = ufl.FacetNormal(mesh)
    q_lim = float(2/np.sqrt(3)*np.log(Re/Ri)*sig0)
    loading = fem.Constant(mesh, PETSc.ScalarType(0.0 * q_lim))

    SQRT2 = np.sqrt(2.)

    def F_ext(v):
        return -loading * ufl.inner(n, v)*ds(4)

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_tensor([[e[0, 0], e[0, 1], 0],
                            [e[0, 1], e[1, 1], 0],
                            [0, 0, 0]])

    def eps_vec(v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_vector([e[0, 0], e[1, 1], 0, SQRT2 * e[0, 1]])

    def as_3D_tensor(X):
        return ufl.as_tensor([[X[0], X[3] / SQRT2, 0],
                            [X[3] / SQRT2, X[1], 0],
                            [0, 0, X[2]]])       


    # vonMises = crm.vonMises(sig0, H)
    # material = crm.Material(crm.IsotropicElasticity(E, nu), vonMises)

    n_quadrature_points = len(C_tang.x.array.reshape((-1, 4, 4)))
    for i in range(n_quadrature_points):
        C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = material.C

    N_patches = int(n_quadrature_points/patch_size)
    residue_size = n_quadrature_points % patch_size

    return_mapping = crm.ReturnMapping(material, patch_size)

    p_values = p.x.array[:n_quadrature_points - residue_size].reshape((-1, patch_size))
    p_old_values = p_old.x.array[:n_quadrature_points - residue_size].reshape((-1, patch_size))
    deps_values = deps.x.array[:4*(n_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    sig_values = sig.x.array[:4*(n_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    sig_old_values = sig_old.x.array[:4*(n_quadrature_points - residue_size)].reshape((-1, patch_size, 4))
    C_tang_values = C_tang.x.array[:4*4*(n_quadrature_points - residue_size)].reshape((-1, patch_size, 4, 4))

    if residue_size != 0:
        return_mapping_residue = crm.ReturnMapping(material, residue_size)
        
        p_values_residue = p.x.array[n_quadrature_points - residue_size:].reshape((1, residue_size))
        p_old_values_residue = p_old.x.array[n_quadrature_points - residue_size:].reshape((1, residue_size))
        deps_values_residue = deps.x.array[4*(n_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        sig_values_residue = sig.x.array[4*(n_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        sig_old_values_residue = sig_old.x.array[4*(n_quadrature_points - residue_size):].reshape((1, residue_size, 4))
        C_tang_values_residue = C_tang.x.array[4*4*(n_quadrature_points - residue_size):].reshape((1, residue_size, 4, 4))


    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
    )

    a_Newton = ufl.inner(eps_vec(v), ufl.dot(C_tang, eps_vec(u_)))*dx 
    res = -ufl.inner(eps(u_), as_3D_tensor(sig))*dx + F_ext(u_)

    form_res = fem.form(res)
    form_a_Newton = fem.form(a_Newton)

    b = fem.petsc.create_vector(form_res)
    A = fem.petsc.create_matrix(form_a_Newton)

    with b.localForm() as b_local:
        b_local.set(0.0)
    fem.petsc.assemble_vector(b, form_res)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs)

    A.zeroEntries()
    fem.petsc.assemble_matrix(A, form_a_Newton, bcs=bcs)
    A.assemble()

    solver = PETSc.KSP().create(mesh.comm)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setOperators(A)

    x_point = np.array([[Ri, 0, 0]])
    cells, points_on_proc = fs.find_cell_by_point(mesh, x_point)

    def error_norm1(uh, u):
        return np.sum(np.abs(u - uh)) / np.sum(np.abs(u))

    def error_norm2(uh, u):
        return np.sqrt(np.dot(u - uh, u - uh)) / np.sqrt(np.dot(u, u))

    C_fake = fem.Function(WT)
    for i in range(n_quadrature_points):
        C_fake.x.array.reshape((-1, 4, 4))[i,:,:] = material.C


    Nitermax, tol = 200, 1e-7  # parameters of the Newton-Raphson procedure
    Nincr = 20
    load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
    results = np.zeros((Nincr+1, 2))
    load_steps = load_steps
    # xdmf = io.XDMFFile(MPI.COMM_WORLD, "plasticity.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
    # xdmf.write_mesh(mesh)

    sig.vector.set(0.0)
    sig_old.vector.set(0.0)
    p.vector.set(0.0)
    p_old.vector.set(0.0)
    u.vector.set(0.0)

    n_quadrature_points = len(C_tang.x.array.reshape((-1, 4, 4)))
    for i in range(n_quadrature_points):
        C_tang.x.array.reshape((-1, 4, 4))[i,:,:] = material.C
    # T.vector.norm - bug, where T is a tensor???

    logging_parts = ['convex_solving', 'convex_solving_cvxpy', 'differentiation', 'return_mapping', 'differentiation_total', 'convex_solving_total']
    main_logging_parts = ['convex_solving', 'convex_solving_cvxpy', 'differentiation']

    time_logging = {}
    time_logging_local = {}
    for part in logging_parts: 
        time_logging[part] = np.zeros((Nincr))
        time_logging_local[part] = np.zeros((n_quadrature_points))

    mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS' : 1.0e-13}

    with common.Timer() as timer: 
        for (i, t) in enumerate(load_steps):
            time_logging_tmp = {}
            for part in logging_parts: 
                time_logging_tmp[part] = []

            loading.value = t * q_lim

            with b.localForm() as b_local:
                b_local.set(0.0)
            b = fem.petsc.assemble_vector(form_res)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(b, bcs)

            nRes0 = b.norm() # Which one? - ufl.sqrt(Res.dot(Res))
            nRes = nRes0
            Du.x.array[:] = 0

            if MPI.COMM_WORLD.rank == 0:
                print(f"\nnRes0 , {nRes0} \n Increment: {str(i+1)}, load = {t * q_lim}")
            niter = 0

            while nRes/nRes0 > tol and niter < Nitermax:
                solver.solve(b, du.vector)
                print('du', np.max(du.x.array), np.min(du.x.array), du.vector.norm())

                Du.vector.axpy(1, du.vector) # Du = Du + 1*du
                Du.x.scatter_forward() 

                with common.Timer() as timer_return_mapping: 
                    fs.interpolate_quadrature(eps_vec(Du), deps) # eps_xy * sqrt(2.)!

                    for q in range(N_patches):
                        return_mapping.deps.value[:] = deps_values[q,:].T
                        return_mapping.sig_old.value[:] = sig_old_values[q,:].T
                        return_mapping.p_old.value = p_old_values[q,:]
                        
                        return_mapping.solve(derivation=True, verbose=False, eps=1e-13) #, alpha=1, scale=5.
                        time_logging_local['convex_solving'][q] = return_mapping.convex_solving_time
                        time_logging_local['convex_solving_cvxpy'][q] = return_mapping.opt_problem._solve_time
                        time_logging_local['differentiation'][q] = return_mapping.differentiation_time

                        sig_values[q,:] = return_mapping.sig.value[:].T
                        p_values[q,:] = return_mapping.p.value
                        C_tang_values[q,:] = return_mapping.C_tang[:]

                    if residue_size != 0: #how to improve ?
                        return_mapping_residue.deps.value[:] = deps_values_residue[0,:].T
                        return_mapping_residue.sig_old.value[:] = sig_old_values_residue[0,:].T
                        return_mapping_residue.p_old.value = p_old_values_residue[0,:]
                        
                        return_mapping_residue.solve(derivation=True, verbose=False, eps=1e-13) #, alpha=1, scale=5.
                        
                        sig_values_residue[0,:] = return_mapping_residue.sig.value[:].T
                        p_values_residue[0,:] = return_mapping_residue.p.value
                        C_tang_values_residue[0,:] = return_mapping_residue.C_tang[:]

                    time_logging_tmp['return_mapping'].append(timer_return_mapping.elapsed()[0])

                for part in main_logging_parts:
                    time_logging_tmp[part].append(np.mean(time_logging_local[part]))

                time_logging_tmp['convex_solving_total'].append(np.sum(time_logging_local['convex_solving']))
                time_logging_tmp['differentiation_total'].append(np.sum(time_logging_local['differentiation']))

                print('C_tang', np.max(C_tang.x.array), np.min(C_tang.x.array))
                print('C_tang relative error', error_norm1(C_tang.x.array, C_fake.x.array), error_norm2(C_tang.x.array, C_fake.x.array))

                A.zeroEntries()
                fem.petsc.assemble_matrix(A, form_a_Newton, bcs=bcs)
                A.assemble()
                with b.localForm() as b_local:
                    b_local.set(0.0)
                fem.petsc.assemble_vector(b, form_res)

                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem.set_bc(b, bcs)

                nRes = b.norm() 
                if MPI.COMM_WORLD.rank == 0:
                    print(f"    Residual: {nRes}")
                niter += 1
            u.vector.axpy(1, Du.vector) # u = u + 1*Du
            u.x.scatter_forward()

            p_old.x.array[:] = p.x.array

            print('p after copy', np.max(p.x.array), np.min(p.x.array), p.vector.norm())
            
            sig_old.x.array[:] = sig.x.array

            # fs.project(p, p_avg)
            
            # xdmf.write_function(u, t)
            # xdmf.write_function(p_avg, t)

            for part in logging_parts:
                time_logging[part][i] = np.mean(time_logging_tmp[part])

            for part in logging_parts:
                print(f'\trank#{MPI.COMM_WORLD.rank}: Time ({part}) = {np.mean(time_logging[part][i]):.5f} (s)')
          
            if len(points_on_proc) > 0:
                results[i+1, :] = (u.eval(points_on_proc, cells)[0], t)
        
        # xdmf.close()

        print(f'\nrank#{MPI.COMM_WORLD.rank}: Time = {timer.elapsed()[0]:.3f} (s)')
        
        important_logging_parts = ['return_mapping', 'differentiation_total', 'convex_solving_total']
        for part in important_logging_parts:
            print(f'\trank#{MPI.COMM_WORLD.rank}: Time ({part}) = {np.sum(time_logging[part]):.3f} (s)')

    return points_on_proc, results


