import meshio
import numpy as np

import ufl
from dolfinx import fem, io, cpp
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

#It works with the msh4 only!!
msh = meshio.read("thick_cylinder.msh")

# Create and save one file for the mesh, and one file for the facets 
triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
line_mesh = create_mesh(msh, "line", prune_z=True)
meshio.write("thick_cylinder.xdmf", triangle_mesh)
meshio.write("mt.xdmf", line_mesh)

with io.XDMFFile(MPI.COMM_WORLD, "thick_cylinder.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")

mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")

# elastic parameters
E = 70e3
nu = 0.3
lmbda = E*nu/(1+nu)/(1-2*nu)
mu = fem.Constant(mesh, PETSc.ScalarType(E/2./(1+nu)))
sig0 = fem.Constant(mesh, PETSc.ScalarType(250))  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus

Re, Ri = 1.3, 1.   # external/internal radius
ds = ufl.Measure("ds", domain=mesh)

deg_u = 2
deg_stress = 2
V = fem.VectorFunctionSpace(mesh, ("CG", deg_u))
We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W = fem.FunctionSpace(mesh, We)
W0 = fem.FunctionSpace(mesh, W0e)

sig = fem.Function(W)
sig_old = fem.Function(W)
n_elas = fem.Function(W)
beta = fem.Function(W0)
p = fem.Function(W0)# , name="Cumulative plastic strain"
dp = fem.Function(W0)
u = fem.Function(V)#, name="Total displacement"
du = fem.Function(V)#, name="Iteration correction"
Du = fem.Function(V)# , name="Current increment"
v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

zero_Du = fem.Function(V)

left_marker = 3
down_marker = 1
left_facets = ft.indices[ft.values == left_marker]
down_facets = ft.indices[ft.values == down_marker]
left_dofs = fem.locate_dofs_topological(V.sub(0), mesh.topology.dim-1, left_facets)
down_dofs = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim-1, down_facets)

bcs = [fem.dirichletbc(PETSc.ScalarType(0), left_dofs, V.sub(0)), fem.dirichletbc(PETSc.ScalarType(0), down_dofs, V.sub(1))]

n = ufl.FacetNormal(mesh)
q_lim = float(2/np.sqrt(3)*np.log(Re/Ri)*sig0.value)

V_real = fem.FunctionSpace(mesh, ("CG", deg_u))
loading = fem.Function(V_real)
loading.interpolate(lambda x: (np.zeros_like(x[1])))
loading.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

def F_ext(v):
    return -loading * ufl.inner(n, v)*ds(4)

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

dx = ufl.Measure(
    "dx",
    domain=mesh,
    metadata={"quadrature_degree": deg_stress, "quadrature_scheme": "default"},
)

a_Newton = ufl.inner(eps(v), sigma_tang(eps(u_)))*dx
res = ufl.inner(eps(u_), as_3D_tensor(sig))*dx + F_ext(u_)

def project(v, target_func, bcs=[]):
    # v->target_func
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

P0 = fem.FunctionSpace(mesh, ("DG", 0))
p_avg = fem.Function(P0, name="Plastic strain")

A = fem.petsc.create_matrix(fem.form(a_Newton))
Res = fem.petsc.create_vector(fem.form(res))

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

Nitermax, tol = 200, 1e-8  # parameters of the Newton-Raphson procedure
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
results = np.zeros((Nincr+1, 2))
for (i, t) in enumerate(load_steps):
    # loading.t = t
    # load_func.interpolate(loading.eval)
    loading.interpolate(lambda x: (t * q_lim * np.ones_like(x[1])))
    loading.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    A.zeroEntries()
    fem.petsc.assemble_matrix(A, fem.form(a_Newton), bcs=bcs)
    A.assemble()

    with Res.localForm() as loc_b:
        loc_b.set(0.)
    fem.petsc.assemble_vector(Res, fem.form(res))
    Res.assemble()

    fem.petsc.apply_lifting(Res, [fem.form(a_Newton)], [bcs], x0=[Du.vector], scale=-1.0)
    Res.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(Res, bcs, Du.vector, -1.0)

    solver.setOperators(A)

    nRes0 = Res.norm() # Which one?
    nRes = 1
    zero_Du.vector.copy(Du.vector)
    Du.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    # Du.interpolate(lambda x : (np.zeros_like(x[0]), np.zeros_like(x[1])))
    # Du.interpolate(fem.Constant(mesh, (ScalarType(0), ScalarType(0))))
    print("Increment:", str(i+1), ' force: ', t * q_lim)
    niter = 0
     
    while nRes > tol and niter < Nitermax:
        solver.solve(Res, du.vector) 
        du.x.scatter_forward() 
        print('du max = ', du.vector.max())

        Du.vector.axpy(1, du.vector) # Du = Du + 1*du
        Du.x.scatter_forward() 

        deps = eps(Du)
        sig_, n_elas_, beta_, dp_ = proj_sig(deps, sig_old, p)

        project(sig_, sig)
        project(n_elas_, n_elas)
        project(beta_, beta)
        
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, fem.form(a_Newton), bcs=bcs)
        A.assemble()
        with Res.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(Res, fem.form(res))
        Res.assemble()

        fem.petsc.apply_lifting(Res, [fem.form(a_Newton)], [bcs], x0=[Du.vector], scale=-1.0)
        Res.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(Res, bcs, Du.vector, -1.0)

        solver.setOperators(A)

        nRes = Res.norm()
        print("    Residual:", nRes, ' dp = ', dp.vector.max())
        niter += 1
    u.vector.axpy(1, Du.vector) # u = u + 1*Du
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    sig.vector.copy(sig_old.vector)
    sig_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    project(dp_, dp)
    p.vector.axpy(1, dp.vector)
    p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
