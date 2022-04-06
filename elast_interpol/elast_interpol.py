# %%
import numpy as np

import ufl
import basix
from dolfinx import mesh, fem, io, plot
from ufl import ds, dx

from mpi4py import MPI
from petsc4py import PETSc
import pyvista

# %%
def project(original_field, target_field, bcs=[]):
    # original_field -> target_field
    # Ensure we have a mesh and attach to measure
    V = target_field.function_space

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(original_field, w) * dx)

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
    solver.solve(b, target_field.vector)  
    target_field.x.scatter_forward()


# %%
L = 1
W = 0.1
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# %%
# Create mesh and define function space
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (L, W)), n=(64, 16),
                            cell_type=mesh.CellType.triangle,)

deg_u = 1
deg_stress = 0
deg_q = 1

DIM = domain.geometry.dim # == domain.topology.dim

QV0e = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=1, dim=3, quad_scheme='default')
QV0dim2e = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=1, dim=2, quad_scheme='default')
DGV0e = ufl.VectorElement("DG", domain.ufl_cell(), degree=0, dim=3)
QV2e = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=2, dim=3, quad_scheme='default')

Q0 = fem.FunctionSpace(domain, ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=0, quad_scheme='default'))
Q1 = fem.FunctionSpace(domain, ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=1, quad_scheme='default'))
Q2 = fem.FunctionSpace(domain, ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=2, quad_scheme='default'))
QV0 = fem.FunctionSpace(domain, QV0e)
QV0dim2 = fem.FunctionSpace(domain, QV0dim2e)
QV2 = fem.FunctionSpace(domain, QV2e)

DG0 = fem.FunctionSpace(domain, ('DG', 0))
DG1 = fem.FunctionSpace(domain, ('DG', 1))
DGV0 = fem.FunctionSpace(domain, DGV0e)

CG1 = fem.FunctionSpace(domain, ('CG', 1))
V = fem.VectorFunctionSpace(domain, ("Lagrange", deg_u))

# %%
num_nodes_global = domain.topology.index_map(domain.topology.dim-2).size_global
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global

# num_dofs_local = (V.dofmap.index_map.size_local) #* V.dofmap.index_map_bs
num_dofs_global = V.dofmap.index_map.size_global #* V.dofmap.index_map_bs
# num_dofs = domain.topology.index_map(domain.topology.dim).size_local 
# V.num_sub_spaces == V.dofmap.index_map_bs
# print(f"Number of dofs (owned) by rank : {num_dofs_local}")
if MPI.COMM_WORLD.rank == 0:
    print(f"Nodes global = {num_nodes_global}, Cells global = {num_cells_global}")
    print(f"Number of dofs global V: {num_dofs_global}")
    print(f"Number of dofs global Q0: {Q0.dofmap.index_map.size_global}")
    print(f"Number of dofs global Q1: {Q1.dofmap.index_map.size_global}")
    print(f"Number of dofs global Q2: {Q2.dofmap.index_map.size_global}")
    print(f"Number of dofs global DG0: {DG0.dofmap.index_map.size_global}")
    print(f"Number of dofs global DG1: {DG1.dofmap.index_map.size_global}")
    print(f"Number of dofs global CG1: {CG1.dofmap.index_map.size_global}")

# %%
facets = mesh.locate_entities_boundary(domain, dim=1,
                                       marker=lambda x: np.isclose(x[0], 0.0))

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0))), dofs=dofs, V=V)

# %%
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_*ufl.div(u)*ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

# Define variational problem
f = fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(-rho*g)))
T = fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0)))
a = ufl.inner(sigma(u), epsilon(v))*dx
b = ufl.inner(f, v)*dx + ufl.inner(T, v)*ds

problem = fem.petsc.LinearProblem(a, b, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.x.scatter_forward()

# %%
# defining function to interpolate function defined over quadrature elements
def interpolate_quadrature(ufl_expr, fem_func):
    q_dim = fem_func.function_space._ufl_element.degree()
    mesh = fem_func.ufl_function_space().mesh
    
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(cells)
    fem_func.x.array[:] = expr_eval.flatten()[:]
    fem_func.x.scatter_forward()
    # with funct.vector.localForm() as funct_local:
    #     funct_local.setBlockSize(funct.function_space.dofmap.bs)
    #     funct_local.setValuesBlocked(
    #         funct.function_space.dofmap.list.array,
    #         expr_eval,
    #         addv=PETSc.InsertMode.INSERT,
        # )


# %%
sigmah = sigma(uh)
sigma_xx_ = sigmah[0,0]
sigma_xy_ = sigmah[0,1]
sigma_yy_ = sigmah[1,1]

sigma_vec = ufl.as_vector([sigmah[0, 0], sigmah[0, 1], sigmah[1, 1]])
sigma_vec2 = ufl.as_vector([sigmah[0, 0], sigmah[0, 1]])

# %%
sigma_xx_Q0 = fem.Function(Q0)
sigma_xx_DG0 = fem.Function(DG0)
sigma_xx_DG1 = fem.Function(DG1)

sigma_xx_DG0_interp = fem.Function(DG0)
sigma_xy_DG0_interp = fem.Function(DG0)
sigma_yy_DG0_interp = fem.Function(DG0)
sigma_DGV0_interp = fem.Function(DGV0)

sigma_xx_DG1_interp = fem.Function(DG1)

sigma_xx_Q0_interp = fem.Function(Q0)
sigma_xy_Q0_interp = fem.Function(Q0)
sigma_yy_Q0_interp = fem.Function(Q0)
sigma_QV0 = fem.Function(QV0)
sigma_QV0dim2 = fem.Function(QV0dim2)

sigma_xx_Q2_interp = fem.Function(Q2)
sigma_xy_Q2_interp = fem.Function(Q2)
sigma_yy_Q2_interp = fem.Function(Q2)
sigma_QV2 = fem.Function(QV2)

# %%
project(sigma_xx_, sigma_xx_Q0)
project(sigma_xx_, sigma_xx_DG0)
project(sigma_xx_, sigma_xx_DG1)
sigma_xx_DG1.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# project(sigma_vec, sigma_QV0) #failed

# %%
expr = fem.Expression(sigma_xx_, DG0.element.interpolation_points)
sigma_xx_DG0_interp.interpolate(expr)
expr = fem.Expression(sigma_xy_, DG0.element.interpolation_points)
sigma_xy_DG0_interp.interpolate(expr)
expr = fem.Expression(sigma_yy_, DG0.element.interpolation_points)
sigma_yy_DG0_interp.interpolate(expr)

expr = fem.Expression(sigma_xx_, DG1.element.interpolation_points)
sigma_xx_DG1_interp.interpolate(expr)
sigma_xx_DG1_interp.x.scatter_forward()

expr = fem.Expression(sigma_vec, DGV0.element.interpolation_points)
sigma_DGV0_interp.interpolate(expr)

interpolate_quadrature(sigma_xx_, sigma_xx_Q0_interp)
interpolate_quadrature(sigma_vec, sigma_QV0)
interpolate_quadrature(sigma_vec2, sigma_QV0dim2)
interpolate_quadrature(sigma_vec, sigma_QV2)

# %%
sigma_xx_Q0_interp, sigma_xy_Q0_interp, sigma_yy_Q0_interp = sigma_QV0.split()
sigma_xx_DG0_interp, sigma_xy_DG0_interp, sigma_yy_DG0_interp = sigma_DGV0_interp.split()
sigma_xx_Q2_interp, sigma_xy_Q2_interp, sigma_yy_Q2_interp = sigma_QV2.split()

# %%
DG0.element.interpolation_points

# %%
DG1.element.interpolation_points

# %%
quadrature_points, _ = basix.make_quadrature(basix.CellType.triangle, 2)
quadrature_points

# %%
num_ghosts = domain.topology.index_map(domain.topology.dim).num_ghosts
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local #== domain.geometry.dofmap.num_nodes
num_nodes_local = domain.topology.index_map(domain.topology.dim-2).size_local #== domain.geometry.x.shape[0]

print(f"rank = {MPI.COMM_WORLD.rank}:")
print(f"\tNodes local = {num_nodes_local}, Cells local = {num_cells_local}, Cells ghost = {num_ghosts}")
print(f"\tDoFs local = {V.dofmap.index_map.size_local}, Number of uh_x = {uh.x.array.shape[0]/V.num_sub_spaces}, Number of sigma_xx_DG0 = {sigma_xx_DG0.x.array.shape[0]}")

# %%
uh.name = "Displacement"
sigma_xx_Q0.name = 'sigma xx Q0'
sigma_xx_DG0.name = 'sigma xx DG0'
sigma_xx_DG1.name = 'sigma xx DG1'
sigma_xx_DG0_interp.name = 'sigma xx DG0 interp'
sigma_xy_DG0_interp.name = 'sigma xy DG0 interp'
sigma_yy_DG0_interp.name = 'sigma yy DG0 interp'

sigma_xx_DG1_interp.name = 'sigma xx DG1 interp'

sigma_xx_Q0_interp.name = 'sigma xx Q0 interp'
sigma_xy_Q0_interp.name = 'sigma xy Q0 interp'
sigma_yy_Q0_interp.name = 'sigma yy Q0 interp'

sigma_xx_Q2_interp.name = 'sigma xx Q2 interp'
sigma_xy_Q2_interp.name = 'sigma xy Q2 interp'
sigma_yy_Q2_interp.name = 'sigma yy Q2 interp'

sigma_DGV0_interp.name = 'sigma DGV0 interp'
sigma_QV0dim2.name = 'sigma QV0dim2'
sigma_QV0.name = "Stress"

with io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(domain)

with io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "a", encoding=io.XDMFFile.Encoding.HDF5) as file:
    file.write_function(uh)
    file.write_function(sigma_xx_Q0)
    file.write_function(sigma_xx_DG0)
    file.write_function(sigma_xx_DG1)
    file.write_function(sigma_xx_DG0_interp)
    file.write_function(sigma_xy_DG0_interp)
    file.write_function(sigma_yy_DG0_interp)
    file.write_function(sigma_xx_DG1_interp)
    file.write_function(sigma_xx_Q0_interp)
    file.write_function(sigma_xy_Q0_interp)
    file.write_function(sigma_yy_Q0_interp)
    # file.write_function(sigma_xx_Q2_interp) #failed!
    file.write_function(sigma_DGV0_interp)
    file.write_function(sigma_QV0dim2)
    # file.write_function(sigma_QV0, 0) #failed!

# %%



