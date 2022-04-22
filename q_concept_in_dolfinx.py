import sys

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import dolfinx as df
import basix
import ufl


def print0(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)


try:
    N = int(sys.argv[1])
except IndexError:
    N = 1

mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, N, N) #, df.mesh.CellType.quadrilateral

deg, q_deg = 1, 2
V = df.fem.VectorFunctionSpace(mesh, ("P", deg))

# quadrature elements and function spaces
QV = ufl.VectorElement(
    "Quadrature", mesh.ufl_cell(), q_deg, quad_scheme="default", dim=3
)
QT = ufl.TensorElement(
    "Quadrature",
    mesh.ufl_cell(),
    q_deg,
    quad_scheme="default",
    shape=(3, 3),
)
VQV = df.fem.FunctionSpace(mesh, QV)
VQT = df.fem.FunctionSpace(mesh, QT)

# define functions
u_, du = ufl.TestFunction(V), ufl.TrialFunction(V)
u = df.fem.Function(V)
q_sigma = df.fem.Function(VQV)
q_dsigma = df.fem.Function(VQT)

num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
num_gauss_local = len(q_sigma.x.array[:]) // 3
num_gauss_global = mesh.comm.reduce(num_gauss_local, op=MPI.SUM, root=0)

print0(f"{num_dofs_global  = }")
print0(f"{num_gauss_global = }")

# define form
dxm = ufl.dx(metadata={"quadrature_degree": q_deg, "quadrature_scheme": "default"})


def eps(u):
    e = ufl.sym(ufl.grad(u))
    return ufl.as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))


R = df.fem.form(ufl.inner(eps(u_), q_sigma) * dxm)
dR = df.fem.form(ufl.inner(eps(du), ufl.dot(q_dsigma, eps(u_))) * dxm)


E, nu = 20000, 0.3


def evaluate_constitutive_law(u):
    """
    magic part!

    Evaluates strain expression eps(u) on all quadrature points of all cells,
    computes stresses sigma and their derivative dsigma on all quadrature
    points and writes them into the appropriate functions q_sigma/q_dsigma.

    u:
        displacement field field

    In "production", a lot of the code below should be executed only once,
    outside of this function.
    """
    # prepare strain evaluation
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
    q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
    strain_expr = df.fem.Expression(eps(u), q_points)

    # Actually compute a strain matrix containing one row per cell.
    strains = strain_expr.eval(cells)
    assert strains.shape[0] == num_cells
    assert strains.shape[1] == len(q_points) * 3

    # Hookes law for plane stress
    C11 = E / (1.0 - nu * nu)
    C12 = C11 * nu
    C33 = C11 * 0.5 * (1.0 - nu)
    C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])

    # here _could_ be a loop over all quadrature points in
    # c++, mfront, numba, ...
    # For this simple case, we can evaluate it as one matrix operation.
    strain_matrix = strains.reshape((-1, 3))
    n_gauss = len(strain_matrix)

    q_sigma.x.array[:] = (strain_matrix @ C).flatten()
    q_dsigma.x.array[:] = np.tile(C.flatten(), n_gauss)


r"""
Set up 

  +---------+
/||         |->
/||         |->
/||         |-> u_bc
  o---------+
 / \
-----
"""


def left(x):
    return np.isclose(x[0], 0.0)


def right(x):
    return np.isclose(x[0], 1.0)


def origin(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))


u_bc = df.fem.Constant(mesh, 0.0)  # expression for boundary displacement

dim = mesh.topology.dim - 1
b_facets_l = df.mesh.locate_entities_boundary(mesh, dim, left)
b_facets_r = df.mesh.locate_entities_boundary(mesh, dim, right)
b_facets_o = df.mesh.locate_entities_boundary(mesh, dim - 1, origin)

b_dofs_l = df.fem.locate_dofs_topological(V.sub(0), dim, b_facets_l)
b_dofs_r = df.fem.locate_dofs_topological(V.sub(0), dim, b_facets_r)
b_dofs_o = df.fem.locate_dofs_topological(V.sub(1), dim - 1, b_facets_o)

bcs = [
    df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)),
    df.fem.dirichletbc(u_bc, b_dofs_r, V.sub(0)),
    df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_o, V.sub(1)),
]


def check_solution(u, u_bc_value):
    """
    Defines a grid of test points xs and compares the FE values u(xs)
    to the analytic displacement solution.

    ux(x,y) =       x * u_bc_value
    uy(x,y) = -nu * y * u_bc_value

    """

    def eval_function_at_points(f, points):
        """
        Evaluates `f` at `points`. Adapted from
        https://jorgensd.github.io/df-tutorial/chapter1/membrane_code.html
        """
        mesh = f.function_space.mesh
        tree = df.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        cell_candidates = df.geometry.compute_collisions(tree, points)
        colliding_cells = df.geometry.compute_colliding_cells(
            mesh, cell_candidates, points
        )

        points_on_proc = []
        cells = []
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        return f.eval(points_on_proc, cells), points_on_proc

    x_check = np.linspace(0, 1, 5)
    x, y = np.meshgrid(x_check, x_check)
    xs = np.vstack([x.flat, y.flat, np.zeros(len(x.flat))]).T
    u_fem, xs = eval_function_at_points(u, xs)

    # analytic solution
    u_ref = np.array([xs[:, 0] * u_bc_value, -nu * xs[:, 1] * u_bc_value]).T
    assert np.linalg.norm(u_fem - u_ref) < 1.0e-10


# Here, we manually build PETSc objects and simulate a quasistaic loading.
# Classes like df.fem.petsc.LinearProblem would trivialize that step. A look
# "under the hood" is quite instructive, though:
b = df.fem.petsc.create_vector(R)
A = df.fem.petsc.create_matrix(dR)

solver = PETSc.KSP().create(mesh.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

f = df.io.XDMFFile(mesh.comm, "displacements.xdmf", "w")
f.write_mesh(mesh)
ts = np.linspace(0.0, 1.0, 5)
u_bc_max = 42.0
for t in ts:
    # update value of Dirichlet BC
    u_bc.value = t * u_bc_max

    print0(f"Solving {t = :6.3f} with {u_bc.value = :6.3f}...")

    evaluate_constitutive_law(u)
    print('q', q_sigma.x.array[:])

    # update matrix (pointless in linear elasticity...)
    A.zeroEntries()
    df.fem.petsc.assemble_matrix(A, dR, bcs=bcs)
    A.assemble()

    # update residual vector
    with b.localForm() as b_local:
        b_local.set(0.0)
    df.fem.petsc.assemble_vector(b, R)
    # print(b[:])

    df.fem.apply_lifting(b, [dR], bcs=[bcs], x0=[u.vector], scale=-1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    df.fem.set_bc(b, bcs, u.vector, -1.0)
    # print(b[:])

    print('\nb', b[:])
    print('\nA', A[:,:])

    solver.setOperators(A)

    # Solve for the displacement increment du, apply it and udpate ghost values
    du = df.fem.Function(V)  # Should be outside of loop, instructive here.
    solver.solve(b, du.vector)
    u.x.array[:] -= du.x.array[:]
    u.x.scatter_forward()
    print('\nu',u.x.array[:])

    # post processing
    check_solution(u, t * u_bc_max)
    f.write_function(u, t)

# for t in np.linspace(0.0, 1.0, 5):
    # update value of Dirichlet BC
# t = 1
# u_bc.value = t * u_bc_max

# print0(f"Solving {t = :6.3f} with {u_bc.value = :6.3f}...")

# evaluate_constitutive_law(u)

# # update matrix (pointless in linear elasticity...)
# A.zeroEntries()
# df.fem.petsc.assemble_matrix(A, dR, bcs=bcs)
# A.assemble()
# print(A[:,:])

# # update residual vector
# with b.localForm() as b_local:
#     b_local.set(0.0)
# df.fem.petsc.assemble_vector(b, R)

# df.fem.apply_lifting(b, [dR], bcs=[bcs], x0=[u.vector], scale=-1.0)
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# df.fem.set_bc(b, bcs, u.vector, -1.0)
# print(u.x.array[:])

# solver.setOperators(A)

# # Solve for the displacement increment du, apply it and udpate ghost values
# du = df.fem.Function(V)  # Should be outside of loop, instructive here.
# solver.solve(b, du.vector)
# u.x.array[:] -= du.x.array[:]
# u.x.scatter_forward()

# # post processing
# check_solution(u, t * u_bc_max)
# f.write_function(u, t)
