import numba
import numpy as np

import ufl
from dolfinx import fem, mesh, io

from mpi4py import MPI
from petsc4py import PETSc

import sys
sys.path.append("../")
import fenicsx_support
import custom_assembling as ca

N = 4
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)

deg, q_deg = 1, 2
V = fem.VectorFunctionSpace(domain, ("P", deg))

# quadrature elements and function spaces
QV = ufl.VectorElement(
    "Quadrature", domain.ufl_cell(), q_deg, quad_scheme="default", dim=3
)

QT = ufl.TensorElement(
    "Quadrature",
    domain.ufl_cell(),
    q_deg,
    quad_scheme="default",
    shape=(3, 3),
)
VQV = fem.FunctionSpace(domain, QV)
VQT = fem.FunctionSpace(domain, QT)

# define functions
u_, du = ufl.TestFunction(V), ufl.TrialFunction(V)
u = fem.Function(V)
q_sigma0 = fem.Function(VQV)
q_dsigma = fem.Function(VQT)

num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
num_gauss_local = len(q_sigma0.x.array[:]) // 3
num_gauss_global = domain.comm.reduce(num_gauss_local, op=MPI.SUM, root=0)

# define form
dxm = ufl.dx(metadata={"quadrature_degree": q_deg, "quadrature_scheme": "default"})

def eps(u):
    e = ufl.sym(ufl.grad(u))
    return ufl.as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))

E, nu = 20000, 0.3

# Hookes law for plane stress
C11 = E / (1.0 - nu * nu)
C12 = C11 * nu
C33 = C11 * 0.5 * (1.0 - nu)
C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]], dtype=PETSc.ScalarType)
C_const = fem.Constant(domain, C)

def sigma(u):
    return ufl.dot(eps(u), C_const)

def get_eval(self:ca.CustomFunction):
    tabulated_eps = self.tabulated_input_expression
    n_gauss_points = len(self.input_expression.X)
    C = self.derivative.value
    local_dim = self.local_dim
    
    @numba.njit
    def eval(values, coeffs_values, constants_values, coordinates, local_index, orientation):
        epsilon_local = np.zeros(local_dim, dtype=PETSc.ScalarType)
        sigma_local = values.reshape((n_gauss_points, -1))

        tabulated_eps(ca.ffi.from_buffer(epsilon_local), 
                      ca.ffi.from_buffer(coeffs_values), 
                      ca.ffi.from_buffer(constants_values), 
                      ca.ffi.from_buffer(coordinates), ca.ffi.from_buffer(local_index), ca.ffi.from_buffer(orientation))
        
        epsilon_local = epsilon_local.reshape((n_gauss_points, -1))

        for i in range(n_gauss_points):
            sigma_local[i][:] = np.dot(C, epsilon_local[i]) 
        
        values = sigma_local.flatten()
    return eval

q_sigma = ca.CustomFunction(VQV, eps(u), C_const, get_eval)

R = ufl.inner(eps(u_), q_sigma) * dxm
dR = ufl.inner(eps(du), ufl.dot(q_sigma.derivative, eps(u_))) * dxm

R0 = ufl.inner(eps(u_), q_sigma0) * dxm
dR0 = ufl.inner(eps(du), ufl.dot(C_const, eps(u_))) * dxm

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


u_bc = fem.Constant(domain, 0.0)  # expression for boundary displacement

dim = domain.topology.dim - 1
b_facets_l = mesh.locate_entities_boundary(domain, dim, left)
b_facets_r = mesh.locate_entities_boundary(domain, dim, right)
b_facets_o = mesh.locate_entities_boundary(domain, dim - 1, origin)

b_dofs_l = fem.locate_dofs_topological(V.sub(0), dim, b_facets_l)
b_dofs_r = fem.locate_dofs_topological(V.sub(0), dim, b_facets_r)
b_dofs_o = fem.locate_dofs_topological(V.sub(1), dim - 1, b_facets_o)

bcs = [
    fem.dirichletbc(PETSc.ScalarType(0), b_dofs_l, V.sub(0)),
    fem.dirichletbc(u_bc, b_dofs_r, V.sub(0)),
    fem.dirichletbc(PETSc.ScalarType(0), b_dofs_o, V.sub(1)),
]

bc_dofs = np.concatenate((b_dofs_l, b_dofs_r, b_dofs_o))

A0 = fem.petsc.create_matrix(fem.form(dR0))
A0.zeroEntries()
b0 = fem.petsc.create_vector(fem.form(R0))
with b0.localForm() as b_local:
    b_local.set(0.0)
u0 = fem.Function(V)

q_sigma.vector.set(0)
q_sigma0.vector.set(0)

solver0 = PETSc.KSP().create(domain.comm)
solver0.setType("preonly")
solver0.getPC().setType("lu")
solver0.setOperators(A0)

# f = io.XDMFFile(domain.comm, "displacements.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
# f.write_mesh(domain)

u = fem.Function(V)
u.name = "Displacement"
my_solver = ca.CustomSolver(dR, R, u, bcs)

u_bc_max = 42.0
ts = np.linspace(0.0, 1.0, 5)
for t in ts:
    # update value of Dirichlet BC
    u_bc.value = t * u_bc_max

    print(f"Solving {t = :6.3f} with {u_bc.value = :6.3f}...")

    eps_calculated = fenicsx_support.interpolate_quadrature(eps(u0), q_deg, domain)
    strain_matrix = eps_calculated.reshape((-1, 3))
    n_gauss = len(strain_matrix) #global in the domain

    q_sigma0.x.array[:] = (strain_matrix @ C).flatten()
    # q_dsigma.x.array[:] = np.tile(C.flatten(), n_gauss)

    # update matrix (pointless in linear elasticity...)
    A0.zeroEntries()
    # update residual vector
    with b0.localForm() as b_local:
        b_local.set(0.0)

    fem.petsc.assemble_matrix(A0, fem.form(dR), bcs=bcs)
    A0.assemble()
    fem.petsc.assemble_vector(b0, fem.form(R))

    fem.apply_lifting(b0, [fem.form(dR)], bcs=[bcs], x0=[u0.vector], scale=-1.0)
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b0, bcs, u0.vector, -1.0)

    du0 = fem.Function(V)  # Should be outside of loop, instructive here.
    solver0.solve(b0, du0.vector)
    u0.x.array[:] -= du0.x.array[:]
    u0.x.scatter_forward()
    
    # Solve for the displacement increment du, apply it and udpate ghost values
    du = fem.Function(V)  # Should be outside of loop, instructive here.

    my_solver.solve(du)
    
    u.x.array[:] -= du.x.array[:]
    u.x.scatter_forward()

    q_sigma0.x.scatter_forward()
    q_sigma.x.scatter_forward()

    print(f'rank = {MPI.COMM_WORLD.rank} u - u0\n {np.linalg.norm(u.x.array[:] - u0.x.array[:])} \n')
    assert np.linalg.norm(u.x.array[:] - u0.x.array[:]) < 1.0e-10
    print(f'rank = {MPI.COMM_WORLD.rank} q_sigma - q_sigma0\n {np.linalg.norm(q_sigma.x.array[:] - q_sigma0.x.array[:])} \n')

    # print(f'{q_sigma.x.array}')
    # print(f'{q_sigma0.x.array}')
    # post processing
    # f.write_function(u, t)

# f.close()
