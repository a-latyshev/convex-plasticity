# %%
from pathlib import Path

import cffi
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np

import ufl
from dolfinx.cpp.fem import Form_complex128, Form_float64
# from dolfinx.fem import (Function, FunctionSpace, IntegralType, dirichletbc,
#                          form, locate_dofs_topological)
# from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
#                                set_bc)
from dolfinx.jit import ffcx_jit
from dolfinx import fem, mesh, io

from mpi4py import MPI
from petsc4py import PETSc

# %%
L = 1
W = 0.1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
mu = 1
lambda_ = beta
g = gamma

# Create mesh and define function space
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (L, W)), n=(64, 16),
                            cell_type=mesh.CellType.triangle,)


# %%
Se = ufl.TensorElement("DG", domain.ufl_cell(), 1, symmetry=True)
Ue = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)

S = fem.FunctionSpace(domain, Se)
U = fem.FunctionSpace(domain, Ue)
CG1 = fem.FunctionSpace(domain, ('CG', 1))
CG2 = fem.FunctionSpace(domain, ('CG', 2))

# %%
# Get local dofmap sizes for later local tensor tabulations
Ssize = S.element.space_dimension
Usize = U.element.space_dimension

sigma, tau = ufl.TrialFunction(S), ufl.TestFunction(S)
u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
mu = fem.Function(CG1)
lambda_ = fem.Function(CG1)

mu.x.array[:] = np.full(len(mu.x.array), 1)
lambda_.x.array[:] = np.full(len(lambda_.x.array), beta)

# %%
# mu = 1
# lambda_ = beta

# mu = fem.Function(CG1)
# lambda_ = fem.Function(CG1)

# mu.x.array[:] = np.full(len(mu.x.array), 1)
# lambda_.x.array[:] = np.full(len(lambda_.x.array), beta)

# mu.x.scatter_forward()
# lambda_.x.scatter_forward()

# %%
dxm = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": 2, "quadrature_scheme": "default"},
)

left_facets = mesh.locate_entities_boundary(domain, dim=1,
                                       marker=lambda x: np.isclose(x[0], 0.0))

dofs = fem.locate_dofs_topological(V=U, entity_dim=1, entities=left_facets)
bc = fem.dirichletbc(value=fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(0))), dofs=dofs, V=U)                            

# %%
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_*ufl.div(u)*ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

# Define variational problem
f = fem.Constant(domain, (PETSc.ScalarType(0), PETSc.ScalarType(-rho*g)))
b = ufl.inner(f, v)*ufl.dx

a00 = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

# %%
# JIT compile individual blocks tabulation kernels
nptype = "complex128" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "float64"
ffcxtype = "double _Complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "double"
ufcx_form00, _, _ = ffcx_jit(domain.comm, a00, form_compiler_params={"scalar_type": ffcxtype})
kernel00 = getattr(ufcx_form00.integrals(0)[0], f"tabulate_tensor_{nptype}")

ffi = cffi.FFI()
cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

# %%
@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    # Prepare target condensed local elem tensor
    A = numba.carray(A_, (Usize, Usize), dtype=PETSc.ScalarType)
    # Tabulate all sub blocks locally
    A00 = np.zeros((Usize, Usize), dtype=PETSc.ScalarType)
    kernel00(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index, permutation)
    
    A[:, :] = A00

# %%
# Prepare a Form with a condensed tabulation kernel
Form = Form_float64 if PETSc.ScalarType == np.float64 else Form_complex128

integrals = {fem.IntegralType.cell: ([(-1, tabulate_condensed_tensor_A.address)], None)}
a_cond = Form([U._cpp_object, U._cpp_object], integrals, [], [], False, None)

A_cond = fem.petsc.assemble_matrix(a_cond, bcs=[bc])
A_cond.assemble()

b_assembled = fem.petsc.assemble_vector(fem.form(b))
fem.petsc.apply_lifting(b_assembled, [a_cond], bcs=[[bc]])
b_assembled.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b_assembled, [bc])

uc = fem.Function(U)
solver = PETSc.KSP().create(A_cond.getComm())
solver.setOperators(A_cond)

# It gives a different result, if we remove next two lines
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

solver.solve(b_assembled, uc.vector)
uc.x.scatter_forward()

# %%
problem = fem.petsc.LinearProblem(a00, b, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# %%
uh2 = fem.Function(U)

A = fem.petsc.assemble_matrix(fem.form(a00), bcs=[bc])
A.assemble()
B = fem.petsc.assemble_vector(fem.form(b))

# A = fem.petsc.create_matrix(fem.form(a00))
# B = fem.petsc.create_vector(fem.form(b))

# A.zeroEntries()
# fem.petsc.assemble_matrix(A, fem.form(a00), bcs=[bc])
# A.assemble()

# with B.localForm() as B_local:
#     B_local.set(0.0)
# fem.petsc.assemble_vector(B, fem.form(b))

fem.apply_lifting(B, [fem.form(a00)], bcs=[[bc]])
B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(B, [bc])

solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.solve(B, uh2.vector)

uh2.x.scatter_forward()

# %%
uc.name = "Displacement"
uh.name = "linear solver"
uh2.name = "manual solver"

with io.XDMFFile(MPI.COMM_WORLD, "solution_0.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(domain)

with io.XDMFFile(MPI.COMM_WORLD, "solution_0.xdmf", "a", encoding=io.XDMFFile.Encoding.HDF5) as file:
    file.write_function(uc)
    file.write_function(uh)
    file.write_function(uh2)

# %%



