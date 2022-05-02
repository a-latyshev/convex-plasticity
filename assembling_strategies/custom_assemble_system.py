# %%
import cffi
import numba
import numpy as np

import ufl
from dolfinx.jit import ffcx_jit
from dolfinx import fem, mesh, io

import ctypes
import ctypes.util
import petsc4py.lib
from mpi4py import MPI
from petsc4py import get_config as PETSc_get_config
from petsc4py import PETSc

import basix

import sys
sys.path.append("../")
import fenicsx_support
import os

# %%
# Get details of PETSc install
petsc_dir = PETSc_get_config()['PETSC_DIR']
petsc_arch = petsc4py.lib.getPathArchPETSc()[1]

# Get PETSc int and scalar types
scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RuntimeError(f"Cannot translate PETSc index size into a C type, index_size: {index_size}.")

if scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        f"Cannot translate PETSc scalar type to a C type, complex: {complex} size: {scalar_size}.")

# Load PETSc library via ctypes
petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_ctypes = ctypes.CDLL(petsc_lib_name)
else:
    try:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

# Get the PETSc MatSetValuesLocal function via ctypes
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
del petsc_lib_ctypes

# %%
ffi = cffi.FFI()

def getKernel(domain, form):
    ufcx_form, _, _ = ffcx_jit(domain.comm, form, form_compiler_params={"scalar_type": "double"})
    kernel = ufcx_form.integrals(0)[0].tabulate_tensor_float64
    return kernel

# %%
class CustomFunction(fem.Function):
    def __init__(self, V: fem.FunctionSpace, ufl_operand, derivative: np.ndarray):
        assert derivative.dtype == PETSc.ScalarType
        
        super().__init__(V)
        self.local_dim = V.element.space_dimension
        self.ufl_operand = ufl_operand
        self.derivative = derivative
        
        if V._ufl_element.family() == 'Quadrature':
            basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
            gauss_points, _ = basix.make_quadrature(basix_celltype, V._ufl_element.degree())
            self.expression = fem.Expression(self.ufl_operand, gauss_points)
        else:
            self.expression = fem.Expression(self.ufl_operand, V.element.interpolation_points)

        self.tabulated_expression = self.expression._ufcx_expression.tabulate_tensor_float64

# %%
N = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)#, mesh.CellType.quadrilateral

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
q_sigma = fem.Function(VQV)
q_dsigma = fem.Function(VQT)

num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
num_gauss_local = len(q_sigma.x.array[:]) // 3
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

q_sigma = CustomFunction(VQV, sigma(u), C)

R = ufl.inner(eps(u_), q_sigma) * dxm
dR = ufl.inner(eps(du), ufl.dot(C_const, eps(u_))) * dxm

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



# %%
# Unpack mesh and dofmap data
num_owned_cells = domain.topology.index_map(domain.topology.dim).size_local
num_cells = num_owned_cells + domain.topology.index_map(domain.topology.dim).num_ghosts
x_dofs = domain.geometry.dofmap.array.reshape(num_cells, 3)
x = domain.geometry.x
dofmap = V.dofmap.list.array.reshape(num_cells, 3).astype(np.dtype(PETSc.IntType))


# %%
map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
N_dofs_element = V.element.space_dimension
N_sub_spaces = V.num_sub_spaces # == V.dofmap.index_map_bs
dofmap = V.dofmap.list.array.reshape(num_cells, int(N_dofs_element/N_sub_spaces))

# print(f'rank = {MPI.COMM_WORLD.rank}, num of cells {num_cells}')
# print('global n cells = ', map_c.size_global)
# print('u shape = ', u.x.array.shape)
# print('q shape = ', q_sigma.x.array.shape)

#This dofmap takes into account dofs of the vector field
dofmap_topological = (N_sub_spaces*np.repeat(dofmap, N_sub_spaces).reshape(-1, N_sub_spaces) + np.arange(N_sub_spaces)).reshape(-1, N_dofs_element).astype(np.dtype(PETSc.IntType)) 

# print(MPI.COMM_WORLD.rank, num_owned_cells,'\n', dofmap)
# print(MPI.COMM_WORLD.rank, dofmap_topological)
print('dofs:', bc_dofs)
print(f'rank = {MPI.COMM_WORLD.rank}, num of cells {num_cells}')
print('global n cells = ', map_c.size_global)
print('u shape = ', u.x.array.shape)
print('q shape = ', q_sigma.x.array.shape)
print('dofs:', bc_dofs)
# %%
@numba.njit
def get_dofs_bc(pos, bc_dofs):
    dofs_bc = []
    for i, dof in enumerate(pos):
        if dof in bc_dofs:
            dofs_bc.append(i)
    return dofs_bc

# %%
basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
eps_expr = fem.Expression(eps(u), q_points)
eps_tabulated = eps_expr._ufcx_expression.tabulate_tensor_float64

# %%
def extractConstants(ufl_expression) -> np.ndarray:
    constants = ufl.algorithms.analysis.extract_constants(ufl_expression)
    constants_values = np.concatenate([const.value.flatten() for const in constants]) if len(constants) !=0 else np.zeros(0, dtype=PETSc.ScalarType)
    return constants_values

# %%
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

@numba.cfunc(c_signature, nopython=True)
def dummy_tabulated(b_, w_, c_, coords_, local_index, orientation):
    pass

# %%
from typing import Union

def extracteCoefficientsAndConstants(form: ufl.form.Form) -> Union[np.ndarray, list, list, np.ndarray]:
    N_coeffs = len(form.coefficients())
    N_coeffs_values_local = np.zeros(N_coeffs)
    tabulated_coeffs = []
    coeffs_constants = []
    for i, coeff in enumerate(form.coefficients()):
        N_coeffs_values_local[i] = coeff.local_dim
        tabulated_coeffs.append(coeff.tabulated_expression)
        coeffs_constants.append(extractConstants(coeff.ufl_operand)) #whatif not all of them are Custom

    if len(tabulated_coeffs) == 0 :
        tabulated_coeffs.append(dummy_tabulated)

    if len(coeffs_constants) == 0 :
        coeffs_constants.append(np.array([-1], dtype=PETSc.ScalarType))

    constants_values = extractConstants(form)
            
    return N_coeffs_values_local, tabulated_coeffs, coeffs_constants, constants_values

# %%
N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A = extracteCoefficientsAndConstants(dR)
N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b = extracteCoefficientsAndConstants(R)

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# %%
eps_calculated = fenicsx_support.interpolate_quadrature(eps(u), q_deg, domain)
# print(eps_calculated.shape) # num_cells * len(eps) * len(n_gauls_on_element)
strain_matrix = eps_calculated.reshape((-1, 3))
n_gauss = len(strain_matrix) #global in the domain

q_sigma.vector.set(0)
q_dsigma.vector.set(0)

q_sigma.x.array[:] = (strain_matrix @ C).flatten()
q_dsigma.x.array[:] = np.tile(C.flatten(), n_gauss)

q_dsigma_values = np.tile(C.flatten(), n_gauss)

u_bc_max = 42.0
t = 1
u_bc.value = t * u_bc_max

A1 = fem.petsc.create_matrix(fem.form(dR))
A1.zeroEntries()
fem.petsc.assemble_matrix(A1, fem.form(dR), bcs=bcs)
A1.assemble()

b1 = fem.petsc.create_vector(fem.form(R))
with b1.localForm() as b_local:
    b_local.set(0.0)
fem.petsc.assemble_vector(b1, fem.form(R))
fem.apply_lifting(b1, [fem.form(dR)], bcs=[bcs], x0=[u.vector], scale=-1.0)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b1, bcs, u.vector, -1.0)

A = fem.petsc.create_matrix(fem.form(dR))
A.zeroEntries()
b = fem.petsc.create_vector(fem.form(R))
with b.localForm() as b_local:
    b_local.set(0.0)

u = fem.Function(V)

map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
N_dofs_element = V.element.space_dimension
N_sub_spaces = V.num_sub_spaces # == V.dofmap.index_map_bs
dofmap = V.dofmap.list.array.reshape(num_cells, int(N_dofs_element/N_sub_spaces))

#This dofmap takes into account dofs of the vector field
dofmap_tmp = (N_sub_spaces*np.repeat(dofmap, N_sub_spaces).reshape(-1, N_sub_spaces) + np.arange(N_sub_spaces)).reshape(-1, N_dofs_element).astype(np.dtype(PETSc.IntType))        

# print(f'dofmap = \n{dofmap}, \n dofmap_vec = \n{dofmap_tmp}')

# dofs_sub0 = V.sub(0).dofmap.list.array.reshape(num_cells, int(N_dofs_element/N_sub_spaces))
# print(dofs_sub0)
# dofs_sub1 = V.sub(1).dofmap.list.array.reshape(num_cells, int(N_dofs_element/N_sub_spaces))
# print(dofs_sub1)

# dofmap_top2 = np.zeros_like(dofmap_tmp)

# for i in np.arange(dofmap_tmp.shape[0]):
#     for j in np.arange(3):
#         dofmap_top2[i,j*2] = dofs_sub0[i][j] 
#         dofmap_top2[i,j*2 + 1] = dofs_sub1[i][j]
# print(dofmap_top2)

kernel_A = getKernel(domain, dR)
kernel_b = getKernel(domain, R)


@numba.njit
def assemble_ufc(A, b, u_values, q_sigma_values, dofs, coords, dofmap, num_cells,
                 N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A, 
                 N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b,
                 kernel_A, kernel_b,
                 mode):
    # num_cells = len(dofmap) # in parallel ?            
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))
    
    coeffs_A = np.full(27, 1, dtype=PETSc.ScalarType) 
    eps_loc = np.zeros(9, dtype=PETSc.ScalarType) #eps has 3 components * 3 gauss points
    coeffs_b = np.full(9, 1, dtype=PETSc.ScalarType) #In fact coeffs_b = q_sigma_local = C @ eps_local
    constants = C #np.array(C, dtype=PETSc.ScalarType)

    N_coeffs_b = N_coeffs_values_local_b.size
    # N_coeffs_A = N_coeffs_values_local_A.size

    # coeffs_A = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_A == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_A)), dtype=PETSc.ScalarType)
               
    # coeffs_b = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_b == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_b)), dtype=PETSc.ScalarType)

    b_local = np.zeros(N_dofs_element, dtype=PETSc.ScalarType)
    A_local = np.zeros((N_dofs_element, N_dofs_element), dtype=PETSc.ScalarType)
    
    for cell in range(num_cells):
        pos = rows = cols = dofmap[cell]

        geometry[:] = coords[dofs[cell],:]
        b_local.fill(0.)
        A_local.fill(0.)
        u_local = u_values[pos]

        coeffs_A.fill(0.)
        coeffs_b.fill(0.)
        # coeffs_b = q_sigma_values.reshape(num_cells, -1)[cell]
        
        for i in np.arange(N_coeffs_b):
            tabulated_coeffs_b[i](ffi.from_buffer(coeffs_b[i*N_coeffs_values_local_b[i]:(i+1)*N_coeffs_values_local_b[i]]), 
                                  ffi.from_buffer(u_local), 
                                  ffi.from_buffer(coeffs_constants_b[i]), 
                                  ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        # for i in np.arange(N_coeffs_A):
        #     tabulated_coeffs_A[i](ffi.from_buffer(coeffs_A[i*N_coeffs_values_local_A[i]:(i+1)*N_coeffs_values_local_A[i]]), 
        #                         ffi.from_buffer(u_local), 
        #                         ffi.from_buffer(coeffs_constants_A[i]), 
        #                         ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        # q_sigma_local = (eps_loc.reshape((-1, 3)) @ C).flatten()
        # print('\ncoeffs_b =\n', coeffs_b)
        # print('eps_calculated = \n', eps_calculated.reshape(num_cells, -1)[cell])

        kernel_b(ffi.from_buffer(b_local), 
                 ffi.from_buffer(coeffs_b),
                 ffi.from_buffer(constants),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        b[pos] += b_local
        
        kernel_A(ffi.from_buffer(A_local), 
                 ffi.from_buffer(coeffs_A),
                 ffi.from_buffer(constants),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))
               
        # Taking into account Dirichlet BCs
        for i in get_dofs_bc(pos, bc_dofs):
            A_local[i,:] = 0
            A_local[:,i] = 0

        MatSetValues_ctypes(A, N_dofs_element, rows.ctypes, N_dofs_element, cols.ctypes, A_local.ctypes, mode)

# assemble_ufc(A.handle, b.array, u.x.array, q_sigma.x.array, x_dofs, x, dofmap_topological, num_owned_cells, 
#              N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A,  
#              N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b, 
#              kernel_A, kernel_b, PETSc.InsertMode.ADD_VALUES)
# A.assemble()

# # Taking into account Dirichlet BCs
# for i in bc_dofs:
#     A[i,i] = 1
# A.assemble()
# # print(MPI.COMM_WORLD.rank, A.getDenseArray())

# fem.apply_lifting(b, [fem.form(dR)], bcs=[bcs], x0=[u.vector], scale=-1.0)
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# fem.set_bc(b, bcs, u.vector, -1.0)

# print('\n', A[:,:] - A1[:,:])
# print(b[:] - b1[:])

# %%
A = fem.petsc.create_matrix(fem.form(dR))
A.zeroEntries()
b = fem.petsc.create_vector(fem.form(R))
with b.localForm() as b_local:
    b_local.set(0.0)

A0 = fem.petsc.create_matrix(fem.form(dR))
A0.zeroEntries()
b0 = fem.petsc.create_vector(fem.form(R))
with b0.localForm() as b_local:
    b_local.set(0.0)

solver = PETSc.KSP().create(domain.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

f = io.XDMFFile(domain.comm, "displacements.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
f.write_mesh(domain)

u = fem.Function(V)
q_sigma.vector.set(0)
q_dsigma.vector.set(0)

u_bc_max = 42.0
ts = np.linspace(0.0, 1.0, 5)
for t in ts:
    # update value of Dirichlet BC
    u_bc.value = t * u_bc_max

    print(f"Solving {t = :6.3f} with {u_bc.value = :6.3f}...")

    eps_calculated = fenicsx_support.interpolate_quadrature(eps(u), q_deg, domain)
    # print(eps_calculated.shape) # num_cells * len(eps) * len(n_gauls_on_element)
    strain_matrix = eps_calculated.reshape((-1, 3))
    n_gauss = len(strain_matrix) #global in the domain

    q_sigma.x.array[:] = (strain_matrix @ C).flatten()
    # q_sigma.x.array[:] = np.full_like(q_sigma.x.array[:], 1)

    q_dsigma.x.array[:] = np.tile(C.flatten(), n_gauss)
    q_sigma.x.scatter_forward()
    q_dsigma.x.scatter_forward()

    q_sigma_values = q_sigma.x.array[:] 
    q_dsigma_values = np.tile(C.flatten(), n_gauss)

    # print('q', q_sigma.x.array[:].reshape(-1, 3))

    A0.zeroEntries()
    with b0.localForm() as b_local:
        b_local.set(0.0)

    fem.petsc.assemble_matrix(A0, fem.form(dR), bcs=bcs)
    A0.assemble()

    fem.petsc.assemble_vector(b0, fem.form(R))
    fem.apply_lifting(b0, [fem.form(dR)], bcs=[bcs], x0=[u.vector], scale=-1.0)
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b0, bcs, u.vector, -1.0)

    A.zeroEntries()
    with b.localForm() as b_local:
        b_local.set(0.0)

    assemble_ufc(A.handle, b.array, u.x.array, q_sigma.x.array, x_dofs, x, dofmap_tmp, num_owned_cells, 
                 N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A, 
                 N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b,
                 kernel_A, kernel_b, PETSc.InsertMode.ADD_VALUES)

    A.assemble()
    for i in bc_dofs:
        # A[i,i] = 1
        A.setValueLocal(i, i, 1)
    A.assemble()

    fem.apply_lifting(b, [fem.form(dR)], bcs=[bcs], x0=[u.vector], scale=-1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs, u.vector, -1.0)

    ai, aj, av = A.getValuesCSR()
    a0i, a0j, a0v = A0.getValuesCSR()
    # print(a0i, '\n', a0j)
    # print(a0i.shape, a0j.shape, a0v.shape)
    # print(f'rank = {MPI.COMM_WORLD.rank}, real = \n{a0v}')
    # print(f'rank = {MPI.COMM_WORLD.rank}, calcul = \n{av}')
    # print(f'rank = {MPI.COMM_WORLD.rank}, diff = \n{(av - a0v)}')

    # print(f'rank = {MPI.COMM_WORLD.rank}, real = \n{b0.array[:]}')
    # print(f'rank = {MPI.COMM_WORLD.rank}, calcul = \n{b.array[:]}')

    solver.setOperators(A0)
    solver.setOperators(A)

    # Solve for the displacement increment du, apply it and udpate ghost values
    du = fem.Function(V)  # Should be outside of loop, instructive here.
    solver.solve(b, du.vector)
    u.x.array[:] -= du.x.array[:]
    u.x.scatter_forward()
    print('\nu', u.x.array[:])


    # print(u.vector.size, u.vector.sizes, u.x.array.shape)
    # post processing
    f.write_function(u, t)

f.close()

# %%



