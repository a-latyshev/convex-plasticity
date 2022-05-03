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
import time
import basix

import os 

import sys
sys.path.append("../")
import fenicsx_support

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

def get_kernel(form:ufl.form.Form):
    """Extracts c-function representation of the form
    
    Args:
        form: required linear or bilinear form
    
    Returns:
        kernel:
            c-function representation of `form`. It is being used for local-element calculations of the form.
            It has a signature `void(*)(double *, double *, double *, double *, int *, uint8_t *)`.

            The 1st argument (`form_local`) takes an array where local form values will be collected
            
            The 2nd argument (`coeff_local`) takes an array of local values of all coefficients presenting in the form.
            
            The 3rd argument (`constants_local`) takes an array of contants values being used in the form
            
            The 4th argument (`geometry`) takes a 3*3 array of cell nodes coordinates.
            
            The 5th argument (`entity_local_index`) takes an array of entity_local_index.
            
            The 6th argument (`permutation`) is an elements permutation option.
    """
    domain = form.ufl_domain().ufl_cargo()
    ufcx_form, _, _ = ffcx_jit(domain.comm, form, form_compiler_params={"scalar_type": "double"})
    kernel = ufcx_form.integrals(0)[0].tabulate_tensor_float64
    return kernel

class CustomFunction(fem.Function):
    """Expands the standard fem.Function

    On the current dolfinx version 0.3.1.0 we use `fem.Function` variable for `g` function defined as follows:  
    .. math:: \\int g \\dot a(u,v) dx \\text{or} \\int g v dx 
    From here we don't know an exact expression of `g`, which we need to create separately. 
    `CustomFunction` class contains this information and can be used to assemble `g` locally using `custom_assembling`.

    Attributes:
        local_dim: 
            local number of dofs on an element
        ufl_operand: 
            ufl representation of the function. It is assumed to contain a `fem.function.Function` in its argument. 
            For example: `g = eps(u)`, where `eps` is an ufl expression and u is `fem.function.Function`.
        derivate:
            derivative of the function. It is assumed to have a type of `np.array(... , dtype=PETSc.ScalarType)`
        tabulated_expression:
            c-function representation of `ufl_operand`. It is being used for local calculations of the function.
            It has a signature `void(*)(double *, double *, double *, double *, int *, uint8_t *)`.
            
            The 1st argument (`g_local`) takes an array where local values of `g` will be collected. Allocate `local_dim` elements!
            
            The 2nd argument (`coeff_local`) takes an array of local values of the operand argument.
            
            The 3rd argument (`constants_local`) takes an array of contants values being used in `ulf_operand`
            
            The 4th argument (`geometry`) takes a 3*3 array of cell nodes coordinates.
            
            The 5th argument (`entity_local_index`) takes an array of entity_local_index.
            
            The 6th argument (`permutation`) is an elements permutation option.
            
    """
    def __init__(self, V: fem.FunctionSpace, ufl_operand, derivative: np.ndarray):
        """Inits `CustomFunction`"""

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

from typing import Union

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

@numba.cfunc(c_signature, nopython=True)
def dummy_tabulated(b_, w_, c_, coords_, local_index, orientation):
    """Simulates a typical tabulated function"""
    pass

def extract_constants(ufl_expression) -> np.ndarray:
    """Extracts and puts together all values of all `fem.function.Constant` presenting in `ufl_expression`

    Returns:
        constants_values: numpy flatten array with values of all constants. Values are sorted in accordance with the order 
        of constants in `ufl_expression`. 
    """
    constants = ufl.algorithms.analysis.extract_constants(ufl_expression)
    constants_values = np.concatenate([const.value.flatten() for const in constants]) if len(constants) !=0 else np.zeros(0, dtype=PETSc.ScalarType)
    return constants_values

def extract_coefficients_and_constants(form: ufl.form.Form) -> Union[np.ndarray, list, list, np.ndarray]:
    """Extracts coefficients and constants of a given form and puts their values all together 

    Args:
        form: linear or bilinear form
    
    Returns: an union of:
        N_coeffs_values_local: an array containing number of local values of all form coefficients
        tabulated_coeffs: a list with coefficients c-function of their ufl-expressions
        coeffs_contants: a list with flatten arrays of all constants presenting in the form coefficient
        constants_values: a flatten array with values of all constants of the form

    Note:
        It is assumed that all form coefficients are `CustomFunction`, which have their own constants `fem.function.Constant`.
        and in the same time its haven't their own coefficients.

    """
    N_coeffs = len(form.coefficients())
    N_coeffs_values_local = np.zeros(N_coeffs)
    tabulated_coeffs = []
    coeffs_constants = []
    for i, coeff in enumerate(form.coefficients()):
        N_coeffs_values_local[i] = coeff.local_dim
        tabulated_coeffs.append(coeff.tabulated_expression)
        coeffs_constants.append(extract_constants(coeff.ufl_operand)) #whatif not all of them are Custom

    #Numba doesn't like empty lists, so we have to fill it with something 
    if len(tabulated_coeffs) == 0 :
        tabulated_coeffs.append(dummy_tabulated)

    if len(coeffs_constants) == 0 :
        coeffs_constants.append(np.array([-1], dtype=PETSc.ScalarType))

    constants_values = extract_constants(form)
            
    return N_coeffs_values_local, tabulated_coeffs, coeffs_constants, constants_values

# %%
N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A = extract_coefficients_and_constants(dR)
N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b = extract_coefficients_and_constants(R)

# %%
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

#Numba doesn't like python lists. It will be deprecated, but I don't see any other possibility to send into a numba-function a set of tabulated functions.
#In theory it's possible using this https://github.com/numba/numba/issues/2542
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# %%
@numba.njit
def get_local_dofs_bc(pos, bc_dofs):
    """Searches bc dofs locally"""
    dofs_bc = []
    intersec = np.intersect1d(bc_dofs, pos)
    if len(intersec != 0):
        for i, dof in enumerate(pos):
            if dof in intersec:
                dofs_bc.append(i)
    return dofs_bc

@numba.njit
def assemble_ufc(A, b, u, geo_dofs, coords, dofmap, num_owned_cells, bc_dofs, N_dofs_element,
                 N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A, 
                 N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b,
                 kernel_A, kernel_b,
                 mode=PETSc.InsertMode.ADD_VALUES):
    """Assembles the matrix A and the vector b using UFC approach 

    As it is a numba-function only numpy arrays, c-functions and other trivial objects and methodes are allowed.
    It's assumed that the matrix and the rhs-vector have coefficients which haven't their own ones.

    Args:
        A:
            a handle of global matrix
        b:
            a numpy array of global rhs-vector values 
        u:
            a numpy array of global solution values 
        geo_dofs:
            geometrical dofs of mesh (elements nodes)
        coords:
            global nodes coordinates
        dofmap:
            a full dof-map of solution function space 
        num_owned_cells:
            a number of cells on a current process (without ghost elements)
        bc_dofs:
            a numpy array containing Dirichlet BC dofs 
        ...

        mode:
            a mode of matrix assembling      
    """
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))

    N_coeffs_b = N_coeffs_values_local_b.size
    N_coeffs_A = N_coeffs_values_local_A.size

    coeffs_A = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_A == 0 else \
               np.zeros(int(np.sum(N_coeffs_values_local_A)), dtype=PETSc.ScalarType)
               
    coeffs_b = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_b == 0 else \
               np.zeros(int(np.sum(N_coeffs_values_local_b)), dtype=PETSc.ScalarType)

    b_local = np.zeros(N_dofs_element, dtype=PETSc.ScalarType)
    A_local = np.zeros((N_dofs_element, N_dofs_element), dtype=PETSc.ScalarType)
    
    for cell in range(num_owned_cells):
        pos = rows = cols = dofmap[cell]
        geometry[:] = coords[geo_dofs[cell], :]
        u_local = u[pos]

        b_local.fill(0.)
        A_local.fill(0.)
        coeffs_A.fill(0.)
        coeffs_b.fill(0.)

        for i in range(N_coeffs_b):
            tabulated_coeffs_b[i](ffi.from_buffer(coeffs_b[i*N_coeffs_values_local_b[i]:(i+1)*N_coeffs_values_local_b[i]]), 
                                  ffi.from_buffer(u_local), 
                                  ffi.from_buffer(coeffs_constants_b[i]), 
                                  ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        # for i in range(N_coeffs_A):
        #     tabulated_coeffs_A[i](ffi.from_buffer(coeffs_A[i*N_coeffs_values_local_A[i]:(i+1)*N_coeffs_values_local_A[i]]), 
        #                           ffi.from_buffer(u_local), 
        #                           ffi.from_buffer(coeffs_constants_A[i]), 
        #                           ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        kernel_b(ffi.from_buffer(b_local), 
                 ffi.from_buffer(coeffs_b),
                 ffi.from_buffer(constants_values_b),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        b[pos] += b_local
        
        kernel_A(ffi.from_buffer(A_local), 
                 ffi.from_buffer(coeffs_A),
                 ffi.from_buffer(constants_values_A),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))
               
        # Taking into account Dirichlet BCs
        # for i in get_local_dofs_bc(pos, bc_dofs): 
        #     A_local[i,:] = 0
        #     A_local[:,i] = 0

        MatSetValues_ctypes(A, N_dofs_element, rows.ctypes, N_dofs_element, cols.ctypes, A_local.ctypes, mode)


# %%
def get_topological_dofmap(V:fem.function.FunctionSpace):
    """Makes a topological dofmap for a vector function space
    Args:
        V: vector function space
    Returns:
        dofmap_topological: a numpy array of size (cells number, number of vector function space dofs)
    """
    N_dofs_element = V.element.space_dimension
    len_funtion_space = V.dofmap.index_map_bs
    dofmap = V.dofmap.list.array.reshape(num_cells, int(N_dofs_element/len_funtion_space))
    dofmap_topological = (len_funtion_space*np.repeat(dofmap, len_funtion_space).reshape(-1, len_funtion_space) + np.arange(len_funtion_space)).reshape(-1, N_dofs_element)
    return dofmap_topological

map_c = domain.topology.index_map(domain.topology.dim)
num_owned_cells = map_c.size_local
num_cells = num_owned_cells + map_c.num_ghosts
N_dofs_element = V.element.space_dimension
len_funtion_space = V.dofmap.index_map_bs # == V.num_sub_spaces for VECtor funcion spaces 
x_dofs = domain.geometry.dofmap.array.reshape(num_cells, 3)
x = domain.geometry.x
dofmap = V.dofmap.list.array.reshape(num_cells, int(N_dofs_element/len_funtion_space)).astype(np.dtype(PETSc.IntType))

#This dofmap takes into account dofs of the vector field
dofmap_topological = get_topological_dofmap(V).astype(np.dtype(PETSc.IntType))

# %%
# eps_calculated = fenicsx_support.interpolate_quadrature(eps(u), q_deg, domain)
# # print(eps_calculated.shape) # num_cells * len(eps) * len(n_gauls_on_element)
# strain_matrix = eps_calculated.reshape((-1, 3))
# n_gauss = len(strain_matrix) #global in the domain


u = fem.Function(V)    

kernel_A = get_kernel(dR)
kernel_b = get_kernel(R)

# %%
A = fem.petsc.create_matrix(fem.form(dR))
A.zeroEntries()
b = fem.petsc.create_vector(fem.form(R))
with b.localForm() as b_local:
    b_local.set(0.0)

solver = PETSc.KSP().create(domain.comm)
solver.setType("preonly")
solver.getPC().setType("lu")

f = io.XDMFFile(domain.comm, "displacements.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5)
f.write_mesh(domain)
u.name = "Displacement"

u = fem.Function(V)
q_sigma.vector.set(0)
q_dsigma.vector.set(0)

u_bc_max = 42.0
ts = np.linspace(0.0, 1.0, 5)


for t in ts:
    # update value of Dirichlet BC
    u_bc.value = t * u_bc_max

    print(f"Solving {t = :6.3f} with {u_bc.value = :6.3f}...")

    # eps_calculated = fenicsx_support.interpolate_quadrature(eps(u), q_deg, domain)
    # # print(eps_calculated.shape) # num_cells * len(eps) * len(n_gauls_on_element)
    # strain_matrix = eps_calculated.reshape((-1, 3))
    # n_gauss = len(strain_matrix) #global in the domain

    # q_sigma.x.array[:] = (strain_matrix @ C).flatten()
    # q_dsigma.x.array[:] = np.tile(C.flatten(), n_gauss)

    # q_sigma_values = q_sigma.x.array[:] 
    # q_dsigma_values = np.tile(C.flatten(), n_gauss)

    # update matrix (pointless in linear elasticity...)
    A.zeroEntries()
    # update residual vector
    with b.localForm() as b_local:
        b_local.set(0.0)

    # start = time.time()
    assemble_ufc(A.handle, b.array, u.x.array, x_dofs, x, dofmap_topological, num_owned_cells, bc_dofs, N_dofs_element,
                 N_coeffs_values_local_A, tabulated_coeffs_A, coeffs_constants_A, constants_values_A, 
                 N_coeffs_values_local_b, tabulated_coeffs_b, coeffs_constants_b, constants_values_b,
                 kernel_A, kernel_b)
    # print(f'iter = {t}, time = {time.time() - start}')
    A.assemble()
    # for i in bc_dofs:
    #     A.setValueLocal(i, i, 1)
    # A.zeroRowsColumnsL(bc_dofs, 1.)
    A.zeroRowsColumnsLocal(bc_dofs, 1.)
    A.assemble()

    fem.apply_lifting(b, [fem.form(dR)], bcs=[bcs], x0=[u.vector], scale=-1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b, bcs, u.vector, -1.0)

    solver.setOperators(A)

    # Solve for the displacement increment du, apply it and udpate ghost values
    du = fem.Function(V)  # Should be outside of loop, instructive here.
    solver.solve(b, du.vector)
    u.x.array[:] -= du.x.array[:]
    u.x.scatter_forward()
    print(f'rank = {MPI.COMM_WORLD.rank}, u = \n {u.x.array[:]}')

    # post processing
    f.write_function(u, t)


f.close()
