import cffi
import numba
import numpy as np

import ufl
from dolfinx.jit import ffcx_jit
from dolfinx import fem, mesh

import ctypes
import ctypes.util
import petsc4py.lib
from mpi4py import MPI
from petsc4py import get_config as PETSc_get_config
from petsc4py import PETSc

import typing
import basix
import os 

from dolfinx import la

from numba.core.errors import NumbaPendingDeprecationWarning
import warnings
#Numba doesn't like python lists. It will be deprecated, but I don't see any other possibility to send into a numba-function a set of tabulated functions.
#In theory it's possible using this https://github.com/numba/numba/issues/2542
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


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

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

def get_dummy_x(V: fem.FunctionSpace) -> la.vector:
    dummy_domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
    dummy_V = fem.FunctionSpace(dummy_domain, V._ufl_element)
    return la.vector(dummy_V.dofmap.index_map, dummy_V.dofmap.index_map_bs)

class DummyFunction(fem.Function): #or ConstantFunction?
    def __init__(self, V: fem.FunctionSpace, name: typing.Optional[str] = None):
        super().__init__(V=V, x=get_dummy_x(V), name=name)
        self.value = self.x.array.reshape((2, -1))[0]
        self.shape = V._ufl_element.reference_value_shape()
    
    def fill(self, value: np.ndarray):
        self.value[:] = value

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
    def __init__(self, V: fem.FunctionSpace, input_ufl_expression, coefficients, get_eval):
        """Inits `CustomFunction`"""
        
        super().__init__(V)
        self.local_dim = V.element.space_dimension
        self.local_shape = V._ufl_element.reference_value_shape()

        self.global_values = self.x.array.reshape((-1, self.local_dim)) #but local on a process

        self.input_ufl_expression = input_ufl_expression
        self.dummies = []
        self.coefficients = []

        for coeff in coefficients:
            self.add_coefficient(coeff)
        
        if V._ufl_element.family() == 'Quadrature':
            basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
            gauss_points, _ = basix.make_quadrature(basix_celltype, V._ufl_element.degree())
            self.input_expression = fem.Expression(self.input_ufl_expression, gauss_points)
        else:
            self.input_expression = fem.Expression(self.input_ufl_expression, V.element.interpolation_points)

        self.tabulated_input_expression = self.input_expression._ufcx_expression.tabulate_tensor_float64

        self.eval = get_eval(self)

    
    def set_values(self, cell, values):
        self.global_values[cell][:] = values

    def add_coefficient(self, coeff, coeff_name:typing.Optional[str] = None):
        if coeff not in self.coefficients and coeff not in self.dummies:
            name = 'default'
            if coeff_name is None:
                name = coeff.name
            else:
                name = coeff_name
            setattr(self, name, coeff)

            if isinstance(coeff, DummyFunction):
                self.dummies.append(getattr(self, name))
            elif isinstance(coeff, fem.Function):
                self.coefficients.append(getattr(self, name))

@numba.cfunc(c_signature, nopython=True)
def dummy_tabulated(b_, w_, c_, coords_, local_index, orientation):
    """Simulates a typical tabulated function"""
    pass

@numba.njit
def dummy_eval(values, coeffs_values, constants_values, coordinates, local_index, orientation):
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

def extract_data(form: ufl.form.Form) -> typing.Union[np.ndarray, list, list, np.ndarray]:
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
    coeffs_eval = []
    coeffs_input_expression_constants = []
    coeffs_values_global = []
    coeffs_dummy_values = []
    coeffs_coeff_values = []

    for i, coeff in enumerate(form.coefficients()):
        if isinstance(coeff, CustomFunction): 
            N_coeffs_values_local[i] = coeff.local_dim
            coeffs_eval.append(coeff.eval)
            for sub_coeff in coeff.coefficients:
                coeffs_coeff_values.append(sub_coeff.x.array.reshape((-1, sub_coeff.function_space.element.space_dimension)))
            
            for sub_coeff in coeff.dummies:
                coeffs_dummy_values.append(sub_coeff.value)

            coeffs_input_expression_constants.append(extract_constants(coeff.input_ufl_expression))
            coeffs_values_global.append(coeff.global_values)

    #Numba doesn't like empty lists, so we have to fill it with something 
    if len(coeffs_eval) == 0 :
        coeffs_eval.append(dummy_eval)

    if len(coeffs_input_expression_constants) == 0 :
        coeffs_input_expression_constants.append(np.array([-1], dtype=PETSc.ScalarType))

    if len(coeffs_values_global) == 0 :
        coeffs_values_global.append(np.array([[-1], [-1]], dtype=PETSc.ScalarType))
        
    if len(coeffs_dummy_values) == 0 :
        coeffs_dummy_values.append(np.array([[-1], [-1]], dtype=PETSc.ScalarType))
        
    if len(coeffs_coeff_values) == 0 :
        coeffs_coeff_values.append(np.array([[-1], [-1]], dtype=PETSc.ScalarType))
    

    constants_values = extract_constants(form)
            
    return N_coeffs_values_local, coeffs_values_global, coeffs_eval, coeffs_input_expression_constants, coeffs_dummy_values, coeffs_coeff_values, constants_values

@numba.njit(fastmath=True)
def assemble_ufc(A, b, u, geo_dofs, coords, dofmap, num_owned_cells, N_dofs_element,
                 N_coeffs_values_local_A, coeffs_values_global_A, coeffs_eval_A, coeffs_constants_A, coeffs_dummy_values_A, coeffs_coeff_values_A, constants_values_A, local_assembling_A, 
                 N_coeffs_values_local_b, coeffs_values_global_b, coeffs_eval_b, coeffs_constants_b, coeffs_dummy_values_b, coeffs_coeff_values_b, constants_values_b, local_assembling_b, 
                 kernel_A, kernel_b, 
                 g, scale, x0, 
                 mode=PETSc.InsertMode.ADD_VALUES):
    """Assembles the matrix A and the vector b using FFCx/UFC approach 

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

    # N_coeffs_b = N_coeffs_values_local_b.size
    # N_coeffs_A = N_coeffs_values_local_A.size

    # coeffs_A = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_A == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_A)), dtype=PETSc.ScalarType)
               
    # coeffs_b = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_b == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_b)), dtype=PETSc.ScalarType)

    b_local = np.zeros(N_dofs_element, dtype=PETSc.ScalarType)
    A_local = np.zeros((N_dofs_element, N_dofs_element), dtype=PETSc.ScalarType)

    for cell in range(num_owned_cells):
        pos = rows = cols = dofmap[cell]
        geometry[:] = coords[geo_dofs[cell], :]
        u_local = u[pos]
        x0_local = x0[pos]
        g_local = g[pos]

        b_local.fill(0.)
        A_local.fill(0.)
        # coeffs_A.fill(0.)

        coeffs_b = local_assembling_b(cell, coeffs_values_global_b, coeffs_coeff_values_b, coeffs_dummy_values_b, coeffs_eval_b, u_local, coeffs_constants_b, geometry, entity_local_index, perm)

        # sigma_local = coeffs_values_global_b[0][cell]
        # p_local = coeffs_coeff_values_b[0][cell]
        # dp_local = coeffs_coeff_values_b[1][cell]
        # sigma_old_local = coeffs_coeff_values_b[2][cell]

        # output_values = coeffs_eval_b[0](sigma_local, 
        #                                  sigma_old_local,
        #                                  p_local,
        #                                  dp_local,
        #                                  u_local, 
        #                                  coeffs_constants_b[0], 
        #                                  geometry, entity_local_index, perm)

        # coeffs_b = sigma_local

        # for i in range(len(coeffs_dummy_values_b)):
        #     coeffs_dummy_values_b[i][:] = output_values[i] #C_tang update

        coeffs_A = local_assembling_A(coeffs_dummy_values_b)

        # coeffs_A = coeffs_dummy_values_b[0]

        kernel_b(ffi.from_buffer(b_local), 
                 ffi.from_buffer(coeffs_b),
                 ffi.from_buffer(constants_values_b),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))
        
        kernel_A(ffi.from_buffer(A_local), 
                 ffi.from_buffer(coeffs_A),
                 ffi.from_buffer(constants_values_A),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        #Local apply lifting : b_local - scale * A_local_brut(g_local - x0_local)
        b_local -= scale * A_local @ ( g_local - x0_local )
        b[pos] += b_local

        MatSetValues_ctypes(A, N_dofs_element, rows.ctypes, N_dofs_element, cols.ctypes, A_local.ctypes, mode)

@numba.njit(fastmath=True)
def assemble_ufc_b(b, u, geo_dofs, coords, dofmap, num_owned_cells, N_dofs_element,
                 N_coeffs_values_local_b, coeffs_values_global_b, coeffs_eval_b, coeffs_constants_b, coeffs_dummy_values_b, coeffs_coeff_values_b, constants_values_b, local_assembling_b, 
                 kernel_A, kernel_b, 
                 g, scale, x0, 
                 mode=PETSc.InsertMode.ADD_VALUES):
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))

    # N_coeffs_b = N_coeffs_values_local_b.size
    # N_coeffs_A = N_coeffs_values_local_A.size

    # coeffs_A = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_A == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_A)), dtype=PETSc.ScalarType)
               
    # coeffs_b = np.zeros(1, dtype=PETSc.ScalarType) if N_coeffs_b == 0 else \
    #            np.zeros(int(np.sum(N_coeffs_values_local_b)), dtype=PETSc.ScalarType)

    b_local = np.zeros(N_dofs_element, dtype=PETSc.ScalarType)
    A_local = np.zeros((N_dofs_element, N_dofs_element), dtype=PETSc.ScalarType)

    for cell in range(num_owned_cells):
        pos = rows = cols = dofmap[cell]
        geometry[:] = coords[geo_dofs[cell], :]
        u_local = u[pos]
        x0_local = x0[pos]
        g_local = g[pos]

        b_local.fill(0.)
        A_local.fill(0.)
        # coeffs_A.fill(0.)

        coeffs_b = local_assembling_b(cell, coeffs_values_global_b, coeffs_coeff_values_b, coeffs_dummy_values_b, coeffs_eval_b, u_local, coeffs_constants_b, geometry, entity_local_index, perm)

        # sigma_local = coeffs_values_global_b[0][cell]
        # p_local = coeffs_coeff_values_b[0][cell]
        # dp_local = coeffs_coeff_values_b[1][cell]
        # sigma_old_local = coeffs_coeff_values_b[2][cell]

        # output_values = coeffs_eval_b[0](sigma_local, 
        #                                  sigma_old_local,
        #                                  p_local,
        #                                  dp_local,
        #                                  u_local, 
        #                                  coeffs_constants_b[0], 
        #                                  geometry, entity_local_index, perm)

        # coeffs_b = sigma_local

        # for i in range(len(coeffs_dummy_values_b)):
        #     coeffs_dummy_values_b[i][:] = output_values[i] #C_tang update

        coeffs_A = local_assembling_A(coeffs_dummy_values_b)

        # coeffs_A = coeffs_dummy_values_b[0]

        kernel_b(ffi.from_buffer(b_local), 
                 ffi.from_buffer(coeffs_b),
                 ffi.from_buffer(constants_values_b),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))
        
        kernel_A(ffi.from_buffer(A_local), 
                 ffi.from_buffer(coeffs_A),
                 ffi.from_buffer(constants_values_A),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        #Local apply lifting : b_local - scale * A_local_brut(g_local - x0_local)
        b_local -= scale * A_local @ ( g_local - x0_local )
        b[pos] += b_local

        MatSetValues_ctypes(A, N_dofs_element, rows.ctypes, N_dofs_element, cols.ctypes, A_local.ctypes, mode)


def get_topological_dofmap(V:fem.function.FunctionSpace):
    """Makes a topological dofmap for a vector function space
    Args:
        V: vector function space
    Returns:
        dofmap_topological: a numpy array of size (cells number, number of vector function space dofs)
    """
    N_dofs_element = V.element.space_dimension
    len_funtion_space = V.dofmap.index_map_bs
    dofmap = V.dofmap.list.array.reshape(-1, int(N_dofs_element/len_funtion_space))
    dofmap_topological = (len_funtion_space*np.repeat(dofmap, len_funtion_space).reshape(-1, len_funtion_space) + np.arange(len_funtion_space)).reshape(-1, N_dofs_element).astype(np.dtype(PETSc.IntType))

    # Another way to do it:
    # for i in np.arange(dofmap_topological.shape[0]):
    # for j in np.arange(3):
    #     dofmap_top2[i,j*2] = dofs_sub0[i][j] 
    #     dofmap_top2[i,j*2 + 1] = dofs_sub1[i][j]
    
    return dofmap_topological

class CustomSolver:

    def __init__(self, 
                dR: ufl.Form,
                R: ufl.Form,
                u: fem.Function,
                local_assembling_A,
                local_assembling_b,
                bcs: typing.List[fem.dirichletbc] = [],
):
        self.u = u
        self.bcs = bcs
        self.bcs_dofs = np.concatenate([bc.dof_indices()[0] for bc in bcs]) if len(bcs) != 0 else np.array([], dtype=PETSc.IntType)

        V = u.function_space
        domain = V.mesh

        self.dR = dR
        self.R = R
        
        self.b_form = fem.form(R)
        self.A_form = fem.form(dR)
        self.b = fem.petsc.create_vector(self.b_form)
        self.A = fem.petsc.create_matrix(self.A_form)

        self.local_assembling_A = local_assembling_A
        self.local_assembling_b = local_assembling_b

        self.g = fem.petsc.create_vector(self.b_form)
        with self.g.localForm() as g_local:
            g_local.set(0.0)
        
        self.x0 = fem.petsc.create_vector(self.b_form)
        with self.x0.localForm() as x0_local:
            x0_local.set(0.0)

        self.comm = domain.comm
        map_c = domain.topology.index_map(domain.topology.dim)
        self.num_owned_cells = map_c.size_local
        num_cells = self.num_owned_cells + map_c.num_ghosts
        self.N_dofs_element = V.element.space_dimension
        self.geo_dofs = domain.geometry.dofmap.array.reshape(num_cells, 3)
        self.coordinates = domain.geometry.x

        #This dofmap takes into account dofs of the vector field
        self.dofmap_topological = get_topological_dofmap(V)

        self.kernel_A = get_kernel(dR)
        self.kernel_b = get_kernel(R)

        self.data_extraction()

        self.solver = self.solver_setup()

    def data_extraction(self):
        self.N_coeffs_values_local_A, self.coeffs_values_global_A, self.coeffs_eval_A, self.coeffs_constants_A, self.coeffs_dummy_values_A, self.coeffs_coeff_values_A, self.constants_values_A = extract_data(self.dR)
        self.N_coeffs_values_local_b, self.coeffs_values_global_b, self.coeffs_eval_b, self.coeffs_constants_b, self.coeffs_dummy_values_b, self.coeffs_coeff_values_b, self.constants_values_b = extract_data(self.R)

    def solver_setup(self):
        solver = PETSc.KSP().create(self.comm)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(self.A)
        return solver

    def assemble(self, scale: float, x0: np.ndarray, b_additional: typing.Optional[PETSc.Vec] = None):
        self.A.zeroEntries()
        with self.b.localForm() as b_local:
            b_local.set(0.0)
                
        fem.set_bc(self.g, self.bcs)
        if x0 is not None:
            fem.set_bc(self.x0, self.bcs, x0=self.g.array + x0, scale=-1.0)

        assemble_ufc(
            self.A.handle, self.b.array, self.u.x.array, self.geo_dofs, self.coordinates, self.dofmap_topological, self.num_owned_cells, self.N_dofs_element,
            self.N_coeffs_values_local_A, self.coeffs_values_global_A, self.coeffs_eval_A, self.coeffs_constants_A, self.coeffs_dummy_values_A, self.coeffs_coeff_values_A, self.constants_values_A, self.local_assembling_A,
            self.N_coeffs_values_local_b, self.coeffs_values_global_b, self.coeffs_eval_b, self.coeffs_constants_b, self.coeffs_dummy_values_b, self.coeffs_coeff_values_b, self.constants_values_b, self.local_assembling_b,
            self.kernel_A, self.kernel_b,
            self.g.array, scale, self.x0.array
        )

        self.A.assemble()
        self.A.zeroRowsColumnsLocal(self.bcs_dofs, 1.)

        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs, x0=x0, scale=scale)

        if b_additional is not None:
            self.b.axpy(1, b_additional)
        fem.set_bc(self.b, self.bcs, x0=x0, scale=scale)

    def solve(self, 
              du: fem.function.Function, 
              scale: float = 1.0,
              x0: np.ndarray = None,
              b_additional: typing.Optional[PETSc.Vec] = None
):
        self.assemble(scale, x0, b_additional)
        self.solver.setOperators(self.A)
        self.solver.solve(self.b, du.vector)
