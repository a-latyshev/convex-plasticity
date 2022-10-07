import cffi
import numba
import numpy as np

import ufl
from dolfinx.jit import ffcx_jit
from dolfinx import fem, mesh, la

import ctypes
import ctypes.util
import petsc4py.lib
from mpi4py import MPI
from petsc4py import get_config as PETSc_get_config
from petsc4py import PETSc

from typing import Optional, List, Dict, Union, Callable
import basix
import os 
import logging

import sys
sys.path.append('../optimisation/')
import plasticity_framework as pf
import convex_return_mapping as crm
import utility_functions as uf

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
            
            The 3rd argument (`constants_local`) takes an array of constants values being used in the form
            
            The 4th argument (`geometry`) takes a 3*3 array of cell nodes coordinates.
            
            The 5th argument (`entity_local_index`) takes an array of entity_local_index.
            
            The 6th argument (`permutation`) is an elements permutation option.

    Note: The form can have several integrals, but here we suppose that there is an only one.
    """
    domain = form.ufl_domain().ufl_cargo()
    ufcx_form, _, _ = ffcx_jit(domain.comm, form, form_compiler_params={"scalar_type": "double"})
    kernel = ufcx_form.integrals(0)[0].tabulate_tensor_float64
    return kernel

def get_dummy_x(V: fem.FunctionSpace) -> la.vector:
    """Creates a vector with the size of two elements mesh.

    It creates the same functional space as the given one, but based on another mesh, which is presented here in the form of a triangulated unit square with two elements. It returns the vector defined in the new functional space. This approach allows to allocate less memory for `fem.Function` objects, for example.

    Args:
        V: A functional space.
    Returns:
        A `VectorMetaClass` object defined on a square mesh with two elements.
    """
    dummy_domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1) #TODO: COMM_WORLD 
    dummy_V = fem.FunctionSpace(dummy_domain, V._ufl_element)
    return la.vector(dummy_V.dofmap.index_map, dummy_V.dofmap.index_map_bs)

class DummyFunction(fem.Function): #or ConstantFunction or BasicFunction or LazyFunction or Sparse function ?
    """Expands the class `fem.Function` to allocate its values only on one element.

    Every `fem.Function` object stores its values globally, but we would like to avoid such a waste of memory updating the function value during the assembling procedure. We need an entity, which would contain only local values on an element. Then we update its values on every step of the custom assembling procedure. A behavior of the such entity is similar to the one of `fem.Constant`, but it is necessary to possess differents values of the entity on every finite element node. We find the `fem.Function` constructor argument `x` very useful here. We can set the `fem.Function` global vector to a vector of another `fem.Function` object defined on a different mesh, which can have less elements (2, for instance).
    
    Attribues: 
        values: A vector containing local values on only finite element.
        shape: A shape of the tensor of the function mathematical representation.  
    """
    def __init__(self, V: fem.FunctionSpace, name: Optional[str] = None):
        """Inits DummyFunction class."""
        super().__init__(V=V, x=get_dummy_x(V), name=name)
        self.value = self.x.array.reshape((2, -1))[0]
        self.shape = V._ufl_element.reference_value_shape()
    
    def fill(self, value: np.ndarray):
        """Sets the function values
        Args:
            value: A numpy array containing values to be set 
        """
        self.value[:] = value

class CustomFunction(fem.Function): #TODO: CustomExpression
    """Expands the class `fem.Function` and associates an appropriate mathematical expression.

    On a current dolfinx version we use `fem.Function` variable for `g` function defined as follows:  
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
        """Inits `CustomFunction`."""
        
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
        """Sets values on an element #`cell`.
        
        Args:
            cell:
                a cell index 
            values:
                local values to be set
        """

        self.global_values[cell][:] = values

    def add_coefficient(self, coeff: Union[DummyFunction, fem.Function], coeff_name:Optional[str] = None):
        """Adds and sorts coefficient of a `CustomFunction`.
            
        It sends the coefficient to `dummies` or `coefficients` according to its type. It creates an attribute of `CustomFunction` under a name of the `coeff` variable or string `coeff_name`.
         
        Args:
            coeff: 
                a coefficient of the CustomFunction
            coeff_name:
                an optional name for a CustomFunction attribute to get access to the coefficient 
        """

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

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

@numba.cfunc(c_signature, nopython=True)
def dummy_tabulated(b_, w_, c_, coords_, local_index, orientation):
    """Simulates a typical tabulated function with appropriate signature."""
    pass

@numba.njit
def dummy_eval(values, coeffs_values, constants_values, coordinates, local_index, orientation):
    """Does nothing. To be used for empty lists, which are sent to a numba function as its argument."""
    pass

def extract_constants(ufl_expression) -> np.ndarray:
    """Extracts and puts together all values of all `fem.function.Constant` presenting in `ufl_expression`.

    Returns:
        constants_values: a numpy flatten array with values of all constants. Values are sorted in accordance with the order of constants in `ufl_expression`. 
    """
    constants = ufl.algorithms.analysis.extract_constants(ufl_expression)
    constants_values = np.concatenate([const.value.flatten() for const in constants]) if len(constants) !=0 else np.zeros(0, dtype=PETSc.ScalarType)
    return constants_values

def extract_data(form: ufl.form.Form) -> Union[np.ndarray, list, list, np.ndarray]:
    """Extracts coefficients and constants of a given form and puts their values all together. 

    Args:
        form: linear or bilinear form
    
    Returns: an union of:
        coeffs_global_values:
            a 2d numpy array containing global values of all coefficients of the form either `CustomFunction` or `fem.Function`
    
        coeffs_eval_list: 
            a list with all `eval` CustomFunctions methods in the form
    
        coeffs_constants_values:
            a flatten numpy array containing all values of constants taking a part in CustomFunction coefficients definition in the form
    
        coeffs_dummies_values:
            a 2d numpy array containing all values of DummyFunction coefficients of CustomFunction coefficients of the form
    
        coeffs_subcoeffs_values: 
            a 2d numpy array containing all values of fem.Function coefficients of CustomFunction coefficients of the form
    
        constants_values:
            a flatten numpy array containing all values of constants of the form

    Note:
        It is assumed that all form coefficients are `CustomFunction`, which have their own constants `fem.function.Constant`.
        and in the same time its haven't their own `CustomFunction`.

    """
    coeffs_eval_list = []
    coeffs_constants_values = []
    coeffs_global_values = []
    coeffs_dummies_values = []
    coeffs_subcoeffs_values = []

    EMPTY_ELEMENT_BELIKE = np.array([[-1], [-1]], dtype=PETSc.ScalarType)

    for i, coeff in enumerate(form.coefficients()):
        if isinstance(coeff, CustomFunction): 
            coeffs_eval_list.append(coeff.eval)

            for sub_coeff in coeff.coefficients:
                coeffs_subcoeffs_values.append(sub_coeff.x.array.reshape((-1, sub_coeff.function_space.element.space_dimension)))
            
            for sub_coeff in coeff.dummies:
                coeffs_dummies_values.append(sub_coeff.value)

            coeffs_constants_values.append(extract_constants(coeff.input_ufl_expression))
            coeffs_global_values.append(coeff.global_values)

    #Numba doesn't like empty lists, so we have to fill it with something 
    if len(coeffs_eval_list) == 0 :
        coeffs_eval_list.append(dummy_eval)

    if len(coeffs_constants_values) == 0 :
        coeffs_constants_values.append(np.array([-1], dtype=PETSc.ScalarType))

    if len(coeffs_global_values) == 0 :
        coeffs_global_values.append(EMPTY_ELEMENT_BELIKE)
        
    if len(coeffs_dummies_values) == 0 :
        coeffs_dummies_values.append(EMPTY_ELEMENT_BELIKE)

    if len(coeffs_subcoeffs_values) == 0 :
        coeffs_subcoeffs_values.append(EMPTY_ELEMENT_BELIKE)

    constants_values = extract_constants(form)
            
    return coeffs_global_values, coeffs_eval_list, coeffs_constants_values, coeffs_dummies_values, coeffs_subcoeffs_values, constants_values

@numba.njit(fastmath=True)
def assemble_ufc(A, b, u, geo_dofs, coords, dofmap, num_owned_cells, N_dofs_element,
                 coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, constants_values_A, local_assembling_A, 
                 coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b, constants_values_b, local_assembling_b, 
                 kernel_A, kernel_b, 
                 g, scale, x0, 
                 mode=PETSc.InsertMode.ADD_VALUES):
    """Assembles the matrix A and the vector b using FFCx/UFC approach 
    
    It applies lifting locally to take into account inhomogeneous Dirichlet boundary conditions.
    As it is a numba-function only numpy arrays, c-functions and other trivial objects and methods are allowed.

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
     
        N_dofs_element:
            a number of dofs in a solution function space 
       
        coeffs_global_values_:
            a 2d numpy array containing global values of all coefficients of an appropriate form (A or b, bilinear or linear) either `CustomFunction` or `fem.Function`
    
        coeffs_eval_list_: 
            a list with all `eval` CustomFunctions methods in an appropriate form (A or b, bilinear or linear)
    
        coeffs_constants_values_:
            a flatten numpy array containing all values of constants taking a part in CustomFunction coefficients definition in an appropriate form (A or b, bilinear or linear)
    
        coeffs_dummies_values_:
            a 2d numpy array containing all values of DummyFunction coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        coeffs_subcoeffs_values_: 
            a 2d numpy array containing all values of fem.Function coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        constants_values_:
            a flatten numpy array containing all values of constants of an appropriate form (A or b, bilinear or linear)
    
        local_assembling_:
            a numba function making some particular calculations inside the assembling loop for an appropriate form (A or b, bilinear or linear)
    
        kernel_A, kernel_b:
            c-function representation of the bilinear and linear forms respectively 
    
        g, scale, x0: applying lifting variables
            b - scale * A * (g - x0)
            g is a vector equal to the inhomogeneous Dirichlet BC at dofs with this BC, and zero elsewhere
    
        mode:
            a mode of matrix assembling      
    """
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))

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

        coeffs_b = local_assembling_b(cell, geometry, entity_local_index, perm, u_local,
                coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b)

        coeffs_A = local_assembling_A(cell, geometry, entity_local_index, perm, u_local,
                coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b)

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
                   coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, constants_values_A, 
                   coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b, constants_values_b, local_assembling_b, 
                   kernel_b):
    """Assembles the vector b using FFCx/UFC approach 

    As it is a numba-function only numpy arrays, c-functions and other trivial objects and methods are allowed.

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
     
        N_dofs_element:
            a number of dofs in a solution function space 
       
        coeffs_global_values_:
            a 2d numpy array containing global values of all coefficients of an appropriate form (A or b, bilinear or linear) either `CustomFunction` or `fem.Function`
    
        coeffs_eval_list_: 
            a list with all `eval` CustomFunctions methods in an appropriate form (A or b, bilinear or linear)
    
        coeffs_constants_values_:
            a flatten numpy array containing all values of constants taking a part in CustomFunction coefficients definition in an appropriate form (A or b, bilinear or linear)
    
        coeffs_dummies_values_:
            a 2d numpy array containing all values of DummyFunction coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        coeffs_subcoeffs_values_: 
            a 2d numpy array containing all values of fem.Function coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        constants_values_:
            a flatten numpy array containing all values of constants of an appropriate form (A or b, bilinear or linear)
    
        local_assembling_b:
            a numba function making some particular calculations inside the assembling loop for a linear form 
    
        kernel_b:
            a c-function representation of the linear form   
    """
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))

    b_local = np.zeros(N_dofs_element, dtype=PETSc.ScalarType)

    for cell in range(num_owned_cells):
        pos = dofmap[cell]
        geometry[:] = coords[geo_dofs[cell], :]
        u_local = u[pos]

        b_local.fill(0.)

        coeffs_b = local_assembling_b(cell, geometry, entity_local_index, perm, u_local,
                coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b)
       
        kernel_b(ffi.from_buffer(b_local), 
                 ffi.from_buffer(coeffs_b),
                 ffi.from_buffer(constants_values_b),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))
        
        b[pos] += b_local

@numba.njit(fastmath=True)
def assemble_ufc_A(A, u, geo_dofs, coords, dofmap, num_owned_cells, N_dofs_element,
                   coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, constants_values_A, local_assembling_A,
                   coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b, constants_values_b, local_assembling_b, 
                   kernel_A, 
                   mode=PETSc.InsertMode.ADD_VALUES):
    """Assembles the matrix A using FFCx/UFC approach
    
    As it is a numba-function only numpy arrays, c-functions and other trivial objects and methods are allowed.

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
     
        N_dofs_element:
            a number of dofs in a solution function space 
       
        coeffs_global_values_:
            a 2d numpy array containing global values of all coefficients of an appropriate form (A or b, bilinear or linear) either `CustomFunction` or `fem.Function`
    
        coeffs_eval_list_: 
            a list with all `eval` CustomFunctions methods in an appropriate form (A or b, bilinear or linear)
    
        coeffs_constants_values_:
            a flatten numpy array containing all values of constants taking a part in CustomFunction coefficients definition in an appropriate form (A or b, bilinear or linear)
    
        coeffs_dummies_values_:
            a 2d numpy array containing all values of DummyFunction coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        coeffs_subcoeffs_values_: 
            a 2d numpy array containing all values of fem.Function coefficients of CustomFunction coefficients of an appropriate form (A or b, bilinear or linear)
    
        constants_values_:
            a flatten numpy array containing all values of constants of an appropriate form (A or b, bilinear or linear)
    
        local_assembling_:
            a numba function making some particular calculations inside the assembling loop for an appropriate form (A or b, bilinear or linear)
    
        kernel_A, kernel_b:
            c-function representations of the bilinear and linear forms respectively 
    
        mode:
            a mode of matrix assembling      
    """
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))

    A_local = np.zeros((N_dofs_element, N_dofs_element), dtype=PETSc.ScalarType)

    for cell in range(num_owned_cells):
        pos = rows = cols = dofmap[cell]
        geometry[:] = coords[geo_dofs[cell], :]
        u_local = u[pos]

        A_local.fill(0.)
        
        coeffs_b = local_assembling_b(cell, geometry, entity_local_index, perm, u_local,
                coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b)

        coeffs_A = local_assembling_A(cell, geometry, entity_local_index, perm, u_local,
                coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b)
        
        kernel_A(ffi.from_buffer(A_local), 
                 ffi.from_buffer(coeffs_A),
                 ffi.from_buffer(constants_values_A),
                 ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index), ffi.from_buffer(perm))

        MatSetValues_ctypes(A, N_dofs_element, rows.ctypes, N_dofs_element, cols.ctypes, A_local.ctypes, mode)

# @numba.njit(fastmath=True)
# def apply_lifting(A, b, u, geo_dofs, coords, dofmap, num_owned_cells, N_dofs_element,
    #              N_coeffs_values_local_A, coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, |

def get_topological_dofmap(V: fem.function.FunctionSpace):
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

class CustomProblem(pf.LinearProblem):
    """Class for solving a variational problem via custom assembling approach.

    Attributes:
        u: A vector containing global values of a solution
        bcs: A list of Dirichlet boundary conditions 
        bcs_dofs: A flatten numpy array containing Dirichlet boundary conditions indexes
        dR: A ufl-expression of a bilinear form
        R: A ufl-expression of a linear form
        A_form: A fem form of a bilinear form
        b_form: A fem form of a linear form
        b: A vector containing global rhs vector of the linear system
        A: A vector containing global matrix of the linear system 
        local_assembling_A: A numba function making some particular calculations inside the assembling loop for bilinear form coefficients
        local_assembling_b: a numba function making some particular calculations inside the assembling loop for linear form coefficients
        g: A vector equal to the inhomogeneous Dirichlet BC at dofs with this BC, and zero elsewhere
        x0:
        comm:
        num_owned_cells: 
        geo_dofs: 
        coordinates: 
        dofmap_topological:
        kernel_A, kernel_b:
        ....

        solver:
    """

    def __init__(
        self, 
        dR: ufl.Form,
        R: ufl.Form,
        u: fem.Function,
        local_assembling_A: numba.core.registry.CPUDispatcher,
        local_assembling_b: numba.core.registry.CPUDispatcher,
        bcs: List[fem.dirichletbc] = [],
        res_Neumann: Optional[ufl.form.Form] = None, 
        Nitermax: int = 200, 
        tol: float = 1e-8,
        # inside_Newton: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Inits CustomProblem."""
        super().__init__(dR, R, u, bcs)

        V = u.function_space
        domain = V.mesh

        self.bcs_dofs = np.concatenate([bc.dof_indices()[0] for bc in bcs]) if len(bcs) != 0 else np.array([], dtype=PETSc.IntType)

        self.local_assembling_A = local_assembling_A
        self.local_assembling_b = local_assembling_b

        self.g = fem.petsc.create_vector(self.b_form)
        with self.g.localForm() as g_local:
            g_local.set(0.0)
        
        self.x0 = fem.petsc.create_vector(self.b_form)
        with self.x0.localForm() as x0_local:
            x0_local.set(0.0)

        map_c = domain.topology.index_map(domain.topology.dim)
        self.num_owned_cells = map_c.size_local
        num_cells = self.num_owned_cells + map_c.num_ghosts
        self.N_dofs_element = V.element.space_dimension
        self.geo_dofs = domain.geometry.dofmap.array.reshape(num_cells, 3)
        self.coordinates = domain.geometry.x

        #This dofmap takes into account dofs of the vector field
        #TODO: We would like to avoid the allocation of that map
        self.dofmap_topological = get_topological_dofmap(V)

        self.kernel_A = get_kernel(self.dR)
        self.kernel_b = get_kernel(self.R)

        self.data_extraction()

        self.solver = self.solver_setup()

        if res_Neumann is not None:
            self.form_res_Neumann = fem.form(res_Neumann)
            self.b_Neumann = fem.petsc.create_vector(self.form_res_Neumann)
        else:
            self.form_res_Neumann = None
            self.b_Neumann = None

        self.Nitermax = Nitermax
        self.tol = tol
        self.du = fem.Function(self.u.function_space)

        # if inside_Newton is not None:
        #     self.inside_Newton = inside_Newton
        # else:
        #     def dummy_func():
        #         pass
        #     self.inside_Newton = dummy_func
        if logger is not None:
            self.logger = logger 
        else:
            self.logger = logging.getLogger('custom_solver')

        self.assemble_matrix()

    def data_extraction(self):
        """Extracts coefficients and constants values and a list of `eval` CustomFunction functions of bilinear and linear forms."""

        self.coeffs_global_values_A, self.coeffs_eval_list_A, self.coeffs_constants_values_A, self.coeffs_dummies_values_A, self.coeffs_subcoeffs_values_A, self.constants_values_A = extract_data(self.dR)

        # self.coeffs_global_values_A, self.coeffs_eval_list_A, self.coeffs_constants_values_A, self.coeffs_dummies_values_A, self.coeffs_subcoeffs_values_A, self.constants_values_A = extract_data(self.dR)

        self.coeffs_global_values_b, self.coeffs_eval_list_b, self.coeffs_constants_values_b, self.coeffs_dummies_values_b, self.coeffs_subcoeffs_values_b, self.constants_values_b = extract_data(self.R)
        # self.coeffs_global_values_b, self.coeffs_eval_list_b, self.coeffs_constants_values_b, self.coeffs_dummies_values_b, self.coeffs_subcoeffs_values_b, self.constants_values_b = extract_data(self.R)

    def assemble_Neumann(self) -> None:
        with self.b_Neumann.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b_Neumann, self.form_res_Neumann)

    def assemble_matrix(self) -> None:
        """Assembles the matrix A and applies Dirichlet boundary conditions."""
        self.A.zeroEntries()
        assemble_ufc_A(
            self.A.handle, self.u.x.array, self.geo_dofs, self.coordinates, self.dofmap_topological, self.num_owned_cells, self.N_dofs_element,
            self.coeffs_global_values_A, self.coeffs_eval_list_A, self.coeffs_constants_values_A, self.coeffs_dummies_values_A, self.coeffs_subcoeffs_values_A, self.constants_values_A, self.local_assembling_A,
            self.coeffs_global_values_b, self.coeffs_eval_list_b, self.coeffs_constants_values_b, self.coeffs_dummies_values_b, self.coeffs_subcoeffs_values_b, self.constants_values_b, self.local_assembling_b,
            self.kernel_A
        )
        self.A.assemble()
        self.A.zeroRowsColumnsLocal(self.bcs_dofs, 1.)

    def assemble_vector(self, 
        scale: float = 1.0, 
        x0: Optional[np.ndarray] = None, 
        # b_additional: Optional[PETSc.Vec] = None
    ) -> None:
        """Assembles the vector b and adds optionally a global vector `b_additional` to it.
        
        Args:
            b_additional: An optional global vector to be added to the vector b. It can be used for applying of Neuman boundary conditions.
        """
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        
        assemble_ufc_b(
            self.b.array, self.u.x.array, self.geo_dofs, self.coordinates, self.dofmap_topological, self.num_owned_cells, self.N_dofs_element,
            self.coeffs_global_values_A, self.coeffs_eval_list_A, self.coeffs_constants_values_A, self.coeffs_dummies_values_A, self.coeffs_subcoeffs_values_A, self.constants_values_A,
            self.coeffs_global_values_b, self.coeffs_eval_list_b, self.coeffs_constants_values_b, self.coeffs_dummies_values_b, self.coeffs_subcoeffs_values_b, self.constants_values_b, self.local_assembling_b,
            self.kernel_b
        )

        #TODO: How to get better?
        if self.b_Neumann is not None:
            self.b.axpy(1, self.b_Neumann)

        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs, x0=x0, scale=scale)

    def assemble(
        self, 
        scale: float = 1.0, 
        x0: Optional[np.ndarray] = None, 
        # b_additional: Optional[PETSc.Vec] = None
    ) -> None:
        """Assembles the hole system and sets up all boundary conditions.
        
        It applies the lifting locally to take into account inhomogeneous Dirichlet boundary conditions as follows:
        
        .. math:: b - scale  A  (g - x0),

        where g is a vector equal to the inhomogeneous Dirichlet BC at dofs with this BC, and zero elsewhere.

        Args:
            scale: A float scale factor for lifting Dirichlet BC.
            x0: A global vector for lifting Dirichlet BC.
            b_additional: A global vector to be summed with the vector `b`. Shell be used to apply Neuman BC.
        """
        self.A.zeroEntries()
        with self.b.localForm() as b_local:
            b_local.set(0.0)
                
        fem.set_bc(self.g, self.bcs)
        # self.x0 != x0, self.x0 is for applying lifting
        if x0 is not None:
            fem.set_bc(self.x0, self.bcs, x0=self.g.array + x0, scale=-1.0)

        assemble_ufc(
            self.A.handle, self.b.array, self.u.x.array, self.geo_dofs, self.coordinates, self.dofmap_topological, self.num_owned_cells, self.N_dofs_element,
            self.coeffs_global_values_A, self.coeffs_eval_list_A, self.coeffs_constants_values_A, self.coeffs_dummies_values_A, self.coeffs_subcoeffs_values_A, self.constants_values_A, self.local_assembling_A,
            self.coeffs_global_values_b, self.coeffs_eval_list_b, self.coeffs_constants_values_b, self.coeffs_dummies_values_b, self.coeffs_subcoeffs_values_b, self.constants_values_b, self.local_assembling_b,
            self.kernel_A, self.kernel_b,
            self.g.array, scale, self.x0.array
        )

        self.A.assemble()
        self.A.zeroRowsColumnsLocal(self.bcs_dofs, 1.)

        if self.b_Neumann is not None:
            # #TODO: It will be called on every
            # self.assemble_Neumann()
            self.b.axpy(1, self.b_Neumann)

        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        fem.set_bc(self.b, self.bcs, x0=x0, scale=scale)

    def solve(self) -> int:
        """Solves the linear system and saves the solution into the vector `du`
        
        Args:
            du: A global vector to be used as a container for the solution of the linear system
        """
        with self.g.localForm() as g_local:
            g_local.set(0.0)
        with self.x0.localForm() as x0_local:
            x0_local.set(0.0)

        self.assemble_Neumann()
        self.assemble_vector()
    
        nRes0 = self.b.norm()
        nRes = nRes0
        niter = 0

        while nRes/nRes0 > self.tol and niter < self.Nitermax:
            self.solver.solve(self.b, self.du.vector)

            self.u.vector.axpy(1, self.du.vector) # Du = Du + 1*du
            self.u.x.scatter_forward() 

            self.assemble()
            
            nRes = self.b.norm() 
            niter += 1
            self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Increment: {niter}, norm(Res/Res0) = {nRes/nRes0:.1e}.')

        
        return niter

class SNESCustomProblem(CustomProblem):
    """
    Problem class compatible with PETSC.SNES solvers.
    """

    def __init__(
        self,
        F_form: ufl.Form,
        u: fem.Function,
        local_assembling_A: numba.core.registry.CPUDispatcher,
        local_assembling_b: numba.core.registry.CPUDispatcher,
        J_form: Optional[ufl.Form] = None,
        bcs: List[fem.dirichletbc] = [],
        petsc_options: Dict[str, Union[str, int, float]] = {},
        prefix: Optional[str] = None,
        # inside_Newton: Optional[Callable] = None,
        # logger: Optional[logging.Logger] = None,
        b_additional: Optional[PETSc.Vec] = None,
    ):
        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix
        self.petsc_options = petsc_options

        if J_form is None:
            V = u.function_space
            J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))
        
        super().__init__(J_form, F_form, u, local_assembling_A, local_assembling_b, bcs)

        # if inside_Newton is not None:
        #     self.inside_Newton = inside_Newton
        # else:
        #     def dummy_func():
        #         pass
        #     self.inside_Newton = dummy_func
        
        # if logger is not None:
        #     self.logger = logger 
        # else:
        #     self.logger = logging.getLogger('SNES_solver')

        self.b_additional = b_additional

                
    def set_petsc_options(self) -> None:
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        
        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solver_setup(self) -> PETSc.SNES:
        # Create nonlinear solver
        snes = PETSc.SNES().create(self.comm)

        # Set options
        snes.setOptionsPrefix(self.prefix)
        self.set_petsc_options()        

        snes.setFunction(self.assemble_vector, self.b)
        snes.setJacobian(self.assemble_matrix, self.A)

        snes.setFromOptions()

        return snes

    def assemble_vector(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # x_{k+1} = x_k + dx_k, where dx_k = x ?
        x.copy(self.u.vector) 
        self.u.x.scatter_forward()

        #TODO: SNES makes the iteration #0, where he calculates the b norm. `inside_Newton()` can be omitted in that case
        # self.inside_Newton()

        super().assemble_vector(self.b_additional)
        #WARNING: This solver doesn't support the lifting technique!
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)
        

    def assemble_matrix(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        super().assemble_matrix()

    def solve(self) -> int:
    
        # start = time.time()

        self.solver.solve(None, self.u.vector)
    
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  {self.prefix} SNES solver converged in {self.solver.getIterationNumber()} iterations with converged reason {self.solver.getConvergedReason()})')
        # self.logger.debug(f'rank#{MPI.COMM_WORLD.rank}  Time (Step) = {time.time() - start:.2f} (s)\n')

        self.u.x.scatter_forward()
        return self.solver.getIterationNumber() 

class CustomPlasticity(pf.AbstractPlasticity):
    def __init__(
        self, 
        material: crm.Material, 
        mesh_name: str = "thick_cylinder.msh", 
        logger: Optional[logging.Logger] = None,
        solver: str = "nonlinear",
    ):
        if not isinstance(material.yield_criterion, crm.vonMises):
            raise RuntimeError(f"Custom plasticity supports only Drucker-Prager material.")
        if solver != 'nonlinear':
            raise RuntimeError(f"Custom plasticity supports only nonlinear solver.")

        sig0 = material.yield_criterion.sig0
        mu_ = material.constitutive_law.mu_
        lambda_ = material.constitutive_law.lambda_
        H = material.yield_criterion.H
        
        super().__init__(sig0, mesh_name, logger)

        TPV = np.finfo(PETSc.ScalarType).eps # très petite value 
        I = np.eye(3)
        J4 = 1./3. * np.tensordot(I, I, axes=0)
        I4 = np.einsum('ij,kl->ikjl', I, I)
        K4 = DEV = I4 - J4
        C_elas = (3*lambda_ + 2*mu_)*J4 + 2*mu_*K4
        # C_elas_const = fem.Constant(mesh, C_elas.astype(np.dtype(PETSc.ScalarType)))

        QTe = ufl.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.deg_stress, shape=C_elas.shape, quad_scheme='default') #, symmetry=True?
        QT = fem.FunctionSpace(self.mesh, QTe)

        self.C_tang = DummyFunction(QT, name='tangent') # 2 * n_gauss_points * 3 * 3 * 3 * 3

        @numba.njit(fastmath=True)
        def as_3D_array(X):
            return np.asarray([[X[0], X[3], 0],
                            [X[3], X[1], 0],
                            [0, 0, X[2]]])
                            
        @numba.njit(fastmath=True)
        def ppos(x):
            return (x + np.sqrt(x**2))/2.

        @numba.njit(fastmath=True)
        def sigma(eps_el):
            return lambda_*np.trace(eps_el)*I + 2*mu_*eps_el

        @numba.njit(fastmath=True)
        def tensor_product(A, B):
            # n_i, n_j = A.shape
            # n_k, n_l = B.shape
            C = np.zeros((*A.shape, *B.shape), dtype=PETSc.ScalarType)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            C[i,j,k,l] = A[i,j] * B[k,l]
            return C
                            
        @numba.njit(fastmath=True)
        def get_C_tang(beta, n_elas):
            return C_elas - 3*mu_*(3*mu_/(3*mu_+H)-beta)*tensor_product(n_elas, n_elas) - 2*mu_*beta*DEV 

        @numba.njit(fastmath=True)
        def inner(A, B):
            return np.sum(A * B)

        def inner_product(C, epsilon):
            i, j, k, l = ufl.indices(4)
            return ufl.as_tensor( (C[i,j,k,l] * epsilon[k,l]), (i,j) )
        
        def get_eval(self: CustomFunction):
            tabulated_eps = self.tabulated_input_expression
            n_gauss_points = len(self.input_expression.X)
            local_shape = self.local_shape
            C_tang_shape = self.tangent.shape

            @numba.njit(fastmath=True)
            def eval(sigma_current_local, sigma_old_local, p_old_local, dp_local, coeffs_values, constants_values, coordinates, local_index, orientation):
                deps_local = np.zeros(n_gauss_points*3*3, dtype=PETSc.ScalarType)
                
                C_tang_local = np.zeros((n_gauss_points, *C_tang_shape), dtype=PETSc.ScalarType)
                
                sigma_old = sigma_old_local.reshape((n_gauss_points, *local_shape))
                sigma_new = sigma_current_local.reshape((n_gauss_points, *local_shape))

                tabulated_eps(ffi.from_buffer(deps_local), 
                            ffi.from_buffer(coeffs_values), 
                            ffi.from_buffer(constants_values), 
                            ffi.from_buffer(coordinates), ffi.from_buffer(local_index), ffi.from_buffer(orientation))
                
                deps_local = deps_local.reshape((n_gauss_points, 3, 3))

                n_elas = np.zeros((3, 3), dtype=PETSc.ScalarType) 
                beta = np.zeros(1, dtype=PETSc.ScalarType) 
                dp = np.zeros(1, dtype=PETSc.ScalarType) 

                for q in range(n_gauss_points):
                    sig_n = as_3D_array(sigma_old[q])
                    sig_elas = sig_n + sigma(deps_local[q])
                    s = sig_elas - np.trace(sig_elas)*I/3.
                    sig_eq = np.sqrt(3./2. * inner(s, s))
                    f_elas = sig_eq - sig0 - H*p_old_local[q]
                    f_elas_plus = ppos(f_elas)
                    dp[:] = f_elas_plus/(3*mu_+H)
                    
                    sig_eq += TPV # for the case when sig_eq is equal to 0.0
                    n_elas[:,:] = s/sig_eq*f_elas_plus/f_elas
                    beta[:] = 3*mu_*dp/sig_eq
            
                    new_sig = sig_elas - beta*s
                    sigma_new[q][:] = np.asarray([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]])
                    dp_local[q] = dp[0]
                    
                    C_tang_local[q][:] = get_C_tang(beta, n_elas)
                
                return [C_tang_local.flatten()] 
            return eval

        @numba.njit(fastmath=True)
        def local_assembling_b(cell, geometry, entity_local_index, perm, u_local,
                        coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                        coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b):
            sigma_local = coeffs_global_values_b[0][cell]
            p_local = coeffs_subcoeffs_values_b[0][cell]
            dp_local = coeffs_subcoeffs_values_b[1][cell]
            sigma_old_local = coeffs_subcoeffs_values_b[2][cell]

            output_values = coeffs_eval_list_b[0](sigma_local, 
                                            sigma_old_local,
                                            p_local,
                                            dp_local,
                                            u_local, 
                                            coeffs_constants_values_b[0], 
                                            geometry, entity_local_index, perm)

            coeffs_b = sigma_local

            for i in range(len(coeffs_dummies_values_b)):
                coeffs_dummies_values_b[i][:] = output_values[i] #C_tang update

            return coeffs_b

        @numba.njit(fastmath=True)
        def local_assembling_A(cell, geometry, entity_local_index, perm, u_local,
                        coeffs_global_values_A, coeffs_eval_list_A, coeffs_constants_values_A, coeffs_dummies_values_A, coeffs_subcoeffs_values_A, 
                        coeffs_global_values_b, coeffs_eval_list_b, coeffs_constants_values_b, coeffs_dummies_values_b, coeffs_subcoeffs_values_b):
            coeffs_A = coeffs_dummies_values_b[0]
            return coeffs_A
        
        self.dp = fem.Function(self.W0)  
        self.sig_old = fem.Function(self.W)
        
        self.sig = CustomFunction(self.W, uf.eps(self.Du), [self.C_tang, self.p, self.dp, self.sig_old], get_eval)

        a_Newton = ufl.inner(uf.eps(self.v_), inner_product(self.sig.tangent, uf.eps(self.u_)))*self.dx
        res = -ufl.inner(uf.eps(self.u_), uf.as_3D_tensor(self.sig))*self.dx 
        res_Neumann = self.F_ext(self.u_)

        self.problem = CustomProblem(a_Newton, res, self.Du, local_assembling_A, local_assembling_b, self.bcs, res_Neumann=res_Neumann, Nitermax=200, tol=1e-8, logger=self.logger)

        def after_Newton():
            self.sig_old.x.array[:] = self.sig.x.array[:]
            self.p.vector.axpy(1, self.dp.vector)
            self.p.x.scatter_forward()
        self.after_Newton = after_Newton

        def initialize_variables():
            print('Hello')
            self.sig.vector.set(0.0)
            self.sig_old.vector.set(0.0)
            self.p.vector.set(0.0)
            self.dp.vector.set(0.0)
            self.u.vector.set(0.0)
            self.C_tang.vector.set(0.0)
        
            # with self.g.localForm() as g_local:
            #     g_local.set(0.0)
            # with self.x0.localForm() as x0_local:
            #     x0_local.set(0.0)

        self.initialize_variables = initialize_variables 
