from dolfinx import fem
import ufl
from petsc4py import PETSc
import basix
import numpy as np
import meshio 

# The function performs a manual projection of an original_field function onto a target_field space 
def project(original_field, target_field:fem.Function, dx:ufl.Measure=ufl.dx, bcs=[]):
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

# Interpolation of a function defined over quadrature elements
def interpolate_quadrature(ufl_expr, fem_func:fem.Function):
    q_dim = fem_func.function_space._ufl_element.degree()
    mesh = fem_func.ufl_function_space().mesh
    
    basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype, q_dim)
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(cells)
    fem_func.x.array[:] = expr_eval.flatten()[:]
    # fem_func.x.scatter_forward()

# Defining the function to interpolate a function defined over quadrature elements
# def interpolate_quadrature(ufl_expr, q_dim, mesh):
#     basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
#     quadrature_points, weights = basix.make_quadrature(basix_celltype, q_dim)
#     map_c = mesh.topology.index_map(mesh.topology.dim)
#     num_cells = map_c.size_local + map_c.num_ghosts
#     cells = np.arange(0, num_cells, dtype=np.int32)

#     expr_expr = fem.Expression(ufl_expr, quadrature_points)
#     expr_eval = expr_expr.eval(cells)
#     return expr_eval

# Interpolate an expression on an element
def interpolate_quadrature_on_element(expr, function_space:fem.FunctionSpace, mesh, cell_number):
    assert function_space.family() == 'Quadrature'
    
    q_dim = function_space.degree()
    
    basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
    quadrature_points, _ = basix.make_quadrature(basix_celltype, q_dim)

    expr_expr = fem.Expression(expr, quadrature_points)
    expr_eval = expr_expr.eval(cell_number)
    return expr_eval

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

from dolfinx.geometry import (BoundingBoxTree, compute_colliding_cells, compute_collisions)


# Defining a cell containing (Ri, 0) point, where we calculate a value of u
def find_cell_by_point(mesh, point):
    cells = []
    points_on_proc = []
    tree = BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions(tree, point)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, point)
    for i, point in enumerate(point):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    return cells, points_on_proc