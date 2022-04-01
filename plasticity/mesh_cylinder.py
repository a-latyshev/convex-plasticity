#!/usr/bin/env python3

from mpi4py import MPI
import json
from pathlib import Path


def mesh_cylinder(Ri, Re, lc, tdim, order=1, msh_file=None, comm=MPI.COMM_WORLD):

    """
    Create mesh of 3d tensile test specimen according to ISO 6892-1:2019 using the Python API of Gmsh.
    """
    # Perform Gmsh work only on rank = 0

    facet_tag_names = {"internal": 18, "external": 19, "bottom": 20, "left": 21}

    tag_names = {"facets": facet_tag_names}

    if comm.rank == 0:

        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.optimize("Netgen")
        model = gmsh.model()
        model.add("Cylinder")
        model.setCurrent("Cylinder")

        p0 = model.geo.addPoint(0.0, 0.0, 0, lc, tag=0)
        p1 = model.geo.addPoint(Ri, 0.0, 0, lc, tag=1)
        p2 = model.geo.addPoint(0.0, Ri, 0.0, lc, tag=2)
        p5 = model.geo.addPoint(Re, 0.0, 0, lc, tag=5)
        p6 = model.geo.addPoint(0.0, Re, 0.0, lc, tag=6)

        top_right_hole_int = model.geo.addCircleArc(p2, p0, p1, tag=18)
        top_right_hole_ext = model.geo.addCircleArc(p5, p0, p6, tag=19)

        bottom = model.geo.addLine(p1, p5, tag=20)
        left = model.geo.addLine(p6, p2, tag=21)

        cloop1 = model.geo.addCurveLoop(
            [
                bottom,
                top_right_hole_ext,
                left,
                top_right_hole_int,
            ]
        )
        # surface_1 =
        model.geo.addPlaneSurface([cloop1])
        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=17)
        model.setPhysicalName(tdim, 17, "Rectangle surface")

        # Set mesh size via points
        # gmsh.model.mesh.setSize(points, lc)  # heuristic

        # gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Define physical groups for subdomains (! target tag > 0)
        # domain = 1
        # gmsh.model.addPhysicalGroup(tdim, [v[1] for v in volumes], domain)
        # gmsh.model.setPhysicalName(tdim, domain, 'domain')
        for k, v in facet_tag_names.items():
            gmsh.model.addPhysicalGroup(tdim - 1, [v], tag=v)
            gmsh.model.setPhysicalName(tdim - 1, v, k)

        model.mesh.generate(tdim)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)
            with open(Path(msh_file).parent.joinpath("tag_names.json"), "w") as f:
                json.dump(tag_names, f)

    if comm.rank == 0:
        model_out = gmsh.model
    else:
        model_out = 0

    return model_out, tdim, tag_names
