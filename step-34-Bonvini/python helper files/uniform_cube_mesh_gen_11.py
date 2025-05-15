#!/usr/bin/env python3
import gmsh
import sys

# Initialize Gmsh
gmsh.initialize()

# Set the Gmsh model name
gmsh.model.add("cube_surface")

mesh_size = 0.125

# Define the corner points of the cube (-1,1)^3
gmsh.model.geo.addPoint(-1, -1, -1, mesh_size, 1)
gmsh.model.geo.addPoint( 1, -1, -1, mesh_size, 2)
gmsh.model.geo.addPoint( 1,  1, -1, mesh_size, 3)
gmsh.model.geo.addPoint(-1,  1, -1, mesh_size, 4)
gmsh.model.geo.addPoint(-1, -1,  1, mesh_size, 5)
gmsh.model.geo.addPoint( 1, -1,  1, mesh_size, 6)
gmsh.model.geo.addPoint( 1,  1,  1, mesh_size, 7)
gmsh.model.geo.addPoint(-1,  1,  1, mesh_size, 8)

# Define lines connecting the points to create cube edges
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)
gmsh.model.geo.addLine(1, 5, 9)
gmsh.model.geo.addLine(2, 6, 10)
gmsh.model.geo.addLine(3, 7, 11)
gmsh.model.geo.addLine(4, 8, 12)

# Define surfaces (the cube faces) via curve loops.
# Note: The ordering of the curves must be chosen consistently.
gmsh.model.geo.addCurveLoop([-4, -3, -2, -1], 1)         # Bottom face (z=-1)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)              # Top face (z=1)
gmsh.model.geo.addCurveLoop([1, 10, -5, -9], 3)           # Front face (y=-1)
gmsh.model.geo.addCurveLoop([2, 11, -6, -10], 4)          # Right face (x=1)
gmsh.model.geo.addCurveLoop([3, 12, -7, -11], 5)          # Back face (y=1)
gmsh.model.geo.addCurveLoop([4, 9, -8, -12], 6)           # Left face (x=-1)

# Create plane surfaces from the curve loops
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)
gmsh.model.geo.addPlaneSurface([3], 3)
gmsh.model.geo.addPlaneSurface([4], 4)
gmsh.model.geo.addPlaneSurface([5], 5)
gmsh.model.geo.addPlaneSurface([6], 6)

# Synchronize the geometric model with the Gmsh kernel
gmsh.model.geo.synchronize()

# For a cube spanning from -1 to 1, the side length is 2.
divisions = int(2 / mesh_size)  # Number of divisions along each edge

# Apply transfinite meshing to all edges (lines)
for line in range(1, 13):
    gmsh.model.mesh.setTransfiniteCurve(line, divisions + 1)

# Apply transfinite meshing to the surfaces for structured quad meshes
for surface in range(1, 7):
    gmsh.model.mesh.setTransfiniteSurface(surface, "Left")  # "Left" creates quad elements

# Recombine triangular elements into quadrilaterals on the surfaces
for surface in range(1, 7):
    gmsh.model.mesh.setRecombine(2, surface)

# Optionally, assign physical groups to the surfaces (each gets a unique ID)
for surface in range(1, 7):
    gmsh.model.addPhysicalGroup(2, [surface], surface)

# Generate the 2D mesh (for the surfaces)
gmsh.model.mesh.generate(2)

# Write the mesh to a file, e.g., in .msh format
gmsh.write("cube_uniform_11.msh")

# Launch the Gmsh GUI (optional)
gmsh.fltk.run()

# Finalize the Gmsh session
gmsh.finalize()
