import gmsh
import sys

# Initialize Gmsh
gmsh.initialize()

# Set the Gmsh model name
gmsh.model.add("cube_surface")

mesh_size = 0.125

# Define the corner points of the cube
gmsh.model.geo.addPoint(0, 0, 0, mesh_size, 1)
gmsh.model.geo.addPoint(1, 0, 0, mesh_size, 2)
gmsh.model.geo.addPoint(1, 1, 0, mesh_size, 3)
gmsh.model.geo.addPoint(0, 1, 0, mesh_size, 4)
gmsh.model.geo.addPoint(0, 0, 1, mesh_size, 5)
gmsh.model.geo.addPoint(1, 0, 1, mesh_size, 6)
gmsh.model.geo.addPoint(1, 1, 1, mesh_size, 7)
gmsh.model.geo.addPoint(0, 1, 1, mesh_size, 8)

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

# Define surfaces (the cube faces)
gmsh.model.geo.addCurveLoop([-4, -3, -2, -1], 1)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
gmsh.model.geo.addCurveLoop([1, 10, -5, -9], 3)
gmsh.model.geo.addCurveLoop([2, 11, -6, -10], 4)
gmsh.model.geo.addCurveLoop([3, 12, -7, -11], 5)
gmsh.model.geo.addCurveLoop([4, 9, -8, -12], 6)

# Add plane surfaces from the curve loops
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)
gmsh.model.geo.addPlaneSurface([3], 3)
gmsh.model.geo.addPlaneSurface([4], 4)
gmsh.model.geo.addPlaneSurface([5], 5)
gmsh.model.geo.addPlaneSurface([6], 6)

# Synchronize the model to generate the surfaces in Gmsh
gmsh.model.geo.synchronize()

# Apply transfinite meshing to the lines for uniform divisions
divisions = int(1 / mesh_size)  # Number of divisions based on mesh size
for line in range(1, 13):  # Apply to all edges of the cube
    gmsh.model.mesh.setTransfiniteCurve(line, divisions + 1)

# Apply transfinite meshing to the surfaces for structured quads
for surface in range(1, 7):  # Apply to all faces of the cube
    gmsh.model.mesh.setTransfiniteSurface(surface, "Left")  # "Left" ensures quad elements

# Recombine triangles into quadrilaterals for the surfaces
for surface in range(1, 7):
    gmsh.model.mesh.setRecombine(2, surface)

# Optionally, assign boundary IDs (if needed for the solver)
for surface in range(1, 7):
    gmsh.model.addPhysicalGroup(2, [surface], surface)  # Assign a unique ID to each surface

# Mesh the surfaces
gmsh.model.mesh.generate(2)

# Optionally, write the mesh to a file (e.g., in .msh format)
gmsh.write("cube_uniform.msh")

# Display the mesh in the Gmsh GUI
gmsh.fltk.run()

# Finalize the Gmsh session
gmsh.finalize()
