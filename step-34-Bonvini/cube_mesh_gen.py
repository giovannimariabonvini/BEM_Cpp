import gmsh
import sys

# Initialize Gmsh
gmsh.initialize()

# Set the Gmsh model name
gmsh.model.add("cube_surface")

mesh_size = 0.0625

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

# Set the meshing algorithm to recombine for quadrilateral elements on the surfaces
for surface in [1, 2, 3, 4, 5, 6]:
    gmsh.model.mesh.setRecombine(2, surface)  # 2 indicates surface dimension

# Optionally, assign boundary IDs (if needed for the solver)
gmsh.model.addPhysicalGroup(2, [1], 1)  # Boundary ID 1 for the first face
gmsh.model.addPhysicalGroup(2, [2], 2)  # Boundary ID 2 for the second face
gmsh.model.addPhysicalGroup(2, [3], 3)  # Boundary ID 3 for the third face
gmsh.model.addPhysicalGroup(2, [4], 4)  # Boundary ID 4 for the fourth face
gmsh.model.addPhysicalGroup(2, [5], 5)  # Boundary ID 5 for the fifth face
gmsh.model.addPhysicalGroup(2, [6], 6)  # Boundary ID 6 for the sixth face

# Mesh the surfaces with quadrilateral elements
gmsh.model.mesh.generate(2)

# Optionally, write the mesh to a file (e.g., in .msh format)
gmsh.write("cube_mesh_4.msh")

# Display the mesh in the Gmsh GUI
gmsh.fltk.run()

# Finalize the Gmsh session
gmsh.finalize()
