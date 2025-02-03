import gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("cube_surface")

mesh_size = 0.25

# Define the corner points of the cube
gmsh.model.geo.addPoint(0, 0, 0, mesh_size, 1)
gmsh.model.geo.addPoint(1, 0, 0, mesh_size, 2)
gmsh.model.geo.addPoint(1, 1, 0, mesh_size, 3)
gmsh.model.geo.addPoint(0, 1, 0, mesh_size, 4)
gmsh.model.geo.addPoint(0, 0, 1, mesh_size, 5)
gmsh.model.geo.addPoint(1, 0, 1, mesh_size, 6)
gmsh.model.geo.addPoint(1, 1, 1, mesh_size, 7)
gmsh.model.geo.addPoint(0, 1, 1, mesh_size, 8)

# Define lines
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

# Define surfaces (cube faces)
gmsh.model.geo.addCurveLoop([-4, -3, -2, -1], 1)  # Face z=0
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)      # Face z=1
gmsh.model.geo.addCurveLoop([1, 10, -5, -9], 3)   # Face y=0
gmsh.model.geo.addCurveLoop([3, 12, -7, -11], 4)  # Face y=1
gmsh.model.geo.addCurveLoop([4, 9, -8, -12], 5)   # Face x=0
gmsh.model.geo.addCurveLoop([2, 11, -6, -10], 6)  # Face x=1



# Add plane surfaces
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([2], 2)
gmsh.model.geo.addPlaneSurface([3], 3)  
gmsh.model.geo.addPlaneSurface([4], 4)
gmsh.model.geo.addPlaneSurface([5], 5)  
gmsh.model.geo.addPlaneSurface([6], 6)

# Synchronize the model
gmsh.model.geo.synchronize()

# Apply transfinite meshing
divisions = int(1 / mesh_size)
for line in range(1, 13):
    gmsh.model.mesh.setTransfiniteCurve(line, divisions + 1)

for surface in range(1, 7):
    gmsh.model.mesh.setTransfiniteSurface(surface, "Left")
    gmsh.model.mesh.setRecombine(2, surface)

# Assign boundary IDs
neumann_faces = [5, 6]  # x = 0 (face ID 5) and x = 1 (face ID 6)
dirichlet_faces = [1, 2, 3, 4]  # Other 4 faces

gmsh.model.addPhysicalGroup(2, neumann_faces, 10)  # Neumann faces are assigned boundary_id = 10
gmsh.model.addPhysicalGroup(2, dirichlet_faces, 20)  # Dirichlet faces are assigned boundary_id = 20

# Generate and save mesh
gmsh.model.mesh.generate(2)
gmsh.write("cube_with_bc.msh")

# Show in GUI
gmsh.fltk.run()
gmsh.finalize()
