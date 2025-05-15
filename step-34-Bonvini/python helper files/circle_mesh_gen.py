import gmsh
import math

# Initialize gmsh and add a new model.
gmsh.initialize()
gmsh.model.add("circle_perimeter")

# Define the mesh size (characteristic length)
lc = 0.1

# Circle parameters
radius = 1.0

# Compute the number of segments needed so that each segment is roughly of length lc.
# The circumference of the circle is 2*pi*radius.
num_points = int(math.ceil(2 * math.pi * radius / lc))
print("Number of points: ", num_points)

# Create points along the circle perimeter.
point_ids = []
for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    # z = 0 for a 2D circle
    pid = gmsh.model.geo.addPoint(x, y, 0, lc)
    point_ids.append(pid)

# Create straight (linear) line segments between consecutive points.
line_ids = []
for i in range(num_points):
    start = point_ids[i]
    end = point_ids[(i + 1) % num_points]  # wrap-around for the closed loop
    line_ids.append(gmsh.model.geo.addLine(start, end))

# Create a closed curve loop from the lines.
curve_loop = gmsh.model.geo.addCurveLoop(line_ids)

# Synchronize the CAD kernel with the gmsh model.
gmsh.model.geo.synchronize()

# Optionally, define a physical group for the 1D mesh (the circle perimeter).
gmsh.model.addPhysicalGroup(1, line_ids, tag=1)

# Generate a 1D mesh along the curve.
gmsh.model.mesh.generate(1)

# Write the mesh to a file.
gmsh.write("circle_mesh.msh")

# Optionally, launch the GUI to inspect the mesh.
gmsh.fltk.run()

# Finalize gmsh.
gmsh.finalize()
