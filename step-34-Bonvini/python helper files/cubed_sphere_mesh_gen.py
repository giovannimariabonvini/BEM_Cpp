import gmsh, math

def corner_string(corners):
    return " ".join(str(c) for c in corners)

###############################################
# Step 1: Create the cube surface mesh
###############################################

gmsh.initialize()
gmsh.model.add("cube")

mesh_size = 0.17  # target mesh size

lc = 1.0  # characteristic length - Dont change this value

# Define 8 corner points of the cube [-1,1]^3
p1 = gmsh.model.geo.addPoint(-1, -1, -1, lc)
p2 = gmsh.model.geo.addPoint( 1, -1, -1, lc)
p3 = gmsh.model.geo.addPoint( 1,  1, -1, lc)
p4 = gmsh.model.geo.addPoint(-1,  1, -1, lc)
p5 = gmsh.model.geo.addPoint(-1, -1,  1, lc)
p6 = gmsh.model.geo.addPoint( 1, -1,  1, lc)
p7 = gmsh.model.geo.addPoint( 1,  1,  1, lc)
p8 = gmsh.model.geo.addPoint(-1,  1,  1, lc)

# Define 12 lines (edges)
l1  = gmsh.model.geo.addLine(p1, p2)
l2  = gmsh.model.geo.addLine(p2, p3)
l3  = gmsh.model.geo.addLine(p3, p4)
l4  = gmsh.model.geo.addLine(p4, p1)
l5  = gmsh.model.geo.addLine(p5, p6)
l6  = gmsh.model.geo.addLine(p6, p7)
l7  = gmsh.model.geo.addLine(p7, p8)
l8  = gmsh.model.geo.addLine(p8, p5)
l9  = gmsh.model.geo.addLine(p1, p5)
l10 = gmsh.model.geo.addLine(p2, p6)
l11 = gmsh.model.geo.addLine(p3, p7)
l12 = gmsh.model.geo.addLine(p4, p8)

# Define 6 surfaces (cube faces)
# Bottom face: p4 -> p3 -> p2 -> p1
cl1 = gmsh.model.geo.addCurveLoop([-l4, -l3, -l2, -l1])
s1  = gmsh.model.geo.addPlaneSurface([cl1])
# Top face: p5 -> p6 -> p7 -> p8
cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
s2  = gmsh.model.geo.addPlaneSurface([cl2])
# Front face: p1 -> p2 -> p6 -> p5
cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
s3  = gmsh.model.geo.addPlaneSurface([cl3])
# Back face: p3 -> p4 -> p8 -> p7
cl4 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
s4  = gmsh.model.geo.addPlaneSurface([cl4])
# Left face: p4 -> p1 -> p5 -> p8
cl5 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
s5  = gmsh.model.geo.addPlaneSurface([cl5])
# Right face: p2 -> p3 -> p7 -> p6
cl6 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
s6  = gmsh.model.geo.addPlaneSurface([cl6])

gmsh.model.geo.synchronize()

# Set structured (transfinite) meshing:
nDiv = round(5/mesh_size)  # number of divisions
for line in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]:
    gmsh.model.mesh.setTransfiniteCurve(line, nDiv + 1)

gmsh.model.mesh.setTransfiniteSurface(s1, corner_string([p1, p2, p3, p4]))
gmsh.model.mesh.setTransfiniteSurface(s2, corner_string([p5, p6, p7, p8]))
gmsh.model.mesh.setTransfiniteSurface(s3, corner_string([p1, p2, p6, p5]))
gmsh.model.mesh.setTransfiniteSurface(s4, corner_string([p4, p3, p7, p8]))
gmsh.model.mesh.setTransfiniteSurface(s5, corner_string([p1, p5, p8, p4]))
gmsh.model.mesh.setTransfiniteSurface(s6, corner_string([p2, p3, p7, p6]))

# Recombine surfaces to obtain quadrilateral elements.
for s in [s1, s2, s3, s4, s5, s6]:
    gmsh.model.mesh.setRecombine(2, s)

# Generate the 2D (surface) mesh.
gmsh.model.mesh.generate(2)

# Write mesh in MSH v2 (text) format for postprocessing.
gmsh.option.setNumber("Mesh.MshFileVersion", 2)
mesh_filename = "cube.msh"
gmsh.write(mesh_filename)
gmsh.finalize()


###############################################
# Step 2: Post-process the mesh file to transform nodes
#         and remove 1D (line) elements.
###############################################

def transform_node(x, y, z):
    # Compute each sqrt argument and clamp negatives to 0:
    arg_x = 1.0 - (y*y)/2.0 - (z*z)/2.0 + (y*y*z*z)/3.0
    if arg_x < 0:
        arg_x = 0.0
    new_x = x * math.sqrt(arg_x)
    
    arg_y = 1.0 - (z*z)/2.0 - (x*x)/2.0 + (z*z*x*x)/3.0
    if arg_y < 0:
        arg_y = 0.0
    new_y = y * math.sqrt(arg_y)
    
    arg_z = 1.0 - (x*x)/2.0 - (y*y)/2.0 + (x*x*y*y)/3.0
    if arg_z < 0:
        arg_z = 0.0
    new_z = z * math.sqrt(arg_z)
    
    return new_x, new_y, new_z

# Read the original MSH file produced by Gmsh (e.g., "cube.msh")
with open("cube.msh", "r") as f:
    lines = f.readlines()

output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Copy the MeshFormat block as is.
    if line.strip() == "$MeshFormat":
        output_lines.append(line)
        i += 1
        while i < len(lines) and lines[i].strip() != "$EndMeshFormat":
            output_lines.append(lines[i])
            i += 1
        if i < len(lines):
            output_lines.append(lines[i])  # "$EndMeshFormat"
        i += 1
    # Process the $Nodes block.
    elif line.strip() == "$Nodes":
        output_lines.append(line)  # "$Nodes"
        i += 1
        node_count = int(lines[i].strip())
        output_lines.append(lines[i])
        i += 1
        # Process exactly node_count node lines.
        for j in range(node_count):
            parts = lines[i].split()
            tag = parts[0]
            x, y, z = map(float, parts[1:4])
            new_x, new_y, new_z = transform_node(x, y, z)
            output_lines.append(f"{tag} {new_x} {new_y} {new_z}\n")
            i += 1
        output_lines.append(lines[i])  # "$EndNodes"
        i += 1
    # Process the $Elements block: remove 1D (line) elements.
    elif line.strip() == "$Elements":
        output_lines.append(line)  # "$Elements"
        i += 1
        elem_count = int(lines[i].strip())
        i += 1
        elem_lines = []
        for j in range(elem_count):
            elem_line = lines[i]
            tokens = elem_line.split()
            # In MSH v2, token 1 is the element type (1 = line, 2 = triangle, 3 = quadr, etc.)
            if int(tokens[1]) != 1:
                elem_lines.append(elem_line)
            i += 1
        output_lines.append(str(len(elem_lines)) + "\n")
        output_lines.extend(elem_lines)
        output_lines.append(lines[i])  # "$EndElements"
        i += 1
    else:
        output_lines.append(line)
        i += 1

# Write the reassembled file.
with open("cubed_sphere_8.msh", "w") as f:
    f.writelines(output_lines)

print("Postprocessed mesh written to 'cubed_sphere.msh'")
