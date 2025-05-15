import gmsh
import math

def transform_node(x, y, z):
    arg_x = 1.0 - (y*y)/2.0 - (z*z)/2.0 + (y*y*z*z)/3.0
    arg_x = max(arg_x, 0.0)
    new_x = x * math.sqrt(arg_x)
    
    arg_y = 1.0 - (z*z)/2.0 - (x*x)/2.0 + (z*z*x*x)/3.0
    arg_y = max(arg_y, 0.0)
    new_y = y * math.sqrt(arg_y)
    
    arg_z = 1.0 - (x*x)/2.0 - (y*y)/2.0 + (x*x*y*y)/3.0
    arg_z = max(arg_z, 0.0)
    new_z = z * math.sqrt(arg_z)
    
    return new_x, new_y, new_z

def create_cubed_sphere_mesh(sphere_radius, translation, target_element_size):
    gmsh.initialize()
    gmsh.model.add("cube")

    # Calculate adaptive mesh size based on sphere radius
    mesh_size = target_element_size / sphere_radius
    lc = 1.0  # Keep this unchanged
    nDiv = round(5 / mesh_size)

    # Define cube points
    p1 = gmsh.model.geo.addPoint(-1, -1, -1, lc)
    p2 = gmsh.model.geo.addPoint(1, -1, -1, lc)
    p3 = gmsh.model.geo.addPoint(1, 1, -1, lc)
    p4 = gmsh.model.geo.addPoint(-1, 1, -1, lc)
    p5 = gmsh.model.geo.addPoint(-1, -1, 1, lc)
    p6 = gmsh.model.geo.addPoint(1, -1, 1, lc)
    p7 = gmsh.model.geo.addPoint(1, 1, 1, lc)
    p8 = gmsh.model.geo.addPoint(-1, 1, 1, lc)

    # Define edges
    lines = [
        gmsh.model.geo.addLine(p1, p2),   # l1
        gmsh.model.geo.addLine(p2, p3),   # l2
        gmsh.model.geo.addLine(p3, p4),   # l3
        gmsh.model.geo.addLine(p4, p1),   # l4
        gmsh.model.geo.addLine(p5, p6),   # l5
        gmsh.model.geo.addLine(p6, p7),   # l6
        gmsh.model.geo.addLine(p7, p8),   # l7
        gmsh.model.geo.addLine(p8, p5),   # l8
        gmsh.model.geo.addLine(p1, p5),   # l9
        gmsh.model.geo.addLine(p2, p6),   # l10
        gmsh.model.geo.addLine(p3, p7),   # l11
        gmsh.model.geo.addLine(p4, p8)    # l12
    ]

    # Define surfaces
    cl1 = gmsh.model.geo.addCurveLoop([-lines[3], -lines[2], -lines[1], -lines[0]])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    cl2 = gmsh.model.geo.addCurveLoop([lines[4], lines[5], lines[6], lines[7]])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    cl3 = gmsh.model.geo.addCurveLoop([lines[0], lines[9], -lines[4], -lines[8]])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    cl4 = gmsh.model.geo.addCurveLoop([lines[2], lines[11], -lines[6], -lines[10]])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    cl5 = gmsh.model.geo.addCurveLoop([lines[3], lines[8], -lines[7], -lines[11]])
    s5 = gmsh.model.geo.addPlaneSurface([cl5])
    cl6 = gmsh.model.geo.addCurveLoop([lines[1], lines[10], -lines[5], -lines[9]])
    s6 = gmsh.model.geo.addPlaneSurface([cl6])

    gmsh.model.geo.synchronize()

    # Set structured meshing
    for line in lines:
        gmsh.model.mesh.setTransfiniteCurve(line, nDiv + 1)

    surfaces = [s1, s2, s3, s4, s5, s6]
    corners = [
        [p1, p2, p3, p4], [p5, p6, p7, p8],
        [p1, p2, p6, p5], [p4, p3, p7, p8],
        [p1, p5, p8, p4], [p2, p3, p7, p6]
    ]
    for s, c in zip(surfaces, corners):
        gmsh.model.mesh.setTransfiniteSurface(s, " ".join(map(str, c)))
        gmsh.model.mesh.setRecombine(2, s)

    gmsh.model.mesh.generate(2)

    # Get nodes and elements
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2)

    # Process nodes with scaling and translation
    nodes = []
    for tag, x, y, z in zip(node_tags, coords[::3], coords[1::3], coords[2::3]):
        nx, ny, nz = transform_node(x, y, z)
        tx, ty, tz = translation
        nodes.append((
            tag,
            nx * sphere_radius + tx,
            ny * sphere_radius + ty,
            nz * sphere_radius + tz
        ))

    # Process elements (quadrilaterals only)
    elements = []
    for elem_type, tags, nodes_in_elem in zip(elem_types, elem_tags, elem_nodes):
        if elem_type == 3:  # Quadrilateral elements (type 3 in Gmsh)
            for tag, elem_nodes in zip(tags, [nodes_in_elem[i:i+4] for i in range(0, len(nodes_in_elem), 4)]):
                elements.append((tag, elem_type, elem_nodes))

    gmsh.finalize()
    return nodes, elements

# Main script
all_nodes = []
all_elements = []
current_max_node_tag = 0
current_max_elem_tag = 0
target_element_size = 0.25 # Uniform physical element size for all spheres

# Create central sphere (radius 3 at origin)
nodes, elements = create_cubed_sphere_mesh(3, (0, 0, 0), target_element_size)
node_tag_map = {tag: current_max_node_tag + tag for tag, _, _, _ in nodes}
all_nodes.extend([(node_tag_map[tag], x, y, z) for tag, x, y, z in nodes])
current_max_node_tag = max(node_tag_map.values())

for elem_tag, elem_type, elem_nodes in elements:
    new_elem_tag = current_max_elem_tag + elem_tag
    adjusted_nodes = [node_tag_map[n] for n in elem_nodes]
    all_elements.append((new_elem_tag, elem_type, adjusted_nodes))
current_max_elem_tag = max([elem[0] for elem in all_elements]) if all_elements else 0

# Create 10 surrounding spheres (radius 1)
n_surrounding = 10
for i in range(n_surrounding):
    theta = 2 * math.pi * i / n_surrounding
    tx = 5 * math.cos(theta)
    ty = 5 * math.sin(theta)
    
    nodes, elements = create_cubed_sphere_mesh(1, (tx, ty, 0), target_element_size)
    
    node_tag_map = {tag: current_max_node_tag + tag for tag, _, _, _ in nodes}
    all_nodes.extend([(node_tag_map[tag], x, y, z) for tag, x, y, z in nodes])
    current_max_node_tag = max(node_tag_map.values())
    
    for elem_tag, elem_type, elem_nodes in elements:
        new_elem_tag = current_max_elem_tag + elem_tag
        adjusted_nodes = [node_tag_map[n] for n in elem_nodes]
        all_elements.append((new_elem_tag, elem_type, adjusted_nodes))
    current_max_elem_tag = max([elem[0] for elem in all_elements])

# Write combined mesh file
with open("planetary_spheres_4.msh", "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    
    f.write("$Nodes\n")
    f.write(f"{len(all_nodes)}\n")
    for tag, x, y, z in all_nodes:
        f.write(f"{tag} {x} {y} {z}\n")
    f.write("$EndNodes\n")
    
    f.write("$Elements\n")
    f.write(f"{len(all_elements)}\n")
    for elem_tag, elem_type, nodes in all_elements:
        f.write(f"{elem_tag} {elem_type} 0 " + " ".join(map(str, nodes)) + "\n")
    f.write("$EndElements\n")

print("Mesh generation complete: planetary_spheres_n.msh")