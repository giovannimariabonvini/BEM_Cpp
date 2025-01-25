import numpy as np
import meshio


def calculate_normals(mesh):
    """
    Calculate the normals for each quadrilateral or triangular element in the mesh.
    """
    points = mesh.points
    normals_by_block = []

    # Loop through all cell blocks
    for cell in mesh.cells:
        cell_type, cell_data = cell.type, cell.data

        if cell_type in ["triangle", "quad"]:
            normals = []
            for element in cell_data:
                # Get the vertices of the element
                p0, p1, p2 = points[element[:3]]

                # Calculate two edge vectors
                v1 = p1 - p0
                v2 = p2 - p0

                # Compute the cross product to get the normal
                normal = np.cross(v1, v2)
                normal /= np.linalg.norm(normal)  # Normalize the normal vector

                normals.append(normal)

            normals_by_block.append(np.array(normals))
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    return normals_by_block


def write_mesh_with_normals(mesh, normals_by_block, output_file):
    """
    Write the mesh with normals included as CellData in a VTU file.
    """
    # Create a list for cell data
    cell_data = {"Normals": []}

    # Add normals for each block
    for normals in normals_by_block:
        cell_data["Normals"].append(normals)

    # Write the mesh with normals to a VTU file
    meshio.write_points_cells(
        output_file,
        mesh.points,
        mesh.cells,
        cell_data=cell_data
    )


# Input and output file paths
input_mesh_file = "cube_uniform_quads_bad_normals.msh"  # Replace with your input mesh file
output_vtk_file = "mesh_with_normals.vtu"

# Read the mesh
mesh = meshio.read(input_mesh_file)

# Calculate normals
normals_by_block = calculate_normals(mesh)

# Write the mesh with normals to a VTU file
write_mesh_with_normals(mesh, normals_by_block, output_vtk_file)

print(f"Mesh with normals written to {output_vtk_file}. Open this file in ParaView to view the normals.")
