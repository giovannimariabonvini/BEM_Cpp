
"""
Run this file with the following command:
python check_normals.py mesh_filename output_normals.vtk

Then, open the output_normals.vtk file in ParaView along with the solution.
If the points in output_normals.vtk are all outside of the surface mesh, the normals are correct.
"""

import sys
import os
import numpy as np
import meshio

def compute_normal_for_quad(quad_points):
    """
    Compute a normalized normal for a quad element.
    Assumes that quad_points is a (4,3) array of the quad’s vertices
    ordered consistently (counterclockwise when viewed from outside).
    
    Here we compute the cross product of two edges (p1-p0 and p3-p0).
    """
    p0, p1, p2, p3 = quad_points
    edge1 = p1 - p0
    edge2 = p3 - p0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])
    return normal / norm

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_normals.py input_mesh.[msh|vtk] [output_file.vtk]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "normals_points.vtk"

    # Read the input mesh using meshio. It supports .msh and .vtk among others.
    try:
        mesh = meshio.read(input_file)
    except Exception as e:
        print("Error reading the mesh:", e)
        sys.exit(1)
    
    # Check that the mesh contains quad cells.
    if "quad" not in mesh.cells_dict:
        print("Error: Input mesh does not contain quad elements.")
        sys.exit(1)
    
    quads = mesh.cells_dict["quad"]  # Each row has 4 indices.
    points = mesh.points  # Coordinates of mesh points, shape (n_points, 3)

    output_points = []
    for cell in quads:
        # Get the coordinates of the quad's vertices.
        quad_pts = points[cell]  # shape (4,3)
        # Compute the center (average of the vertices).
        center = np.mean(quad_pts, axis=0)
        # Compute the normalized normal vector.
        normal = compute_normal_for_quad(quad_pts)
        # Form the output point: center + normal.
        pt_out = center + normal
        output_points.append(pt_out)
    
    output_points = np.array(output_points)
    
    # Create a new mesh with these points. Here, we create a "vertex" cell type.
    # Each point is an isolated vertex.
    vertex_cells = [("vertex", np.arange(len(output_points)).reshape(-1, 1))]
    output_mesh = meshio.Mesh(points=output_points, cells=vertex_cells)
    
    # Write the output mesh to a VTK file.
    try:
        meshio.write(output_file, output_mesh, file_format="vtk")
        print("Output saved to", output_file)
    except Exception as e:
        print("Error writing output:", e)

if __name__ == "__main__":
    main()
