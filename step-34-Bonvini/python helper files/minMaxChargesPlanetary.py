import meshio
import numpy as np

# Specify the desired solution file
mesh = meshio.read("3d_boundary_solution_0.vtk")

phi_n = mesh.point_data.get("phi_n")
if phi_n is None:
    cd = mesh.cell_data.get("phi_n")
    if cd: phi_n = cd[0]
if phi_n is None:
    raise KeyError("phi_n non trovato in point_data né in cell_data")

print("Min phi_n:", np.min(phi_n))
print("Max phi_n:", np.max(phi_n))
