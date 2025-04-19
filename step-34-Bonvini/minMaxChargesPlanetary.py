import meshio
import numpy as np

# 1) Leggi file
mesh = meshio.read("3d_boundary_solution_0.vtk")

# 2) Prendi phi_n (prima point_data poi cell_data)
phi_n = mesh.point_data.get("phi_n")
if phi_n is None:
    # mesh.cell_data è dict di liste, in legacy VTK di solito lista di un solo array
    cd = mesh.cell_data.get("phi_n")
    if cd: phi_n = cd[0]
if phi_n is None:
    raise KeyError("phi_n non trovato in point_data né in cell_data")

# 3) Min/max
print("Min phi_n:", np.min(phi_n))
print("Max phi_n:", np.max(phi_n))
