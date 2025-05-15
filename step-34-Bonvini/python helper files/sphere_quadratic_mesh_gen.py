#!/usr/bin/env python3

import gmsh
import vtk
import numpy as np

gmsh.initialize()
gmsh.model.add("sphere")

radius = 1.0
# Make a sphere
gmsh.model.occ.addSphere(0, 0, 0, radius)
gmsh.model.occ.synchronize()

# Global mesh size factor
mesh_size = 0.25
gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size)

# Request second-order elements
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

# Generate a 2D surface mesh
gmsh.model.mesh.generate(2)

# Make all surfaces get recombined
gmsh.option.setNumber("Mesh.RecombineAll", 1)

# Set the recombination algorithm to something other than blossom(0).
# 1 = Simple, 2 = Simple full, 3 = Simple bloss
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

# Attempt to recombine
try:
    gmsh.model.mesh.recombine()
except Exception as e:
    print("Recombination failed:", e)
    # Possibly skip or fallback if needed

# Write to temp file
temp_vtk_path = "temp_sphere_mesh.vtk"
gmsh.write(temp_vtk_path)

# Optionally show the GUI
gmsh.fltk.run()
gmsh.finalize()

# Now filter in VTK
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(temp_vtk_path)
reader.Update()
grid = reader.GetOutput()

filtered_grid = vtk.vtkUnstructuredGrid()
filtered_grid.SetPoints(grid.GetPoints())

# Accept second-order quads/tris plus possible leftover linear
valid_cell_types = [
    vtk.VTK_QUADRATIC_QUAD, vtk.VTK_QUADRATIC_TRI,
    vtk.VTK_QUAD, vtk.VTK_TRIANGLE
]

for i in range(grid.GetNumberOfCells()):
    ctype = grid.GetCellType(i)
    if ctype in valid_cell_types:
        filtered_grid.InsertNextCell(ctype, grid.GetCell(i).GetPointIds())

# Add cell data arrays for MaterialID, ManifoldID
material_id_array = vtk.vtkIntArray()
material_id_array.SetName("MaterialID")
material_id_array.SetNumberOfValues(filtered_grid.GetNumberOfCells())
material_id_array.Fill(1)
filtered_grid.GetCellData().AddArray(material_id_array)

manifold_id_array = vtk.vtkIntArray()
manifold_id_array.SetName("ManifoldID")
manifold_id_array.SetNumberOfValues(filtered_grid.GetNumberOfCells())
manifold_id_array.Fill(0)
filtered_grid.GetCellData().AddArray(manifold_id_array)

# Final output
filtered_vtk_path = "sphere_mesh_quadratic.vtk"
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName(filtered_vtk_path)
writer.SetInputData(filtered_grid)
writer.SetFileVersion(42)
writer.Write()

# Print final cell types
final_cell_types = {}
for i in range(filtered_grid.GetNumberOfCells()):
    c = filtered_grid.GetCellType(i)
    final_cell_types[c] = final_cell_types.get(c,0) + 1

print("Filtered cell types in the VTK file:")
for ctype, count in final_cell_types.items():
    print(f"  - {vtk.vtkCellTypes.GetClassNameFromTypeId(ctype)} (ID {ctype}): {count}")

print(f"Filtered VTK file saved to: {filtered_vtk_path}")

# Fix the VTK version line
with open(filtered_vtk_path, "r") as f:
    lines = f.readlines()

lines[0] = "# vtk DataFile Version 3.0\n"

with open(filtered_vtk_path, "w") as f:
    f.writelines(lines)
