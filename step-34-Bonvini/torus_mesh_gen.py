import gmsh
import vtk
import numpy as np

gmsh.initialize()
# Create a new model
gmsh.model.add("torus")

# Parameters for the torus
r_major = 6  # Major radius (distance from the center of the hole to the center of the tube)
r_minor = 2  # Minor radius (radius of the tube)

# Create the torus geometry
gmsh.model.occ.addTorus(0, 0, 0, r_major, r_minor)
gmsh.model.occ.synchronize()

# Define mesh size field
mesh_size = 0.5
gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size)  # Set global mesh size factor

# Generate a 2D mesh with recombined quadrilaterals
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine triangles into quads
gmsh.model.mesh.recombine()

# Save the mesh as a temporary .vtk file
temp_vtk_path = "temp_torus_mesh.vtk"
gmsh.write(temp_vtk_path)
gmsh.fltk.run()
gmsh.finalize()

# Read and filter the mesh in VTK format
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(temp_vtk_path)
reader.Update()
grid = reader.GetOutput()

# Create a new unstructured grid for valid cells only
filtered_grid = vtk.vtkUnstructuredGrid()
filtered_grid.SetPoints(grid.GetPoints())

# Supported 2D cell types
valid_cell_types = [vtk.VTK_QUAD, vtk.VTK_TRIANGLE]

# Add valid cells to the new grid
for i in range(grid.GetNumberOfCells()):
    cell_type = grid.GetCellType(i)
    if cell_type in valid_cell_types:
        filtered_grid.InsertNextCell(grid.GetCellType(i), grid.GetCell(i).GetPointIds())

# Add CELL_DATA for MaterialID and ManifoldID
material_id_array = vtk.vtkIntArray()
material_id_array.SetName("MaterialID")
material_id_array.SetNumberOfValues(filtered_grid.GetNumberOfCells())
material_id_array.Fill(1)  # Default MaterialID value
filtered_grid.GetCellData().AddArray(material_id_array)

manifold_id_array = vtk.vtkIntArray()
manifold_id_array.SetName("ManifoldID")
manifold_id_array.SetNumberOfValues(filtered_grid.GetNumberOfCells())
manifold_id_array.Fill(0)  # Default ManifoldID value
filtered_grid.GetCellData().AddArray(manifold_id_array)

# Save the filtered mesh as a .vtk file for use with dealii
filtered_vtk_path = "torus_mesh_4.vtk"
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName(filtered_vtk_path)
writer.SetInputData(filtered_grid)
writer.SetFileVersion(42)  # Ensure compatibility with VTK 3.0
writer.Write()

# Verify the cell types in the filtered VTK file
final_cell_types = {}
for i in range(filtered_grid.GetNumberOfCells()):
    cell_type = filtered_grid.GetCellType(i)
    if cell_type not in final_cell_types:
        final_cell_types[cell_type] = 0
    final_cell_types[cell_type] += 1

print("Filtered cell types in the VTK file:")
for cell_type, count in final_cell_types.items():
    print(f" - {vtk.vtkCellTypes.GetClassNameFromTypeId(cell_type)} (ID {cell_type}): {count}")

print(f"Filtered VTK file saved to: {filtered_vtk_path}")

# Modify the .vtk file to set version 3.0 for compatibility with deal.II
with open(filtered_vtk_path, "r") as file:
    vtk_data = file.readlines()

# Update VTK version to 3.0
vtk_data[0] = "# vtk DataFile Version 3.0\n"

# Write the cleaned .vtk file
with open(filtered_vtk_path, "w") as file:
    file.writelines(vtk_data)
