# BEM_Cpp
C++ library for 3D Boundary Element Method.

This C++ script is used to solve the Laplace equation with collocation Boundary Element Method (BEM). 
The equation it solves and the problem formulation are widely discussed in the project report that you can find in the GitHub repository.
To properly compile the script you should install the finite element library deal.II in version 9.6.0. Its installation and use are well documented in their official website (https://www.dealii.org/). 

To run the simulation you should run the following commands from the terminal:
cmake .
make run

The parameters to specify the boundary conditions, mesh, convergence and linear solvers are passed to the code with a file called parameters.prm. The file contains comment to help the user specify the data of the.

Helper python files are present in the repository and can be used for the following purposes:
_mesh_gen.py files -> produce the appropriate mesh for the problem (sphere,torus,cube)
check_normals.py -> produce an output_normals.vtk file that contains a set of points. Visualize this file along with the mesh file in Paraview and if the points lie outside the mesh then the normals have the correct directions.

MESH REQUIREMENTS
If a user want to adopt a new geometry he must produce a mesh file with the following constraints:
1) only .msh and .vtk files are accepted;
2) all elements must be quads of 1st order (each element is formed by 4 points connected with 4 segments);
3) the normals to every element must be pointing outward with respect to the mesh interior so every surface be oriented accordingly (check_normals.py can be used to check the actual normal directions);

The test case 1 of the project report can be performed as follows:
1) leave the parameters.prm file as present in the GitHub repository.
2) form terminal: cmake .
3) from terminal: make run
4) 3d_boundary_solution_0.vtk 3d_boundary_solution_1.vtk 3d_boundary_solution_2.vtk files are produced and can be visualized in Paraview
5) a convergence_table.txt file is also produced
6) from terminal: python3 plot_convergence.py
This will produce two .png images with the L2 and Linf convergence rates for the problem
