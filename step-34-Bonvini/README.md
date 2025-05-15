# 3D BEM Solver (step-34-Bonvini)

This repository implements a collocation Boundary Element Method solver for the 3D Laplace/Yukawa equation, extended from the deal.II **step-34** tutorial. It supports mixed Dirichlet–Neumann–Robin boundary conditions, arbitrary surface meshes, and both Laplace and Yukawa kernels. 

---

## Dependencies

- **deal.II 9.6.0** or later 
  - must be configured with **muParser** support 
  - installation guide: https://dealii.org/current/readme.html 
- **CMake ≥ 3.13.4** 
- A C++17–capable compiler 

---

## Installing deal.II
git clone https://github.com/dealii/dealii.git
cd dealii
mkdir build && cd build
cmake \
  -DDEAL_II_WITH_MUPARSER=ON \
  -DDEAL_II_WITH_PETSC=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/dealii-install 
  .. 
make -j$(nproc) install 
make test 

## Run a test case
First clone the repository with:

git clone https://github.com/giovannimariabonvini/BEM_Cpp.git 

Then to run Test case 3 (in reduced form: only first 6 refinements):

cd BEM_Cpp/step-34-Bonvini 
cmake . 
make run 

The test case will be the one specified by the BEM_Cpp/step-34-Bonvini/parameter.prm file. This test case coincide with Test case 3 (explained in the project report).
To run other test cases copy and paste the  parameter.prm file and .msh files contained in the folders "Test case 1" or "Test case 2" in the main folder BEM_Cpp/step-34-Bonvini then compile and run with make run.

To check convergence for Test case 3 do:

python plot_convergence.py

and then check the convergence in the 2 .png images that appear in the main folder.
