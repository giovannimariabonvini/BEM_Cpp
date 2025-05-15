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
  -DCMAKE_INSTALL_PREFIX=~/dealii-install \
  .. \
make -j$(nproc) install \
make test 

## Run a test case
First clone the repository with:

git clone https://github.com/giovannimariabonvini/BEM_Cpp.git 

Then to run a test case:

cd BEM_Cpp/step-34-Bonvini \
cmake . \
make run 

The test case will be the one specified by the BEM_Cpp/step-34-Bonvini/parameter.prm file. This test case coincide with Test case 1 (explained in the report, also available in the repository)
To run other test cases copy and paste the content of parameter.prm file contained in the folders "multi_spheres_benchmark" or "screened_Poisson_sphere_benchmark" in the file BEM_Cpp/step-34-Bonvini/parameter.prm the compile and run with make run.

