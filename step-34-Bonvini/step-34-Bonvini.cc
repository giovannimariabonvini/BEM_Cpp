/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2009 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Luca Heltai, Cataldo Manigrasso, 2009
 */


// @sect3{Include files}

// The program starts with including a bunch of include files that we will use
// in the various parts of the program. Most of them have been discussed in
// previous tutorials already:
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/signaling_nan.h>

// And here are a few C++ standard header files that we will need:
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace Step34
{
  using namespace dealii;


  // @sect3{Single and double layer operator kernels}

  // First, let us define a bit of the boundary integral equation machinery.

  // The following two functions are the actual calculations of the single and
  // double layer potential kernels, that is $G$ and $\nabla G$. They are well
  // defined only if the vector $R = \mathbf{y}-\mathbf{x}$ is different from
  // zero.
  namespace LaplaceKernel
  {
    template <int dim>
    double single_layer(const Tensor<1, dim> &R)
    {
      switch (dim)
        {
          case 2:
            return (-std::log(R.norm()) / (2 * numbers::PI));

          case 3:
            return (1. / (R.norm() * 4 * numbers::PI));

          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
    }



    template <int dim>
    Tensor<1, dim> double_layer(const Tensor<1, dim> &R)
    {
      switch (dim)
        {
          case 2:
            return R / (-2 * numbers::PI * R.norm_square());
          case 3:
            return R / (-4 * numbers::PI * R.norm_square() * R.norm());

          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
    }
  } // namespace LaplaceKernel


  // @sect3{The BEMProblem class}

  // The structure of a boundary element method code is very similar to the
  // structure of a finite element code, and so the member functions of this
  // class are like those of most of the other tutorial programs. In
  // particular, by now you should be familiar with reading parameters from an
  // external file, and with the splitting of the different tasks into
  // different modules. The same applies to boundary element methods, and we
  // won't comment too much on them, except on the differences.
  template <int dim>
  class BEMProblem
  {
  public:
    BEMProblem(const unsigned int fe_degree      = 1,
               const unsigned int mapping_degree = 1);

    void run();

  private:
    void read_parameters(const std::string &filename);

    void refine_and_resize();

    void read_mesh(const std::string &mesh_file);

    // The only really different function that we find here is the assembly
    // routine. We wrote this function in the most possible general way, in
    // order to allow for easy generalization to higher order methods and to
    // different fundamental solutions (e.g., Stokes or Maxwell).
    //
    // The most noticeable difference is the fact that the final matrix is
    // full, and that we have a nested loop inside the usual loop on cells
    // that visits all support points of the degrees of freedom.  Moreover,
    // when the support point lies inside the cell which we are visiting, then
    // the integral we perform becomes singular.
    //
    // The practical consequence is that we have two sets of quadrature
    // formulas, finite element values and temporary storage, one for standard
    // integration and one for the singular integration, which are used where
    // necessary.
    void assemble_system();

    // There are two options for the solution of this problem. The first is to
    // use a direct solver, and the second is to use an iterative solver. We
    // opt for the second option.
    //
    // The matrix that we assemble is not symmetric, and we opt to use the
    // GMRES method; however the construction of an efficient preconditioner
    // for boundary element methods is not a trivial issue. Here we use a non
    // preconditioned GMRES solver. The options for the iterative solver, such
    // as the tolerance, the maximum number of iterations, are selected
    // through the parameter file.
    void solve_system();

    void retrieve_solution();

    void set_boundary_flags();

    void recombine_matrices();

    void release_memory();

    void find_mesh_size();

    // bool is_point_exterior(const Point<dim> &p);

    // Once we obtained the solution, we compute the $L^2$ error of the
    // computed potential as well as the $L^\infty$ error of the approximation
    // of the solid angle. The mesh we are using is an approximation of a
    // smooth curve, therefore the computed diagonal matrix of fraction of
    // angles or solid angles $\alpha(\mathbf{x})$ should be constantly equal
    // to $\frac 12$. In this routine we output the error on the potential and
    // the error in the approximation of the computed angle. Notice that the
    // latter error is actually not the error in the computation of the angle,
    // but a measure of how well we are approximating the sphere and the
    // circle.
    //
    // Experimenting a little with the computation of the angles gives very
    // accurate results for simpler geometries. To verify this you can comment
    // out, in the read_domain() method, the tria.set_manifold(1, manifold)
    // line, and check the alpha that is generated by the program. By removing
    // this call, whenever the mesh is refined new nodes will be placed along
    // the straight lines that made up the coarse mesh, rather than be pulled
    // onto the surface that we really want to approximate. In the three
    // dimensional case, the coarse grid of the sphere is obtained starting
    // from a cube, and the obtained values of alphas are exactly $\frac 12$
    // on the nodes of the faces, $\frac 34$ on the nodes of the edges and
    // $\frac 78$ on the 8 nodes of the vertices.
    void compute_errors(const unsigned int cycle);

    // Once we obtained a solution on the codimension one domain, we want to
    // interpolate it to the rest of the space. This is done by performing
    // again the convolution of the solution with the kernel in the
    // compute_exterior_solution() function.
    //
    // We would like to plot the velocity variable which is the gradient of
    // the potential solution. The potential solution is only known on the
    // boundary, but we use the convolution with the fundamental solution to
    // interpolate it on a standard dim dimensional continuous finite element
    // space. The plot of the gradient of the extrapolated solution will give
    // us the velocity we want.
    //
    // In addition to the solution on the exterior domain, we also output the
    // solution on the domain's boundary in the output_results() function, of
    // course.

    void compute_exterior_solution();

    void output_results(const unsigned int cycle);

    // To allow for dimension independent programming, we specialize this
    // single function to extract the singular quadrature formula needed to
    // integrate the singular kernels in the interior of the cells.
    Quadrature<dim - 1> get_singular_quadrature(
      const typename DoFHandler<dim - 1, dim>::active_cell_iterator &cell,
      const unsigned int index) const;


    // The usual deal.II classes can be used for boundary element methods by
    // specifying the "codimension" of the problem. This is done by setting
    // the optional second template arguments to Triangulation, FiniteElement
    // and DoFHandler to the dimension of the embedding space. In our case we
    // generate either 1 or 2 dimensional meshes embedded in 2 or 3
    // dimensional spaces.
    //
    // The optional argument by default is equal to the first argument, and
    // produces the usual finite element classes that we saw in all previous
    // examples.
    //
    // The class is constructed in a way to allow for arbitrary order of
    // approximation of both the domain (through high order mappings) and the
    // finite element space. The order of the finite element space and of the
    // mapping can be selected in the constructor of the class.

    Triangulation<dim - 1, dim> tria;
    const FE_Q<dim - 1, dim>    fe;
    DoFHandler<dim - 1, dim>    dof_handler;
    MappingQ<dim - 1, dim>      mapping;
    std::vector<std::string> mesh_filenames;

    // In BEM methods, the matrix that is generated is dense. Depending on the
    // size of the problem, the final system might be solved by direct LU
    // decomposition, or by iterative methods. In this example we use an
    // unpreconditioned GMRES method. Building a preconditioner for BEM method
    // is non trivial, and we don't treat this subject here.

    FullMatrix<double> system_matrix;
    Vector<double>     system_rhs;

    FullMatrix<double> H;
    FullMatrix<double> G;

    // The next two variables will denote the solution $\phi$ as well as a
    // vector that will hold the values of $\alpha(\mathbf x)$ (the fraction
    // of $\Omega$ visible from a point $\mathbf x$) at the support points of
    // our shape functions.

    Vector<double> phi;
    Vector<double> phi_n;
    Vector<double> alpha;
    Vector<double> ls_solution;

    std::vector<bool> assign_dirichlet;
    std::vector<bool> assign_neumann;

    Vector<double> boundary_type;

    bool exterior_integration_domain;
    double phi_at_infinity;

    // const unsigned int bc_type = 1;

    // The convergence table is used to output errors in the exact solution
    // and in the computed alphas.

    ConvergenceTable convergence_table;

    // The following variables are the ones that we fill through a parameter
    // file.  The new objects that we use in this example are the
    // Functions::ParsedFunction object and the QuadratureSelector object.
    //
    // The Functions::ParsedFunction class allows us to easily and quickly
    // define new function objects via parameter files, with custom
    // definitions which can be very complex (see the documentation of that
    // class for all the available options).
    //
    // We will allocate the quadrature object using the QuadratureSelector
    // class that allows us to generate quadrature formulas based on an
    // identifying string and on the possible degree of the formula itself. We
    // used this to allow custom selection of the quadrature formulas for the
    // standard integration, and to define the order of the singular
    // quadrature rule.
    //
    // We also define a couple of parameters which are used in case we wanted
    // to extend the solution to the entire domain.

    Functions::ParsedFunction<dim> exact_solution_phi;
    Functions::ParsedFunction<dim> exact_solution_phi_n;
    Functions::ParsedFunction<dim> neumann_function;
    Functions::ParsedFunction<dim> dirichlet_function;
    // Since Neumann and Dirichlet regions are complementary we only need to specify one of the two
    Functions::ParsedFunction<dim> neumann_boundary_region; 

    unsigned int                         singular_quadrature_order;
    std::shared_ptr<Quadrature<dim - 1>> quadrature;

    SolverControl solver_control;

    unsigned int external_refinement;

    bool run_in_this_dimension;
    bool extend_solution;
    double mesh_size;
  };


  // @sect4{BEMProblem::BEMProblem and BEMProblem::read_parameters}

  // The constructor initializes the various object in much the same way as
  // done in the finite element programs such as step-4 or step-6. The only
  // new ingredient here is the ParsedFunction object, which needs, at
  // construction time, the specification of the number of components.
  //
  // For the exact solution the number of vector components is one, and no
  // action is required since one is the default value for a ParsedFunction
  // object. The wind, however, requires dim components to be
  // specified. Notice that when declaring entries in a parameter file for the
  // expression of the Functions::ParsedFunction, we need to specify the
  // number of components explicitly, since the function
  // Functions::ParsedFunction::declare_parameters is static, and has no
  // knowledge of the number of components.
  template <int dim>
  BEMProblem<dim>::BEMProblem(const unsigned int fe_degree,
                              const unsigned int mapping_degree)
    : fe(fe_degree)
    , dof_handler(tria)
    , mapping(mapping_degree)
    , singular_quadrature_order(5)
    , external_refinement(5)
    , run_in_this_dimension(true)
    , extend_solution(true)
  {}


  template <int dim>
  void BEMProblem<dim>::read_parameters(const std::string &filename)
  {
    deallog << std::endl
            << "Parsing parameter file " << filename << std::endl
            << "for a " << dim << " dimensional simulation. " << std::endl;

    ParameterHandler prm;
    
    prm.declare_entry("Extend solution on the -2,2 box", "true", Patterns::Bool());
    prm.declare_entry("External refinement", "5", Patterns::Integer());
    prm.declare_entry("Run 2d simulation", "true", Patterns::Bool());
    prm.declare_entry("Run 3d simulation", "true", Patterns::Bool());
    prm.declare_entry("Mesh filenames",
                  "sphere_mesh.msh",
                  Patterns::Anything(),
                  "Semicolon-separated list of mesh filenames.");
    prm.declare_entry("Exterior domain", "true", Patterns::Bool());
    prm.declare_entry("Infinity Dirichlet value", "0.0", Patterns::Double());

    prm.enter_subsection("Quadrature rules");
    {
      prm.declare_entry(
        "Quadrature type",
        "gauss",
        Patterns::Selection(
          QuadratureSelector<(dim - 1)>::get_quadrature_names()));
      prm.declare_entry("Quadrature order", "4", Patterns::Integer());
      prm.declare_entry("Singular quadrature order", "5", Patterns::Integer());
    }
    prm.leave_subsection();

    // For both two and three dimensions, we set the default input data to be
    // such that the solution is $x+y$ or $x+y+z$. The actually computed
    // solution will have value zero at infinity. In this case, this coincide
    // with the exact solution, and no additional corrections are needed, but
    // you should be aware of the fact that we arbitrarily set $\phi_\infty$,
    // and the exact solution we pass to the program needs to have the same
    // value at infinity for the error to be computed correctly.
    //
    // The use of the Functions::ParsedFunction object is pretty straight
    // forward. The Functions::ParsedFunction::declare_parameters function
    // takes an additional integer argument that specifies the number of
    // components of the given function. Its default value is one. When the
    // corresponding Functions::ParsedFunction::parse_parameters method is
    // called, the calling object has to have the same number of components
    // defined here, otherwise an exception is thrown.
    //
    // When declaring entries, we declare both 2 and three dimensional
    // functions. However only the dim-dimensional one is ultimately
    // parsed. This allows us to have only one parameter file for both 2 and 3
    // dimensional problems.
    //
    // Notice that from a mathematical point of view, the wind function on the
    // boundary should satisfy the condition $\int_{\partial\Omega}
    // \mathbf{v}\cdot \mathbf{n} d \Gamma = 0$, for the problem to have a
    // solution. If this condition is not satisfied, then no solution can be
    // found, and the solver will not converge.

    prm.enter_subsection("Exact solution phi 2d");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "0.5*(x+y)");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi_n 2d");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "x+y");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi 3d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0.5*(x+y+z)");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi_n 3d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "x+y+z");
    }
    prm.leave_subsection();

    prm.enter_subsection("Neumann function 2d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "x+y");
    }
    prm.leave_subsection();

    prm.enter_subsection("Dirichlet function 2d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0.5*(x+y)");
    }
    prm.leave_subsection();

   prm.enter_subsection("Neumann region 2d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0.0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Neumann function 3d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "x+y+z");
    }
    prm.leave_subsection();

    prm.enter_subsection("Dirichlet function 3d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0.5*(x+y+z)");
    }
    prm.leave_subsection();

   prm.enter_subsection("Neumann region 3d");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0.0");
    }
    prm.leave_subsection();

    // In the solver section, we set all SolverControl parameters. The object
    // will then be fed to the GMRES solver in the solve_system() function.
    prm.enter_subsection("Solver");
    SolverControl::declare_parameters(prm);
    prm.leave_subsection();

    // After declaring all these parameters to the ParameterHandler object,
    // let's read an input file that will give the parameters their values. We
    // then proceed to extract these values from the ParameterHandler object:
    prm.parse_input(filename);

    external_refinement = prm.get_integer("External refinement");
    extend_solution     = prm.get_bool("Extend solution on the -2,2 box");

    const std::string filenames_str = prm.get("Mesh filenames");
    deallog << "Mesh files provided: " << filenames_str << std::endl;
    mesh_filenames = dealii::Utilities::split_string_list(filenames_str, ' ');
    
    exterior_integration_domain     = prm.get_bool("Exterior domain");
    phi_at_infinity     = prm.get_double("Infinity Dirichlet value");

    prm.enter_subsection("Quadrature rules");
    {
      quadrature = std::shared_ptr<Quadrature<dim - 1>>(
        new QuadratureSelector<dim - 1>(prm.get("Quadrature type"),
                                        prm.get_integer("Quadrature order")));
      singular_quadrature_order = prm.get_integer("Singular quadrature order");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi " + std::to_string(dim) + "d");
    {
      exact_solution_phi.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi_n " + std::to_string(dim) + "d");
    {
      exact_solution_phi_n.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Neumann function " + std::to_string(dim) + "d");
    {
      neumann_function.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Dirichlet function " + std::to_string(dim) + "d");
    {
      dirichlet_function.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Neumann region " + std::to_string(dim) + "d");
    {
      neumann_boundary_region.parse_parameters(prm);
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    solver_control.parse_parameters(prm);
    prm.leave_subsection();


    // Finally, here's another example of how to use parameter files in
    // dimension independent programming.  If we wanted to switch off one of
    // the two simulations, we could do this by setting the corresponding "Run
    // 2d simulation" or "Run 3d simulation" flag to false:
    run_in_this_dimension =
      prm.get_bool("Run " + std::to_string(dim) + "d simulation");
  }


  // @sect4{BEMProblem::read_domain}

  // A boundary element method triangulation is basically the same as a
  // (dim-1) dimensional triangulation, with the difference that the vertices
  // belong to a (dim) dimensional space.
  //
  // Some of the mesh formats supported in deal.II use by default three
  // dimensional points to describe meshes. These are the formats which are
  // compatible with the boundary element method capabilities of deal.II. In
  // particular we can use either UCD or GMSH formats. In both cases, we have
  // to be particularly careful with the orientation of the mesh, because,
  // unlike in the standard finite element case, no reordering or
  // compatibility check is performed here.  All meshes are considered as
  // oriented, because they are embedded in a higher dimensional space. (See
  // the documentation of the GridIn and of the Triangulation for further
  // details on orientation of cells in a triangulation.) In our case, the
  // normals to the mesh are external to both the circle in 2d or the sphere
  // in 3d.
  //
  // The other detail that is required for appropriate refinement of
  // the boundary element mesh is an accurate description of the
  // manifold that the mesh approximates. We already saw this
  // several times for the boundary of standard finite element meshes
  // (for example in step-5 and step-6), and here the principle and
  // usage is the same, except that the SphericalManifold class takes
  // an additional template parameter that specifies the embedding
  // space dimension.

  // @sect4{BEMProblem::refine_and_resize}

  // This function globally refines the mesh, distributes degrees of freedom,
  // and resizes matrices and vectors.

  template <int dim>
  void BEMProblem<dim>::refine_and_resize()
  {
    tria.refine_global(0); // Increase this number for more refinement (0 = no refinement)
    dof_handler.distribute_dofs(fe);

    const unsigned int n_dofs = dof_handler.n_dofs();

    system_matrix.reinit(n_dofs, n_dofs);
    H.reinit(n_dofs, n_dofs);
    G.reinit(n_dofs, n_dofs);

    system_rhs.reinit(n_dofs);
    phi.reinit(n_dofs);
    phi_n.reinit(n_dofs);
    alpha.reinit(n_dofs);
    ls_solution.reinit(n_dofs);
  }

  template <int dim>
  void BEMProblem<dim>::read_mesh(const std::string &mesh_file)
  {
    std::ifstream in(mesh_file);
    if (!in)
    {
        throw std::runtime_error("Error: " + mesh_file + " not found.");
    }

    GridIn<dim - 1, dim> gi;
    gi.attach_triangulation(tria);

    // Access the file extension
    std::filesystem::path meshPath(mesh_file);
    std::string ext = meshPath.extension().string();

    if (ext == ".msh")
      gi.read_msh(in);
    else if (ext == ".vtk")
      gi.read_vtk(in);
    else
      throw std::runtime_error("Error: Mesh file format not supported. Supported formats are .msh and .vtk");


    // Note: No manifold is set for imported meshes.
  }


  template <int dim>
  void BEMProblem<dim>::set_boundary_flags()
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
    assign_dirichlet.resize(n_dofs, false);
    assign_neumann.resize(n_dofs, false);

    std::vector<Point<dim>> support_points(n_dofs);
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    // Tolerance for the evaluation of the Neumann boundary region
    // const double tol = 1e-12;

    // Per ogni punto di supporto, valuta la funzione parsata:
    // se il valore è zero (entro tol) il punto è Neumann, altrimenti Dirichlet.
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      double value = neumann_boundary_region.value(support_points[i]);
      if(value == 0) //std::fabs(value) < tol
      {
        assign_neumann[i] = true;
      }
      else
      {
        assign_dirichlet[i] = true;
      }

      if (assign_neumann[i] && assign_dirichlet[i])
      {
        std::cout << "Error: Support point " << i << " is both Neumann and Dirichlet" << std::endl;
      }
      if(!assign_neumann[i] && !assign_dirichlet[i])
      {
        std::cout << "Error: Support point " << i << " is neither Neumann nor Dirichlet" << std::endl;
      }
    }
  
    // Count the true values in each vector
    unsigned int n_neumann = 0;
    unsigned int n_dirichlet = 0;
    boundary_type.reinit(n_dofs);
    boundary_type = -1;
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      if (assign_neumann[i])
      {
        boundary_type[i] = 0;
        ++n_neumann;
      }
      else if (assign_dirichlet[i])
      {
        boundary_type[i] = 1;
        ++n_dirichlet;
      } else
      {
        std::cerr << "Error: Boundary type not assigned to support point " << i << std::endl;
        return;
      }
    }

    deallog << "Boundary flags assigned: " 
              << n_neumann << " Neumann nodes and " 
              << n_dirichlet << " Dirichlet nodes" << std::endl;

  }




  // @sect4{BEMProblem::assemble_system}

  // The following is the main function of this program, assembling the matrix
  // that corresponds to the boundary integral equation.
  template <int dim>
  void BEMProblem<dim>::assemble_system()
  {
    // First we initialize an FEValues object with the quadrature formula for
    // the integration of the kernel in non singular cells. This quadrature is
    // selected with the parameter file, and needs to be quite precise, since
    // the functions we are integrating are not polynomial functions.
    FEValues<dim - 1, dim> fe_v(mapping,
                                fe,
                                *quadrature,
                                update_values | update_normal_vectors |
                                  update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points = fe_v.n_quadrature_points;

    std::vector<types::global_dof_index> local_dof_indices(
      fe.n_dofs_per_cell());

    // Unlike in finite element methods, if we use a collocation boundary
    // element method, then in each assembly loop we only assemble the
    // information that refers to the coupling between one degree of freedom
    // (the degree associated with support point $i$) and the current
    // cell. This is done using a vector of fe.dofs_per_cell elements, which
    // will then be distributed to the matrix in the global row $i$. The
    // following object will hold this information:
    Vector<double> local_H_row_i(fe.n_dofs_per_cell());
    Vector<double> local_G_row_i(fe.n_dofs_per_cell());

    // The index $i$ runs on the collocation points, which are the support
    // points of the $i$th basis function, while $j$ runs on inner integration
    // points.

    // We construct a vector of support points which will be used in the local
    // integrations:
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim - 1, dim>(mapping,
                                                       dof_handler,
                                                       support_points);
    // Here we assign the bc to phi and phi_n iterating through the support points

    // Initialize phi and phi_n vectors
    phi.reinit(dof_handler.n_dofs());
    phi_n.reinit(dof_handler.n_dofs());

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
        const Point<dim> &p = support_points[i];
        if (assign_dirichlet[i])
          phi[i] = dirichlet_function.value(p);
        if (assign_neumann[i])
          phi_n[i] = neumann_function.value(p);
    }

    // After doing so, we can start the integration loop over all cells, where
    // we first initialize the FEValues object and get the values of
    // $\mathbf{\tilde v}$ at the quadrature points (this vector field should
    // be constant, but it doesn't hurt to be more general):

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_v.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        const std::vector<Point<dim>> &q_points = fe_v.get_quadrature_points();
        const std::vector<Tensor<1, dim>> &mesh_outward_normals = fe_v.get_normal_vectors();
        std::vector<Tensor<1, dim>> normals(n_q_points);
        if(exterior_integration_domain)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            normals[q] = -mesh_outward_normals[q];
        } else if (!exterior_integration_domain)
        {
          normals = mesh_outward_normals;
        }

        // We then form the integral over the current cell for all degrees of
        // freedom (note that this includes degrees of freedom not located on
        // the current cell, a deviation from the usual finite element
        // integrals). The integral that we need to perform is singular if one
        // of the local degrees of freedom is the same as the support point
        // $i$. A the beginning of the loop we therefore check whether this is
        // the case, and we store which one is the singular index:
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
          {
            local_H_row_i = 0;
            local_G_row_i = 0;

            bool         is_singular    = false;
            unsigned int singular_index = numbers::invalid_unsigned_int;

            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              if (local_dof_indices[j] == i)
                {
                  singular_index = j;
                  is_singular    = true;
                  break;
                }

            // We then perform the integral. If the index $i$ is not one of
            // the local degrees of freedom, we simply have to add the single
            // layer terms to the right hand side, and the double layer terms
            // to the matrix:
            if (is_singular == false)
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const Tensor<1, dim> R = q_points[q] - support_points[i];

                    for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                     {
                        local_H_row_i(j) +=
                          ((LaplaceKernel::double_layer(R) * normals[q]) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                        local_G_row_i(j) +=
                          ((LaplaceKernel::single_layer(R)) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                      /*
                      if(exterior_integration_domain)
                      {
                        local_H_row_i(j) -=
                          ((LaplaceKernel::double_layer(R) * normals[q]) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                        local_G_row_i(j) -=
                          ((LaplaceKernel::single_layer(R)) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                      } else if (!exterior_integration_domain)
                      {
                        local_H_row_i(j) +=
                          ((LaplaceKernel::double_layer(R) * normals[q]) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                        local_G_row_i(j) +=
                          ((LaplaceKernel::single_layer(R)) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                      }
                      */
                     }
                  }
              }
            else
              {
                // Now we treat the more delicate case. If we are here, this
                // means that the cell that runs on the $j$ index contains
                // support_point[i]. In this case both the single and the
                // double layer potential are singular, and they require
                // special treatment.
                //
                // Whenever the integration is performed with the singularity
                // inside the given cell, then a special quadrature formula is
                // used that allows one to integrate arbitrary functions
                // against a singular weight on the reference cell.
                //
                // The correct quadrature formula is selected by the
                // `get_singular_quadrature()` function, which is explained in
                // detail below.
                Assert(singular_index != numbers::invalid_unsigned_int,
                       ExcInternalError());

                const Quadrature<dim - 1> singular_quadrature =
                  get_singular_quadrature(cell, singular_index);

                FEValues<dim - 1, dim> fe_v_singular(
                  mapping,
                  fe,
                  singular_quadrature,
                  update_jacobians | update_values | update_normal_vectors |
                    update_quadrature_points);

                fe_v_singular.reinit(cell);

                const std::vector<Tensor<1, dim>> &singular_mesh_outward_normals =
                  fe_v_singular.get_normal_vectors();
                const std::vector<Point<dim>> &singular_q_points =
                  fe_v_singular.get_quadrature_points();
                std::vector<Tensor<1, dim>> singular_normals(singular_quadrature.size());
                if(exterior_integration_domain)
                {
                  for (unsigned int q = 0; q < singular_quadrature.size(); ++q)
                    singular_normals[q] = -singular_mesh_outward_normals[q];
                } else if (!exterior_integration_domain)
                {
                  singular_normals = singular_mesh_outward_normals;
                }

                for (unsigned int q = 0; q < singular_quadrature.size(); ++q)
                  {
                    const Tensor<1, dim> R =
                      singular_q_points[q] - support_points[i];

                    for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                      {
                          local_H_row_i(j) +=
                            ((LaplaceKernel::double_layer(R) *
                              singular_normals[q]) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                          local_G_row_i(j) += 
                            ((LaplaceKernel::single_layer(R)) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                        /*
                        if(exterior_integration_domain)
                        {
                          local_H_row_i(j) -=
                            ((LaplaceKernel::double_layer(R) *
                              singular_normals[q]) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                          local_G_row_i(j) -= 
                            ((LaplaceKernel::single_layer(R)) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                        } else if (!exterior_integration_domain)
                        {
                          local_H_row_i(j) +=
                            ((LaplaceKernel::double_layer(R) *
                              singular_normals[q]) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                          local_G_row_i(j) += 
                            ((LaplaceKernel::single_layer(R)) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                        }
                        */
                      }
                  }
              }

            // Finally, we need to add the contributions of the current cell
            // to the global matrix.
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
             {
              H(i, local_dof_indices[j]) += local_H_row_i(j);
              G(i, local_dof_indices[j]) += local_G_row_i(j);
             }
          }
      }

    // The second part of the integral operator is the term
    // $\alpha(\mathbf{x}_i) \phi_j(\mathbf{x}_i)$. Since we use a collocation
    // scheme, $\phi_j(\mathbf{x}_i)=\delta_{ij}$ and the corresponding matrix
    // is a diagonal one with entries equal to $\alpha(\mathbf{x}_i)$.

    // One quick way to compute this diagonal matrix of the solid angles, is
    // to use the Neumann matrix itself. It is enough to multiply the matrix
    // with a vector of elements all equal to -1, to get the diagonal matrix
    // of the alpha angles, or solid angles (see the formula in the
    // introduction for this). The result is then added back onto the system
    // matrix object to yield the final form of the matrix:
    Vector<double> ones(dof_handler.n_dofs());
    ones.add(-1.);

    H.vmult(alpha, ones);
    if(exterior_integration_domain)
      alpha.add(1);

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      H(i, i) += alpha(i);
    }
  }

  // This function constructs the system matrix and right hand side for the mixed boundary conditions case
  template <int dim>
  void BEMProblem<dim>::recombine_matrices()
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
        
    system_matrix.reinit(n_dofs, n_dofs);
    system_rhs.reinit(n_dofs);

    for (unsigned int i = 0; i < n_dofs; ++i) 
    {
      if (assign_dirichlet[i]) 
      {
        // phi[i] is known, phi_n[i] unknown.
        // Eq: G * phi_n = H * phi + phi_inf.
        // Build the i-th row: sistem_matrix(i,j) = G(i,j)
        for (unsigned int j = 0; j < n_dofs; ++j)
          system_matrix(i, j) = G(i, j);
        
        // Rhs: (H*phi)_i + phi_inf
        double sum = 0;
        for (unsigned int j = 0; j < n_dofs; ++j)
          sum += H(i, j) * phi(j);
        system_rhs(i) = sum + phi_at_infinity;
      }
      else if (assign_neumann[i]) 
      {
        // phi_n[i] is known, phi[i] unknown
        // Eq: H * phi = G * phi_n - phi_inf.
        for (unsigned int j = 0; j < n_dofs; ++j)
          system_matrix(i, j) = H(i, j);
        
        double sum = 0;
        for (unsigned int j = 0; j < n_dofs; ++j)
          sum += G(i, j) * phi_n(j);
        system_rhs(i) = sum - phi_at_infinity;
      }
    }
  }


  // @sect4{BEMProblem::solve_system}

  // The next function simply solves the linear system.
  template <int dim>
  void BEMProblem<dim>::solve_system()
  {
    SolverGMRES<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, ls_solution, system_rhs, PreconditionIdentity());
  }
  
  // @sect4{BEMProblem::retrieve_solution}

  // This function assign the solution of the linear system to the corresponding parts of phi and phi_n
  template <int dim>
  void BEMProblem<dim>::retrieve_solution()
  {
      const unsigned int n_dofs = dof_handler.n_dofs();

      for (unsigned int i = 0; i < n_dofs; ++i) 
      {
          if (assign_dirichlet[i]) {
              // If Dirichlet boundary is assigned here, update phi_n
              phi_n[i] = ls_solution[i];
          }
          if (assign_neumann[i]) {
              // If Neumann boundary is assigned here, update phi
              phi[i] = ls_solution[i];
          }
      }

  }

  template <int dim>
  void BEMProblem<dim>::find_mesh_size()
  {
    double mesh_size = 0.0;

    // Loop over all active cells in the boundary (codim=1) triangulation
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        // Assert that each cell is a quadrilateral with 4 vertices
        Assert(cell->n_vertices() == 4, ExcNotImplemented());

        const Point<dim> v0 = cell->vertex(0);
        const Point<dim> v1 = cell->vertex(1);
        const Point<dim> v2 = cell->vertex(2);
        const Point<dim> v3 = cell->vertex(3);

        // Compute length of the two diagonals
        const double diag1 = (v2 - v0).norm(); // diagonal from v0 to v2
        const double diag2 = (v3 - v1).norm(); // diagonal from v1 to v3

        // Cell size is the larger of the two diagonals
        const double cell_size = std::max(diag1, diag2);

        // Update mesh_size if this cell is larger than the current max
        if (cell_size > mesh_size)
          mesh_size = cell_size;
      }

    // store mesh_size in a class variable:
    this->mesh_size = mesh_size;
  }



  // @sect4{BEMProblem::compute_errors}

  // The computation of the errors is exactly the same in all other example
  // programs, and we won't comment too much. Notice how the same methods that
  // are used in the finite element methods can be used here.
  template <int dim>
  void BEMProblem<dim>::compute_errors(const unsigned int cycle)
  {
  
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim - 1, dim>(mapping,
                                                       dof_handler,
                                                       support_points);


    double Linf_error_phi = 0.0;
    double Linf_error_phi_n = 0.0;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      const double exact_val_phi = exact_solution_phi.value(support_points[i]);
      const double diff_phi      = std::fabs(phi[i] - exact_val_phi);
      if (diff_phi > Linf_error_phi)
        Linf_error_phi = diff_phi;

      const double exact_val_phi_n = exact_solution_phi_n.value(support_points[i]);
      const double diff_phi_n      = std::fabs(phi_n[i] - exact_val_phi_n);
      if (diff_phi_n > Linf_error_phi_n)
        Linf_error_phi_n = diff_phi_n;
    }
    
    // WARNING ----------------------------------------------------------------
    // L2 error can be big even if nodal values are the same bacause, since no manifold is set, the mesh is not perfectly aligned with the exact solution and 
    // the quadrature integration evaluates the function phi at different points than the exact_solution_phi.
    Vector<float> difference_per_cell_phi(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      phi,
                                      exact_solution_phi,
                                      difference_per_cell_phi,
                                      QGauss<(dim - 1)>(2 * fe.degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error_phi =
      VectorTools::compute_global_error(tria,
                                        difference_per_cell_phi,
                                        VectorTools::L2_norm);

    Vector<float> difference_per_cell_phi_n(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                  dof_handler,
                                  phi_n,
                                  exact_solution_phi_n,
                                  difference_per_cell_phi_n,
                                  QGauss<(dim - 1)>(2 * fe.degree + 1),
                                  VectorTools::L2_norm);
    const double L2_error_phi_n = 
      VectorTools::compute_global_error(tria,
                                        difference_per_cell_phi_n,
                                        VectorTools::L2_norm);

    // The error in the alpha vector can be computed directly using the
    // Vector::linfty_norm() function, since on each node, the value should be
    // $\frac 12$. All errors are then output and appended to our
    // ConvergenceTable object for later computation of convergence rates:
    Vector<double> difference_per_node(alpha);
    difference_per_node.add(-.5);

    const double       alpha_error    = difference_per_node.linfty_norm();
    const unsigned int n_active_cells = tria.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();

    deallog << "Cycle " << cycle << ':' << std::endl
            << "   Number of active cells:       " << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: " << n_dofs 
            << std::endl
            << "   Mesh size:                    " << mesh_size << std::endl;

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Mesh_size", mesh_size);
    convergence_table.add_value("L2(phi)", L2_error_phi);
    convergence_table.add_value("Linfty(phi)", Linf_error_phi);
    convergence_table.add_value("L2(phi_n)", L2_error_phi_n);
    convergence_table.add_value("Linfty(phi_n)", Linf_error_phi_n);
    convergence_table.add_value("Linfty(alpha)", alpha_error);
  }


  // Singular integration requires a careful selection of the quadrature
  // rules. In particular the deal.II library provides quadrature rules which
  // are tailored for logarithmic singularities (QGaussLog, QGaussLogR), as
  // well as for 1/R singularities (QGaussOneOverR).
  //
  // Singular integration is typically obtained by constructing weighted
  // quadrature formulas with singular weights, so that it is possible to
  // write
  //
  // \f[ \int_K f(x) s(x) dx = \sum_{i=1}^N w_i f(q_i) \f]
  //
  // where $s(x)$ is a given singularity, and the weights and quadrature
  // points $w_i,q_i$ are carefully selected to make the formula above an
  // equality for a certain class of functions $f(x)$.
  //
  // In all the finite element examples we have seen so far, the weight of the
  // quadrature itself (namely, the function $s(x)$), was always constantly
  // equal to 1.  For singular integration, we have two choices: we can use
  // the definition above, factoring out the singularity from the integrand
  // (i.e., integrating $f(x)$ with the special quadrature rule), or we can
  // ask the quadrature rule to "normalize" the weights $w_i$ with $s(q_i)$:
  //
  // \f[ \int_K f(x) s(x) dx = \int_K g(x) dx = \sum_{i=1}^N
  //   \frac{w_i}{s(q_i)} g(q_i) \f]
  //
  // We use this second option, through the @p factor_out_singularity
  // parameter of both QGaussLogR and QGaussOneOverR.
  //
  // These integrals are somewhat delicate, especially in two dimensions, due
  // to the transformation from the real to the reference cell, where the
  // variable of integration is scaled with the determinant of the
  // transformation.
  //
  // In two dimensions this process does not result only in a factor appearing
  // as a constant factor on the entire integral, but also on an additional
  // integral altogether that needs to be evaluated:
  //
  // \f[ \int_0^1 f(x)\ln(x/\alpha) dx = \int_0^1 f(x)\ln(x) dx - \int_0^1
  //  f(x) \ln(\alpha) dx.  \f]
  //
  // This process is taken care of by the constructor of the QGaussLogR class,
  // which adds additional quadrature points and weights to take into
  // consideration also the second part of the integral.
  //
  // A similar reasoning should be done in the three dimensional case, since
  // the singular quadrature is tailored on the inverse of the radius $r$ in
  // the reference cell, while our singular function lives in real space,
  // however in the three dimensional case everything is simpler because the
  // singularity scales linearly with the determinant of the
  // transformation. This allows us to build the singular two dimensional
  // quadrature rules only once and, reuse them over all cells.
  //
  // In the one dimensional singular integration this is not possible, since
  // we need to know the scaling parameter for the quadrature, which is not
  // known a priori. Here, the quadrature rule itself depends also on the size
  // of the current cell. For this reason, it is necessary to create a new
  // quadrature for each singular integration.
  //
  // The different quadrature rules are built inside the
  // get_singular_quadrature, which is specialized for dim=2 and dim=3, and
  // they are retrieved inside the assemble_system function. The index given
  // as an argument is the index of the unit support point where the
  // singularity is located.

  template <>
  Quadrature<2> BEMProblem<3>::get_singular_quadrature(
    const DoFHandler<2, 3>::active_cell_iterator &,
    const unsigned int index) const
  {
    Assert(index < fe.n_dofs_per_cell(),
           ExcIndexRange(0, fe.n_dofs_per_cell(), index));

    static std::vector<QGaussOneOverR<2>> quadratures;
    if (quadratures.empty())
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        quadratures.emplace_back(singular_quadrature_order,
                                 fe.get_unit_support_points()[i],
                                 true);
    return quadratures[index];
  }


  template <>
  Quadrature<1> BEMProblem<2>::get_singular_quadrature(
    const DoFHandler<1, 2>::active_cell_iterator &cell,
    const unsigned int                            index) const
  {
    Assert(index < fe.n_dofs_per_cell(),
           ExcIndexRange(0, fe.n_dofs_per_cell(), index));

    return QGaussLogR<1>(singular_quadrature_order,
                         fe.get_unit_support_points()[index],
                         1. / cell->measure(),
                         true);
  }



  // @sect4{BEMProblem::compute_exterior_solution}

  // We'd like to also know something about the value of the potential $\phi$
  // in the exterior domain: after all our motivation to consider the boundary
  // integral problem was that we wanted to know the velocity in the exterior
  // domain!
  //
  // To this end, let us assume here that the boundary element domain is
  // contained in the box $[-2,2]^{\text{dim}}$, and we extrapolate the actual
  // solution inside this box using the convolution with the fundamental
  // solution. The formula for this is given in the introduction.
  //
  // The reconstruction of the solution in the entire space is done on a
  // continuous finite element grid of dimension dim. These are the usual
  // ones, and we don't comment any further on them. At the end of the
  // function, we output this exterior solution in, again, much the usual way.
  /*
  // Helper functions to know if a given point is inside or outside of the mesh
  // Helper for 3d: ray-triangle intersection (Möller–Trumbore algorithm)
  // Helper for 3d: ray-triangle intersection (Möller–Trumbore algorithm)
  bool ray_intersects_triangle(const Point<3> &ray_origin,
                              const Tensor<1,3> &ray_dir,
                              const Point<3> &v0,
                              const Point<3> &v1,
                              const Point<3> &v2,
                              double &t)
  {
    const double EPS = 1e-12;
    Tensor<1,3> edge1 = v1 - v0;
    Tensor<1,3> edge2 = v2 - v0;
    Tensor<1,3> h = cross_product_3d(ray_dir, edge2);
    double a = edge1 * h;
    if (std::fabs(a) < EPS)
      return false; // Ray is parallel to the triangle.
    double f = 1.0 / a;
    Tensor<1,3> s = ray_origin - v0;
    double u = f * (s * h);
    if (u < 0.0 || u > 1.0)
      return false;
    Tensor<1,3> q = cross_product_3d(s, edge1);
    double v = f * (ray_dir * q);
    if (v < 0.0 || u + v > 1.0)
      return false;
    t = f * (edge2 * q);
    return (t > EPS);
  }


  // Helper for 2d: ray-segment intersection
  // Here, we define the cross product of 2D vectors as the scalar
  // cross(a,b)= a_x*b_y - a_y*b_x.
  double cross_2d(const Tensor<1,2> &a, const Tensor<1,2> &b)
  {
    return a[0]*b[1] - a[1]*b[0];
  }

  bool ray_intersects_segment(const Point<2> &ray_origin,
                              const Tensor<1,2> &ray_dir,
                              const Point<2> &v0,
                              const Point<2> &v1,
                              double &t)
  {
    Tensor<1,2> d = v1 - v0; // segment direction
    double cross_rd = cross_2d(ray_dir, d);
    const double EPS = 1e-12;
    if (std::fabs(cross_rd) < EPS)
      return false; // ray and segment are parallel or degenerate.
    // Using formulas:
    // t = cross( (v0 - ray_origin), d ) / cross(ray_dir, d)
    // s = cross( (v0 - ray_origin), ray_dir ) / cross(ray_dir, d)
    t = cross_2d(v0 - ray_origin, d) / cross_rd;
    double s = cross_2d(v0 - ray_origin, ray_dir) / cross_rd;
    if (t >= 0 && s >= 0 && s <= 1)
      return true;
    return false;
  }

  template <int dim>
  bool ray_intersect_cell(const Point<dim> &ray_origin,
                          const Tensor<1, dim> &ray_dir,
                          typename Triangulation<dim-1, dim>::active_cell_iterator cell,
                          double &t)
  {
    if constexpr (dim == 3) // constexpr if (C++17) means that this i compiled only if dim == 3
    {
      // For a 3D problem, the closed surface is 2D.
      // We assume that each cell is a quadrilateral.
      // Split the quad into two triangles: (v0,v1,v2) and (v0,v2,v3).
      Assert(cell->n_vertices() == 4, ExcInternalError());
      const Point<3> v0 = cell->vertex(0);
      const Point<3> v1 = cell->vertex(1);
      const Point<3> v2 = cell->vertex(2);
      const Point<3> v3 = cell->vertex(3);
      double t1, t2;
      bool hit1 = ray_intersects_triangle(ray_origin, ray_dir, v0, v1, v2, t1);
      bool hit2 = ray_intersects_triangle(ray_origin, ray_dir, v0, v2, v3, t2);
      if (hit1 && hit2)
        t = std::min(t1, t2);
      else if (hit1)
        t = t1;
      else if (hit2)
        t = t2;
      return hit1 || hit2;
    }
    else if constexpr (dim == 2) // constexpr if (C++17) means that this i compiled only if dim == 2
    {
      // For a 2D problem, the closed curve is 1D.
      // We assume that each cell is a segment with 2 vertices.
      Assert(cell->n_vertices() == 2, ExcInternalError());
      // In the 2D branch, our cell vertices are Point<2>
      const Point<2> v0 = cell->vertex(0);
      const Point<2> v1 = cell->vertex(1);
      return ray_intersects_segment(ray_origin, ray_dir, v0, v1, t);
    }
    else
    {
      Assert(false, ExcNotImplemented());
      return false;
    }
  }



  template <int dim>
  bool BEMProblem<dim>::is_point_exterior(const Point<dim> &p)
  {
    // Compute the center of the domain.
    // Here we assume the center is the origin; adjust if necessary.
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.0;

    // Choose the ray direction as the vector from the center to p.
    Tensor<1, dim> ray_dir = p - center;
    double norm = ray_dir.norm();
    if (norm < 1e-12)
    {
      // If p coincides with the center, choose an arbitrary ray direction.
      ray_dir[0] = 1.0;
      for (unsigned int d = 1; d < dim; ++d)
        ray_dir[d] = 0.0;
    }
    else
      ray_dir /= norm; // Normalize the ray direction.

    unsigned int intersection_count = 0;
    double t; // Intersection parameter (unused here)

    // Loop over all active cells in the closed surface (or curve) mesh.
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      if (ray_intersect_cell<dim>(p, ray_dir, cell, t))
        ++intersection_count;
    }

    // According to the ray-casting method, if the number of intersections is odd,
    // the point is inside; if even, it is exterior.
    return (intersection_count % 2 == 0);
  }
  */

    
  template <int dim>
  void BEMProblem<dim>::compute_exterior_solution()
  {
    // Create a triangulation for the external plate (surface mesh)
    Triangulation<dim-1, dim> external_tria;
    GridGenerator::hyper_cube(external_tria, -2, 2);

    // Use a finite element of dimension (dim-1) (embedded in dim) for the plate.
    // For example, for dim=3, this creates FE_Q<2,3>.
    const FE_Q<dim-1, dim> external_fe(1);
    DoFHandler<dim-1, dim> external_dh(external_tria);
    Vector<double> external_phi;
    double temp_quadrature_value = 0;

    external_tria.refine_global(external_refinement);
    external_dh.distribute_dofs(external_fe);
    external_phi.reinit(external_dh.n_dofs());

    // FEValues for the boundary integral (using the same mapping and FE as for the original BEM)
    FEValues<dim-1, dim> fe_v(mapping,
                              fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points = fe_v.n_quadrature_points;
    std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());

    std::vector<double> local_phi(n_q_points);
    std::vector<double> local_phi_n(n_q_points);

    // Map dofs to support points on the external plate
    std::vector<Point<dim>> external_support_points(external_dh.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>(StaticMappingQ1<dim-1, dim>::mapping,
                                                    external_dh,
                                                    external_support_points);

    // Loop over the cells in the original boundary mesh
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_v.reinit(cell);
      const std::vector<Point<dim>> &q_points = fe_v.get_quadrature_points();
      const std::vector<Tensor<1, dim>> &mesh_outward_normals = fe_v.get_normal_vectors();
      std::vector<Tensor<1, dim>> normals(n_q_points);
      if(exterior_integration_domain)
        for (unsigned int q = 0; q < n_q_points; ++q)
          normals[q] = -mesh_outward_normals[q];

      cell->get_dof_indices(dofs);
      fe_v.get_function_values(phi, local_phi);
      fe_v.get_function_values(phi_n, local_phi_n);

      // For each support point of the external mesh, accumulate the convolution integral
      for (unsigned int i = 0; i < external_dh.n_dofs(); ++i)
      {
        // bool is_exterior = is_point_exterior(external_support_points[i]);
        
        temp_quadrature_value = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, dim> R = q_points[q] - external_support_points[i];

          temp_quadrature_value += ( (LaplaceKernel::single_layer(R) * local_phi_n[q] +
                  LaplaceKernel::double_layer(R) * normals[q] * local_phi[q])
                  * fe_v.JxW(q) );
        }

        if(exterior_integration_domain)
        {
          external_phi(i) += temp_quadrature_value + phi_at_infinity;
        } else if (!exterior_integration_domain)
        {
          external_phi(i) += temp_quadrature_value;
        }
        /*
        if(is_exterior)
        {
          if(exterior_integration_domain)
          {
            external_phi(i) = temp_quadrature_value + phi_at_infinity;
          } else if (!exterior_integration_domain)
          {
            // Assign NaN values to interior points when the solution is for an external domain problem
            external_phi(i) = 10000;
          }
        } else if (!is_exterior)
        {
          if(exterior_integration_domain)
          {
            // Assign NaN values to exterior points when the solution is for an internal domain problem
            external_phi(i) = 10000;
          } else if (!exterior_integration_domain)
          {
            external_phi(i) = temp_quadrature_value;
          }
        }
        */
        
      }
    }

    // Use a DataOut object that is compatible with a (dim-1)-dimensional DoFHandler
    DataOut<dim-1, dim> data_out;
    data_out.attach_dof_handler(external_dh);
    data_out.add_data_vector(external_phi, "external_phi");
    data_out.build_patches();

    const std::string filename = std::to_string(dim) + "d_external.vtk";
    std::ofstream file(filename);
    data_out.write_vtk(file);
  }

  

  // @sect4{BEMProblem::output_results}

  // Outputting the results of our computations is a rather mechanical
  // tasks. All the components of this function have been discussed before.
  template <int dim>
  void BEMProblem<dim>::output_results(const unsigned int cycle)
  {
    DataOut<dim - 1, dim> dataout;

    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim - 1, dim>(mapping,
                                                       dof_handler,
                                                       support_points);
    Vector<double> phi_exact(dof_handler.n_dofs());
    Vector<double> phi_n_exact(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      phi_exact[i]   = exact_solution_phi.value(support_points[i]);
      phi_n_exact[i] = exact_solution_phi_n.value(support_points[i]);
    }

    dataout.attach_dof_handler(dof_handler);
    dataout.add_data_vector(phi, "phi", DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(phi_n, "phi_n", DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(alpha,
                            "alpha",
                            DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(boundary_type, "boundary_type", DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(phi_exact, "phi_exact", DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(phi_n_exact, "phi_n_exact", DataOut<dim - 1, dim>::type_dof_data);
    dataout.build_patches(mapping,
                          mapping.get_degree(),
                          DataOut<dim - 1, dim>::curved_inner_cells);

    const std::string filename = std::to_string(dim) + "d_boundary_solution_" +
                                 std::to_string(cycle) + ".vtk";
    std::ofstream file(filename);

    dataout.write_vtk(file);

    if (cycle == mesh_filenames.size() - 1)
      {
        convergence_table.set_precision("L2(phi)", 3);
        convergence_table.set_precision("Linfty(phi)", 3);
        convergence_table.set_precision("L2(phi_n)", 3);
        convergence_table.set_precision("Linfty(phi_n)", 3);
        convergence_table.set_precision("Linfty(alpha)", 3);

        convergence_table.set_scientific("L2(phi)", true);
        convergence_table.set_scientific("Linfty(phi)", true);
        convergence_table.set_scientific("L2(phi_n)", true);
        convergence_table.set_scientific("Linfty(phi_n)", true);
        convergence_table.set_scientific("Linfty(alpha)", true);

        convergence_table.evaluate_convergence_rates(
          "L2(phi)", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Linfty(phi)", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "L2(phi_n)", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Linfty(phi_n)", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "Linfty(alpha)", ConvergenceTable::reduction_rate_log2);
        deallog << std::endl;

        convergence_table.write_text(std::cout);
        std::ofstream out("convergence_table.txt");
        convergence_table.write_text(out);
      }
  }

  // @sect4{BEMProblem::release_memory}
  template <int dim>
  void BEMProblem<dim>::release_memory()
  {
    // Sgancia tutti i DoF dal dof_handler
    dof_handler.clear();

    // Elimina completamente la triangolazione
    tria.clear();

    // Reinizializza le matrici e i vettori con dimensione 0
    system_matrix.reinit(0, 0);
    H.reinit(0, 0);
    G.reinit(0, 0);
    system_rhs.reinit(0);

    phi.reinit(0);
    phi_n.reinit(0);
    alpha.reinit(0);
    ls_solution.reinit(0);
    
    boundary_type.reinit(0);
    assign_neumann.resize(0);
    assign_dirichlet.resize(0);

    mesh_size = 0;
  }


  // @sect4{BEMProblem::run}

  // This is the main function. It should be self explanatory in its
  // briefness:
  template <int dim>
  void BEMProblem<dim>::run()
  {
    read_parameters("parameters_cube.prm");

    if (run_in_this_dimension == false)
      {
        deallog << "Run in dimension " << dim
                << " explicitly disabled in parameter file. " << std::endl;
        return;
      }

    for (unsigned int cycle = 0; cycle < mesh_filenames.size(); ++cycle)
      {
        deallog << "Cycle started, using mesh file: " << mesh_filenames[cycle] << std::endl;
        read_mesh(mesh_filenames[cycle]);
        refine_and_resize();
        set_boundary_flags();
        assemble_system();
        recombine_matrices();
        solve_system();
        retrieve_solution();
        find_mesh_size();
        compute_errors(cycle);
        output_results(cycle);
        release_memory();
      }

    if (extend_solution == true)
      compute_exterior_solution();
  }
} // namespace Step34


// @sect3{The main() function}

// This is the main function of this program. It is exactly like all previous
// tutorial programs:
int main()
{
  try
    {
      using namespace Step34;

      const unsigned int degree         = 1;
      const unsigned int mapping_degree = 1;

      deallog.depth_console(3);
      //BEMProblem<2> laplace_problem_2d(degree, mapping_degree);
      //laplace_problem_2d.run();

      BEMProblem<3> laplace_problem_3d(degree, mapping_degree);
      laplace_problem_3d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
