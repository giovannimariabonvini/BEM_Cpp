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
#include <map>
#include <functional>


// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace Step34
{
  using namespace dealii;

  namespace LaplaceKernel // ONLY FOR SOLID ANGLE ALPHA COMPUTATION
  {
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
  }


  template <int dim>
  class BEMProblem
  {
  public:
    BEMProblem(const unsigned int fe_degree, const unsigned int mapping_degree = 1, const std::string &parameters_filename = "parameters.prm");
    void run();

  private:
    void read_parameters(const std::string &filename);
    void init_structures();
    void read_mesh(const std::string &mesh_file);
    void assemble_system();
    void solve_system();
    void retrieve_solution();
    void set_boundary_flags();
    void recombine_matrices();
    void release_memory();
    void find_mesh_size();
    void compute_errors(const unsigned int cycle);
    void compute_exterior_solution();
    void output_results(const unsigned int cycle);

    Quadrature<dim - 1> get_singular_quadrature(
      const typename DoFHandler<dim - 1, dim>::active_cell_iterator &cell,
      const unsigned int index) const;

    Triangulation<dim - 1, dim> tria;
    const FE_Q<dim - 1, dim>    fe;
    DoFHandler<dim - 1, dim>    dof_handler;
    MappingQ<dim - 1, dim>      mapping;
    std::vector<std::string> mesh_filenames;

    FullMatrix<double> system_matrix;
    Vector<double>     system_rhs;

    FullMatrix<double> H;
    FullMatrix<double> G;

    FullMatrix<double> H_Laplace; // Only for alpha calculation

    Vector<double> phi;
    Vector<double> phi_n;
    Vector<double> alpha;
    Vector<double> ls_solution;

    std::vector<bool> assign_dirichlet;
    std::vector<bool> assign_neumann;
    std::vector<bool> assign_robin;   
    Vector<double>    beta;      
    Vector<double>    g;             

    Vector<double> boundary_type;

    bool exterior_integration_domain;
    double phi_at_infinity;

    ConvergenceTable convergence_table;

    Functions::ParsedFunction<dim> exact_solution_phi;
    Functions::ParsedFunction<dim> exact_solution_phi_n;

    Functions::ParsedFunction<dim> neumann_function;
    Functions::ParsedFunction<dim> dirichlet_function;
    Functions::ParsedFunction<dim> robin_beta_function;
    Functions::ParsedFunction<dim> robin_g_function;

    Functions::ParsedFunction<dim> region_function; 

    Functions::ParsedFunction<dim> single_layer_function; // G
    Functions::ParsedFunction<dim> double_layer_function; // ∇G

    // Helpers for the kernel functions
    double single_layer(const Tensor<1,dim> &R) const;
    Tensor<1,dim> double_layer(const Tensor<1,dim> &R) const;

    // Name of the singular quadrature rule the user picks
    std::string singular_quadrature_type;

    unsigned int                         singular_quadrature_order;
    std::shared_ptr<Quadrature<dim - 1>> quadrature;

    SolverControl solver_control;

    double mesh_size;

    std::string parameters_filename;
  };


  template <int dim>
  BEMProblem<dim>::BEMProblem(const unsigned int fe_degree, const unsigned int mapping_degree, const std::string &parameters_filename)
    : fe(fe_degree)  
    , dof_handler(tria)
    , mapping(mapping_degree)
    , double_layer_function(dim)
    , singular_quadrature_order(5)
    , parameters_filename(parameters_filename)
  {}

  template <int dim>
  double BEMProblem<dim>::single_layer(const Tensor<1,dim> &R) const
  {
    Point<dim> p;
    for (unsigned d = 0; d < dim; ++d)
      p[d] = R[d];
    
    return single_layer_function.value(p);
  }

  template <int dim>
  Tensor<1,dim> BEMProblem<dim>::double_layer(const Tensor<1,dim> &R) const
  {
    Point<dim> p;
    for (unsigned d = 0; d < dim; ++d)
      p[d] = R[d];

    Tensor<1,dim> result;
    for (unsigned c = 0; c < dim; ++c)
      result[c] = double_layer_function.value(p, c);

    return result;
  }


  // The read_parameters() function is the one that reads the parameter file
  template <int dim>
  void BEMProblem<dim>::read_parameters(const std::string &filename)
  {
    Assert(dim == 3, ExcInternalError());
    deallog << "\nParsing parameter file " << filename
            << "\nfor a 3-dimensional simulation.\n";

    ParameterHandler prm;

    // 1) Global options
    prm.declare_entry("Mesh filenames",         "cubed_sphere.msh", Patterns::Anything());
    prm.declare_entry("Exterior domain",           "false", Patterns::Bool());
    prm.declare_entry("Infinity Dirichlet value",  "0.0",   Patterns::Double());

    prm.enter_subsection("Single layer function");
      Functions::ParsedFunction<dim>::declare_parameters(prm);       // 1 component (scalar)
    prm.leave_subsection();

    prm.enter_subsection("Double layer function");
      Functions::ParsedFunction<dim>::declare_parameters(prm, dim);     // 'dim' components (vector)
    prm.leave_subsection();

    // 2) Exact solution (optional)
    prm.enter_subsection("Exact solution phi 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi_n 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    // 3) Quadrature rules
    prm.enter_subsection("Quadrature rules");
      prm.declare_entry("Quadrature type",            "gauss",
        Patterns::Selection(QuadratureSelector<2>::get_quadrature_names()));
      prm.declare_entry("Quadrature order",           "4",    Patterns::Integer());
      prm.declare_entry("Singular quadrature type",  "one_over_r",
        Patterns::Anything());
      prm.declare_entry("Singular quadrature order",  "5",    Patterns::Integer());
    prm.leave_subsection();

    // 4) Solver parameters
    prm.enter_subsection("Solver");
      SolverControl::declare_parameters(prm);
    prm.leave_subsection();

    // 5) Boundary regions (0=Neumann,1=Dirichlet,2=Robin)
    prm.enter_subsection("Boundary regions");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    // 6) Boundary data
    prm.enter_subsection("Dirichlet function 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Neumann function 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Robin beta function 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Robin g function 3d");
      Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.leave_subsection();

    // Read the file
    prm.parse_input(filename);

    // Store global values
    mesh_filenames              = Utilities::split_string_list(prm.get("Mesh filenames"), ' ');
    exterior_integration_domain = prm.get_bool("Exterior domain");
    phi_at_infinity             = prm.get_double("Infinity Dirichlet value");

    // Read single‐layer kernel parameters 
    prm.enter_subsection("Single layer function");
      single_layer_function.parse_parameters(prm);
    prm.leave_subsection();

    // Read double‐layer kernel parameters 
    prm.enter_subsection("Double layer function");
      double_layer_function.parse_parameters(prm);
    prm.leave_subsection();

    // Read quadrature rules
    prm.enter_subsection("Quadrature rules");
      quadrature = std::make_shared<Quadrature<2>>(
        QuadratureSelector<2>(
          prm.get("Quadrature type"),
          prm.get_integer("Quadrature order")));
      singular_quadrature_type  = prm.get("Singular quadrature type");
      singular_quadrature_order = prm.get_integer("Singular quadrature order");
    prm.leave_subsection();

    // Read solver parameters
    prm.enter_subsection("Solver");
      solver_control.parse_parameters(prm);
    prm.leave_subsection();

    // Parse exact solutions
    prm.enter_subsection("Exact solution phi 3d");
      exact_solution_phi.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Exact solution phi_n 3d");
      exact_solution_phi_n.parse_parameters(prm);
    prm.leave_subsection();

    // Parse boundary flag function
    prm.enter_subsection("Boundary regions");
      region_function.parse_parameters(prm);
    prm.leave_subsection();

    // Parse boundary data functions
    prm.enter_subsection("Dirichlet function 3d");
      dirichlet_function.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Neumann function 3d");
      neumann_function.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Robin beta function 3d");
      robin_beta_function.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("Robin g function 3d");
      robin_g_function.parse_parameters(prm);
    prm.leave_subsection();
  }


  template <int dim>
  void BEMProblem<dim>::init_structures()
  {
    tria.refine_global(0); // Leave to 0 (no refinement) since we don't have a manifold description of the geometry, just a mesh
    dof_handler.distribute_dofs(fe);

    const unsigned int n_dofs = dof_handler.n_dofs();

    system_matrix.reinit(n_dofs, n_dofs);
    H.reinit(n_dofs, n_dofs);
    G.reinit(n_dofs, n_dofs);

    H_Laplace.reinit(n_dofs, n_dofs);

    system_rhs.reinit(n_dofs);
    phi.reinit(n_dofs);
    phi_n.reinit(n_dofs);
    alpha.reinit(n_dofs);
    ls_solution.reinit(n_dofs);

    beta.reinit(n_dofs);
    g.reinit(n_dofs);

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
  }


  template <int dim>
  void BEMProblem<dim>::set_boundary_flags()
  {
    const unsigned int n_dofs = dof_handler.n_dofs();
    assign_dirichlet.resize(n_dofs, false);
    assign_neumann.resize(n_dofs, false);
    assign_robin.resize(n_dofs, false);

    std::vector<Point<dim>> support_points(n_dofs);
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    for (unsigned int i=0; i<n_dofs; ++i)
    {
      const Point<dim> p = support_points[i];
      const double flag = region_function.value(p);

      if (std::abs(flag) < 1e-12)                 // ---- Neumann
      {
          assign_neumann[i] = true;
          phi_n[i]          = neumann_function.value(p);
      }
      else if (std::abs(flag-1.0) < 1e-12)        // ---- Dirichlet
      {
          assign_dirichlet[i] = true;
          phi[i]              = dirichlet_function.value(p);
      }
      else if (std::abs(flag-2.0) < 1e-12)        // ---- Robin
      {
          assign_robin[i] = true;
          beta[i]         = robin_beta_function.value(p);
          g[i]            = robin_g_function.value(p);
      }
      else
          AssertThrow(false, ExcMessage("Region function must be 0,1,2"));
    }
  
    // Count the true values in each vector
    unsigned int n_neumann = 0;
    unsigned int n_dirichlet = 0;
    unsigned int n_robin = 0;
    boundary_type.reinit(n_dofs);
    boundary_type = 2;
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
      } else if (assign_robin[i])
      {
        boundary_type[i] = 2;
        ++n_robin;
      }
      else
      {
        std::cerr << "Error: Boundary type not assigned to support point " << i << std::endl;
        return;
      }
    }

    deallog << "Boundary flags assigned: " 
              << n_neumann << " Neumann, " 
              << n_dirichlet << " Dirichlet and " 
              << n_robin << " Robin nodes " << std::endl;

  }


  // The following is the core function of this program, assembling the matrix
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

    Vector<double> local_H_Laplace_row_i(fe.n_dofs_per_cell());

    // The index i runs on the collocation points, which are the support
    // points of the i-th basis function, while j runs on inner integration
    // points.

    // We construct a vector of support points which will be used in the local
    // integrations:
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points<dim - 1, dim>(mapping,
                                                       dof_handler,
                                                       support_points);
  

    const unsigned int n_cells   = tria.n_active_cells();
    unsigned int       cell_idx  = 0;
    unsigned int       next_pct  = 10;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        ++cell_idx;
        // compute the current % for log output
        const unsigned int pct = (cell_idx * 100) / n_cells;
        if (pct >= next_pct)
        {
            std::cout << "Progress: " << next_pct << "%\n";
            next_pct += 10;
        }
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
        // freedom. The integral that we need to perform is singular if one
        // of the local degrees of freedom is the same as the support point
        // $i$. At the beginning of the loop we therefore check whether this is
        // the case, and we store which one is the singular index:
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
          {
            local_H_row_i = 0;
            local_G_row_i = 0;

            local_H_Laplace_row_i = 0;

            bool         is_singular    = false;
            unsigned int singular_index = numbers::invalid_unsigned_int;

            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              if (local_dof_indices[j] == i)
                {
                  singular_index = j;
                  is_singular    = true;
                  break;
                }

            if (is_singular == false)
              {
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const Tensor<1, dim> R = q_points[q] - support_points[i]; // y - x (y=integration var, x=collocation point)

                    for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                     {
                        local_H_row_i(j) +=
                          ((double_layer(R) * normals[q]) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                        local_G_row_i(j) +=
                          ((single_layer(R)) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                        
                        local_H_Laplace_row_i(j) +=
                          ((LaplaceKernel::double_layer(R) * normals[q]) *
                          fe_v.shape_value(j, q) * fe_v.JxW(q));
                          
                     }
                  }
              }
            else
              {
                // Now we treat the more delicate case. If we are here, this
                // means that the cell that runs on the j index contains
                // support_point[i]. In this case the single layer potential is singular and it require
                // special treatment.

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
                          // local_H_row_i(j) += 0; since (y-x) is perpendicular to the normal vector

                          local_G_row_i(j) += 
                            ((single_layer(R)) *
                            fe_v_singular.shape_value(j, q) *
                            fe_v_singular.JxW(q));
                           
                          //local_H_Laplace_row_i(j) += 0; since (y-x) is perpendicular to the normal vector
                            
                      }
                  }
              }

            // Finally, we need to add the contributions of the current cell
            // to the global matrix.
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
            {
              H(i, local_dof_indices[j]) += local_H_row_i(j);
              G(i, local_dof_indices[j]) += local_G_row_i(j);

              H_Laplace(i, local_dof_indices[j]) += local_H_Laplace_row_i(j);
            }
          }
      }

    Vector<double> ones(dof_handler.n_dofs());
    ones.add(-1.); 
    
    H_Laplace.vmult(alpha, ones);

    // H.vmult(alpha, ones);

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
    const unsigned int N = dof_handler.n_dofs();

    system_matrix.reinit(N, N);
    system_rhs.reinit(N);

    for (unsigned int i = 0; i < N; ++i)
    {
      // DIRICHLET ROW
      if (assign_dirichlet[i])
      {
        for (unsigned int j = 0; j < N; ++j)
          system_matrix(i, j) = G(i, j);

        double rhs = 0.0;

        for (unsigned int j = 0; j < N; ++j)
          if (assign_dirichlet[j])                    // φ_j known
            rhs += H(i, j) * phi(j);

        for (unsigned int j = 0; j < N; ++j)
          if (assign_robin[j])                        // q = g - βφ,  g known
            rhs -= G(i, j) * g(j);

        system_rhs(i) = rhs + phi_at_infinity;
      }
      // NEUMANN ROW
      else if (assign_neumann[i])
      {
        for (unsigned int j = 0; j < N; ++j)
          system_matrix(i, j) = H(i, j);

        double rhs = 0.0;

        for (unsigned int j = 0; j < N; ++j)
          if (assign_neumann[j])                      // q_j known
            rhs += G(i, j) * phi_n(j);

        for (unsigned int j = 0; j < N; ++j)
          if (assign_robin[j])                        // β_j * φ_j unknown ⇒ g known
            rhs += beta(j) * H(i, j) * g(j);

        system_rhs(i) = rhs - phi_at_infinity;
      }
      // ROBIN ROW
      else if (assign_robin[i])
      {
        for (unsigned int j = 0; j < N; ++j)
          system_matrix(i,j) = H(i,j) - beta(i) * G(i,j);

        double rhs = 0.0;
        for (unsigned int j = 0; j < N; ++j)
          if (assign_robin[j])      // g is known on all robin nodes
            rhs -= G(i,j) * g(j);

        system_rhs(i) = rhs;
      }
      else
        AssertThrow(false, ExcMessage("Boundary flag not set"));
    }
  }

  // In BEM methods, the matrix that is generated is dense. Depending on the
  // size of the problem, the final system might be solved by direct LU
  // decomposition, or by iterative methods. In this example we use an
  // unpreconditioned GMRES method. Building a preconditioner for BEM method
  // is non trivial, and we don't treat this subject here.
  template <int dim>
  void BEMProblem<dim>::solve_system()
  {
    SolverGMRES<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, ls_solution, system_rhs, PreconditionIdentity());
  }
  

  // This function assign the solution of the linear system to the corresponding parts of phi and phi_n
  template <int dim>
  void BEMProblem<dim>::retrieve_solution()
  {
      const unsigned int n_dofs = dof_handler.n_dofs();

      for (unsigned int i=0, idx=0; i<n_dofs; ++i)
      {
          if (assign_dirichlet[i])      
            phi_n[i] = ls_solution[idx++];
          else if (assign_neumann[i])     
            phi[i] = ls_solution[idx++];
          else      /* Robin */
          {
              phi[i]   = ls_solution[idx++];
              phi_n[i] = g[i] - beta[i]*phi[i];   // q = g - βφ
          }
      }

  }

  template <int dim>
  void BEMProblem<dim>::find_mesh_size()
  {
    double accumulated_size = 0.0;
    unsigned int n_cells     = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      Assert(cell->n_vertices() == 4, ExcNotImplemented());

      const double diag1 = (cell->vertex(2) - cell->vertex(0)).norm();
      const double diag2 = (cell->vertex(3) - cell->vertex(1)).norm();

      accumulated_size += std::max(diag1, diag2);
      ++n_cells;
    }

    // Guard against an empty triangulation
    Assert(n_cells > 0, ExcMessage("No active cells in boundary mesh."));

    this->mesh_size = accumulated_size / static_cast<double>(n_cells);
  }


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

    // Compute the Linf norm of the solution i.e. the maximum value of the absolute value of the solution
    double Linf_norm_phi = 0.0;
    double Linf_norm_phi_n = 0.0;
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    {
      if (std::fabs(exact_solution_phi.value(support_points[i])) > Linf_norm_phi)
        Linf_norm_phi = std::fabs(exact_solution_phi.value(support_points[i]));
      if (std::fabs(exact_solution_phi_n.value(support_points[i])) > Linf_norm_phi_n)
        Linf_norm_phi_n = std::fabs(exact_solution_phi_n.value(support_points[i]));
    }

    // Normalize the Linf error with the Linf norm of the solution
    Linf_error_phi /= Linf_norm_phi;
    Linf_error_phi_n /= Linf_norm_phi_n;

    
    // WARNING ----------------------------------------------------------------
    // L2 error can be big even if nodal values are the same bacause, since no manifold is set, the mesh is not perfectly aligned with the exact solution and 
    // the quadrature integration evaluates the function phi at different points than the exact_solution_phi.

    // Compute the local L2 error on each cell
    Vector<float> difference_per_cell_phi(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      phi,
                                      exact_solution_phi,
                                      difference_per_cell_phi,
                                      QGauss<(dim - 1)>(2 * fe.degree + 1),
                                      VectorTools::L2_norm);

    Vector<float> difference_per_cell_phi_n(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                  dof_handler,
                                  phi_n,
                                  exact_solution_phi_n,
                                  difference_per_cell_phi_n,
                                  QGauss<(dim - 1)>(2 * fe.degree + 1),
                                  VectorTools::L2_norm);

    // Sum up the local contributions to obtain the global error
    double L2_error_phi = VectorTools::compute_global_error(tria,
                                        difference_per_cell_phi,
                                        VectorTools::L2_norm);
    
    double L2_error_phi_n = VectorTools::compute_global_error(tria,
                                        difference_per_cell_phi_n,
                                        VectorTools::L2_norm);

    // Compute the L2 norm of the exact solution
    Vector<double> zero_function;
    zero_function.reinit(dof_handler.n_dofs());

    Vector<float> L2_norm_phi_by_cell(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      zero_function,
                                      exact_solution_phi,
                                      L2_norm_phi_by_cell,
                                      QGauss<(dim - 1)>(2 * fe.degree + 1),
                                      VectorTools::L2_norm);
        
    Vector<float> L2_norm_phi_n_by_cell(tria.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      zero_function,
                                      exact_solution_phi_n,
                                      L2_norm_phi_n_by_cell,
                                      QGauss<(dim - 1)>(2 * fe.degree + 1),
                                      VectorTools::L2_norm);
    
    const double L2_norm_phi = VectorTools::compute_global_error(tria,
                                                                 L2_norm_phi_by_cell,
                                                                 VectorTools::L2_norm);
    
    const double L2_norm_phi_n = VectorTools::compute_global_error(tria,
                                                                 L2_norm_phi_n_by_cell,
                                                                 VectorTools::L2_norm);

    // Normalize the global error with the L2 norm of the exact solution
    L2_error_phi /= L2_norm_phi;
    L2_error_phi_n /= L2_norm_phi_n;
                                 

    // The error in the alpha vector can be computed directly using the
    // Vector::linfty_norm() function, since on each node, the value should be
    // 0.5. All errors are then output and appended to our
    // ConvergenceTable object for later computation of convergence rates:
    Vector<double> difference_per_node(alpha);
    difference_per_node.add(-.5);

    const double       alpha_error    = difference_per_node.linfty_norm();
    const unsigned int n_active_cells = tria.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();

    deallog << "   Number of active cells:       " << n_active_cells
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


  template <>
  Quadrature<2>
  BEMProblem<3>::get_singular_quadrature(
    const typename DoFHandler<2,3>::active_cell_iterator &cell,
    unsigned int                                              index) const
  {
    // 1) Define builder type and map (static so it’s initialized once)
    using SQBuilder =
      std::function<Quadrature<2>(unsigned int,     // order
                                const Point<2>&,    // support point
                                double,             // scale 
                                bool)>;             // symmetric?

    static const std::map<std::string, SQBuilder> builder_map = {
      { "one_over_r", [](auto order, const auto &pt, auto, auto sym) {
          return QGaussOneOverR<2>(order, pt, sym);
        }
      },
      // Telles’ transformation rule (maps base Gauss to singular point)
      { "telles", [](auto order, const auto &pt, auto, auto) {
          return QTelles<2>(order, pt);
        }
      },
      // Duffy transformation (collapses square to triangle)
      { "duffy", [](auto order, const auto &, auto, auto) {
          return QDuffy(order, 1.0);
        }
      },
      // Polar‐coordinate rule on triangle
      { "triangle_polar", [](auto order, const auto &, auto, auto) {
          return QTrianglePolar(order);
        }
      }
      // add here more entries here if you implement other 2D singular rules
      // see https://dealii.org/current/doxygen/deal.II/quadrature__lib_8h.html for full list
    };


    // 2) Prepare scale and support points
    const double scale           = 1.0 / cell->measure();
    const auto &support_points   = fe.get_unit_support_points();

    // 3) Look up the user’s choice
    auto it = builder_map.find(singular_quadrature_type);
    AssertThrow(it != builder_map.end(),
                ExcMessage("Unknown singular quadrature type: "
                          + singular_quadrature_type));
    const auto &builder = it->second;

    // 4) Build & cache one Quadrature per support point
    static std::vector<Quadrature<2>> quadratures;
    if (quadratures.empty())
    {
      quadratures.reserve(fe.n_dofs_per_cell());
      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        quadratures.push_back(
          builder(singular_quadrature_order, // order
                  support_points[i],         // point
                  scale,                     // scale
                  true));                    // symmetric?
    }

    // 5) Return the requested quadrature
    return quadratures[index];
  }


  // This function computes the solution on the external plate. Since its not the focus of this work
  // is now commented out. It is left here for future reference and extensions of the code.

  /* 
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
  */
  

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
    dataout.add_data_vector(beta, "beta", DataOut<dim - 1, dim>::type_dof_data);
    dataout.add_data_vector(g, "g", DataOut<dim - 1, dim>::type_dof_data); 
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
          "L2(phi)", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "Linfty(phi)", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "L2(phi_n)", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "Linfty(phi_n)", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "Linfty(alpha)", ConvergenceTable::reduction_rate);
        deallog << std::endl;

        convergence_table.write_text(std::cout);
        std::ofstream out("convergence_table.txt");
        convergence_table.write_text(out);
      }
  }

  template <int dim>
  void BEMProblem<dim>::release_memory()
  {
    dof_handler.clear();

    tria.clear();

    system_matrix.reinit(0, 0);
    H.reinit(0, 0);
    G.reinit(0, 0);
    system_rhs.reinit(0);

    H_Laplace.reinit(0, 0);

    phi.reinit(0);
    phi_n.reinit(0);
    alpha.reinit(0);
    ls_solution.reinit(0);
    
    boundary_type.reinit(0);
    assign_neumann.resize(0);
    assign_dirichlet.resize(0);

    mesh_size = 0;
  }

  // This is the run function. It should be self explanatory in its
  // briefness:
  template <int dim>
  void BEMProblem<dim>::run()
  {
    read_parameters(parameters_filename);

    for (unsigned int cycle = 0; cycle < mesh_filenames.size(); ++cycle)
      {
        deallog << "Using mesh file: " << mesh_filenames[cycle] << std::endl;
        read_mesh(mesh_filenames[cycle]);
        init_structures();
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

    //if (extend_solution == true)
      // compute_exterior_solution();
  }
} // namespace Step34


int main()
{
  try
    {
      using namespace Step34;

      const std::string parameters_filename = "parameters.prm";

      const unsigned int mapping_degree = 1;

      ParameterHandler prm;
      prm.declare_entry("Polynomial degree", "1", Patterns::Integer());
      prm.parse_input(parameters_filename, "", "", /* skip_undefined= */ true);
      const unsigned int user_fe_degree = prm.get_integer("Polynomial degree");

      deallog.depth_console(3);

      BEMProblem<3> laplace_problem_3d(user_fe_degree, mapping_degree, parameters_filename);
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

