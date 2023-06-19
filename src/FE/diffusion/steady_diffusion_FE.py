#
# Conv-React Test
# ===================
#

import dolfinx
from dolfinx.fem import (
    Expression, Function, FunctionSpace,
    assemble_scalar, form
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx import log
from ufl import(
     TestFunction, TrialFunction, FiniteElement, VectorElement,
     SpatialCoordinate,
     dx, dS, inner, grad, avg, jump,
     exp,
     FacetArea, FacetNormal 
)
from dolfinx.io import (
    XDMFFile
)
import dolfinx.plot as plot
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from PIL import Image

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def printMPI(string, end='\n'):
    if rank == 0:
        print(string, end=end)

DOLFIN_EPS = 1e-16

#
# Problem class
#
class problem_FE(object):
    r"""
    DG numerical solution of
    b \cdot \grad u + \mu \cdot u = f   en \Omega,
    u = 0 on \partial\Omega^- = \{ x \in \partial\Omega, b(x)\cdot n(x) <0 \}
    """

    def __init__(self, diff_parameters, nx=32):
        #
        # Load PDE and discretization parameters
        #
        diff = self
        p = diff.parameters = diff_parameters

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/../../meshes/" + f"mesh_{p.mesh}_nx-{nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        # mesh = diff.mesh = Mesh(mesh_file)
        # diff.mesh = Mesh()
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = diff.mesh = infile.read_mesh()

        #
        # Build DG, FE spaces and functions
        #
        diff.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
        diff.Vh = FunctionSpace(mesh, diff.P1c)
        diff.u, diff.u_trial, diff.ub =  Function(diff.Vh), TrialFunction(diff.Vh), TestFunction(diff.Vh)

        x = SpatialCoordinate(diff.mesh)

        u_exact = diff.u_exact = Expression(-x[0]*(x[0]-1)*x[1]*(x[1]-1), diff.Vh.element.interpolation_points())
        diff.u_exact = Function(diff.Vh)
        diff.u_exact.interpolate(u_exact)

        f = diff.f = 2*x[1]*(x[1]-1) + 2*x[0]*(x[0]-1)

    def variational_problem_u(self):
        """Build variational problem"""
        #
        # Load variables from diff problem
        #
        diff = self
        u_trial, ub = diff.u_trial, diff.ub
        f = diff.f

        #
        # Variational problem
        #
        a_u = inner(grad(u_trial),grad(ub))*dx
        
        L_u = f * ub * dx

        diff.a_u = a_u
        diff.L_u = L_u

        facets = dolfinx.mesh.locate_entities_boundary(diff.mesh, dim=1, marker=lambda x: np.logical_or(np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),np.isclose(x[1], 0.0)),np.isclose(x[1],1.0)))
        dofs = dolfinx.fem.locate_dofs_topological(V=diff.Vh, entity_dim=1, entities=facets)
        diff.bc = dolfinx.fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=diff.Vh)
    
    def problem_solve(self, verbosity=0):
        #
        # Load variables from diff problem
        #
        diff = self

        #
        # PETSc options
        #
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        # petsc_options = {"ksp_type": "gmres", "pc_type": "ilu"}

        #
        # Define problem
        #
        problem_u = LinearProblem(diff.a_u, diff.L_u, bcs=[diff.bc], petsc_options=petsc_options)

        # Solve u
        diff.u.x.array[:] = problem_u.solve().x.array
        diff.u.x.scatter_forward()


# ---------------------------

def print_info(u_data):
# energy, dynamics = 0):
    u_max, u_min = u_data
    s = f"{u_max:.4e}"
    s += f" {u_min:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    # Define remaining parameters
    parser.add_argument('--mesh', default="square")
    
    # Other parameters
    parser.add_argument('--verbosity', default=0, help="No extra information shown")
    parser.add_argument('--plot', default=10, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--save', default=1, help="No figures and output saved")
    parser.add_argument('--savefile', default="diff", help="Name of output file")
    parser.add_argument('--server', default=0, help="Set to 1 if the code is set to run on a server")

    param = parser.parse_args()

    return param

#
# Main program
#
if(__name__ == "__main__"):
    #
    # Define parameters
    #
    p = parameters = define_parameters()
    printMPI("Parameters:")
    for k, v in vars(parameters).items():
        printMPI(f"  {k} = {v}")
    
    if int(p.verbosity):
        log.set_log_level(log.LogLevel.INFO)
        opts = PETSc.Options()
        opts["ksp_monitor"] = True
        # print(dir(opts))
        # opts[f"{option_prefix}verbose"] = True
        # log.set_log_active(False)
    else:
        log.set_log_level(log.LogLevel.ERROR)

    # Create new files
    with open("errorsL2.txt", 'w') as file:
            file.write('')
    with open("errorsH1.txt", 'w') as file:
        file.write('')

    for nx in [4,8,16,32]:
        #
        # Init problem
        #
        diff_FE = problem_FE(parameters,nx)
        diff_FE.variational_problem_u()

        #
        # Save output
        #
        do_save = bool(p.save)
        server = bool(p.server)
        base_name_save = p.savefile

        #
        # Plot
        #
        if server:
            pyvista.start_xvfb()
        if do_save:
            pyvista.OFF_SCREEN = True

        do_plot = (int(p.plot) > 0)
        plot_mesh = (int(p.plot_mesh) > 0)
        pyvista.set_plot_theme("document")

        if plot_mesh: # Plot mesh
            topology, cell_types, geometry = plot.create_vtk_mesh(diff_FE.mesh, diff_FE.mesh.topology.dim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=True, color="white")
            plotter.view_xy()
            if pyvista.OFF_SCREEN:
                plotter.screenshot("mesh.png", transparent_background=True)

                comm.Barrier()
                if rank == 0:
                    img = Image.open(f"mesh.png")
                    width, height = img.size
                    # Setting the points for cropped image
                    left = width/6
                    top = 0.08 * height
                    right = 5 * width/6
                    bottom = 0.92 * height
                    im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
                    im_cropped.save(f"mesh.png")
                comm.Barrier()
            else:
                plotter.show()

        #
        # More info
        #  
        printMPI("More info:")
        tdim = diff_FE.mesh.topology.dim
        num_cells = diff_FE.mesh.topology.index_map(tdim).size_local
        h = dolfinx.cpp.mesh.h(diff_FE.mesh, tdim, range(num_cells))

        #
        # Save max, min and energy
        #
        max_u_list = []
        min_u_list = []

        #
        # Print info
        #
        printMPI("u_max u_min")

        #
        # Solve problem
        #
        diff_iterations = diff_FE.problem_solve(verbosity=int(p.verbosity))

        u = diff_FE.u
        u_ex = diff_FE.u_exact

        # Print error
        errorL2 = np.sqrt(assemble_scalar(form((u-u_ex)**2*dx)))
        errorH1 = np.sqrt(assemble_scalar(form((u-u_ex)**2*dx + inner(grad(u-u_ex), grad(u-u_ex))*dx)))
        printMPI(f"Error L2: {errorL2: .2e}")
        printMPI(f"Error H1: {errorH1: .2e}")

        with open("errorsL2.txt", 'a') as file:
            file.write(str(errorL2)+'\n')
        with open("errorsH1.txt", 'a') as file:
            file.write(str(errorH1)+'\n')

        #
        # Print info
        #
        u_max, u_min = comm.allreduce(max(u.x.array), MPI.MAX), comm.allreduce(min(u.x.array), MPI.MIN)
        u_ex_max, u_ex_min = comm.allreduce(max(u_ex.x.array), MPI.MAX), comm.allreduce(min(u_ex.x.array), MPI.MIN)
        if rank == 0:
            max_u_list.append(u_max)
            min_u_list.append(u_min)

        if rank == 0:
            print_info((u_max, u_min))
            
        #
        # Plot
        #
        
        # Properties of the scalar bar
        sargs = dict(height=0.6, vertical=True, position_x=0.8, position_y=0.2, title='', label_font_size=24, shadow=True,n_labels=5, fmt="%.2g", font_family="arial")

        # Plot exact solution

        # Create a grid to attach the DoF values
        cells, types, x = plot.create_vtk_mesh(diff_FE.Vh)
        grid = pyvista.UnstructuredGrid(cells, types, x)

        grid.point_data["u_ex"] = u_ex.x.array[:]
        grid.set_active_scalars("u_ex")

        plotter = pyvista.Plotter()

        # warped = grid.warp_by_scalar()
        warped = grid.warp_by_scalar(factor=10)
        plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalar_bar_args=sargs,  cmap=mpl.colormaps["plasma"])
        plotter.view_xz()

        # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
        # otherwise create interactive plot
        if pyvista.OFF_SCREEN:
            figure = plotter.screenshot(f"./{base_name_save}_u_ex_nx-{int(nx)}.png", transparent_background=True)
                    
        comm.Barrier()
        if rank == 0:
            img = Image.open(f"./{base_name_save}_u_ex_nx-{int(nx)}.png")
            width, height = img.size
            # Setting the points for cropped image
            left = width/5
            top = 0.15 * height
            right = 0.9 * width
            bottom = 0.85 * height
            im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
            im_cropped.save(f"./{base_name_save}_u_ex_nx-{int(nx)}.png")
        else:
            plotter.show()
        plotter.close()

        # Plot approximation
        grid.point_data["u"] = u.x.array[:]
        grid.set_active_scalars("u")

        plotter = pyvista.Plotter()

        # warped = grid.warp_by_scalar()
        warped = grid.warp_by_scalar(factor=10)
        plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalar_bar_args=sargs,  cmap=mpl.colormaps["plasma"])
        plotter.view_xz()

        # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
        # otherwise create interactive plot
        if pyvista.OFF_SCREEN:
            figure = plotter.screenshot(f"./{base_name_save}_u_FE_nx-{int(nx)}.png", transparent_background=True)
                    
        comm.Barrier()
        if rank == 0:
            img = Image.open(f"./{base_name_save}_u_FE_nx-{int(nx)}.png")
            width, height = img.size
            # Setting the points for cropped image
            left = width/5
            top = 0.15 * height
            right = 0.9 * width
            bottom = 0.85 * height
            im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
            im_cropped.save(f"./{base_name_save}_u_FE_nx-{int(nx)}.png")
        else:
            plotter.show()
        plotter.close()