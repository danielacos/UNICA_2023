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

    def __init__(self, conv_react_parameters):
        #
        # Load PDE and discretization parameters
        #
        conv_react = self
        p = conv_react.parameters = conv_react_parameters

        A_coef = conv_react.A_coef = float(p.A)
        conv = conv_react.conv = float(p.conv)
        mu = conv_react.mu = 2 * A_coef
        exp_coef = conv_react.exp_coef = float(p.exp)

        file_path = os.path.dirname(os.path.abspath(__file__))
        printMPI(f"file_path = {file_path}")
        mesh_file = f"{file_path}/../../meshes/" + f"mesh_{p.mesh}_nx-{p.nx}.xdmf"
        printMPI(f"mesh_file = {mesh_file}")

        #
        # Read mesh
        #
        # mesh = conv_react.mesh = Mesh(mesh_file)
        # conv_react.mesh = Mesh()
        with XDMFFile(comm, mesh_file, 'r') as infile:
            mesh = conv_react.mesh = infile.read_mesh()
        
        conv_react.nx = int(p.nx)

        #
        # Build DG, FE spaces and functions
        #
        conv_react.P1c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        conv_react.P1cv = VectorElement(FiniteElement("Lagrange", mesh.ufl_cell(), 1))
        conv_react.Vh = FunctionSpace(mesh, conv_react.P1c)
        conv_react.Vhv = FunctionSpace(mesh, conv_react.P1cv)
        conv_react.u, conv_react.u_trial, conv_react.ub =  Function(conv_react.Vh), TrialFunction(conv_react.Vh), TestFunction(conv_react.Vh)

        x = SpatialCoordinate(conv_react.mesh)

        u_exact = conv_react.u_exact = Expression(exp(-A_coef * exp_coef * ((x[0] - 0.3)**2 + (x[1] - 0.3)**2)), conv_react.Vh.element.interpolation_points())
        conv_react.u_exact = Function(conv_react.Vh)
        conv_react.u_exact.interpolate(u_exact)

        f = conv_react.f = mu * exp(-A_coef * exp_coef * ((x[0] - 0.3)**2 + (x[1] - 0.3)**2)) * (exp_coef * conv * 0.3 * x[1] - exp_coef * conv * 0.3 * x[0] + 1)

        def beta(x):
            vals = np.zeros((mesh.geometry.dim, x.shape[1]))
            vals[0] = conv * x[1]
            vals[1] = -conv * x[0]
            return vals
        # conv_react.beta = beta
        conv_react.beta = Function(conv_react.Vhv)
        conv_react.beta.interpolate(beta)

    def variational_problem_u(self):
        """Build variational problem"""
        #
        # Load variables from conv_react problem
        #
        conv_react = self
        nx = conv_react.nx
        u_trial, ub = conv_react.u_trial, conv_react.ub
        mu, beta, f = conv_react.mu, conv_react.beta, conv_react.f

        #
        # Variational problem
        #
        e_len = FacetArea(conv_react.mesh)
        n_e = FacetNormal(conv_react.mesh)
        l = 1.0/nx

        a_u = inner(beta, grad(u_trial)) * ub * dx \
            + mu * u_trial * ub * dx
        
        L_u = f * ub * dx

        conv_react.a_u = a_u
        conv_react.L_u = L_u
    
    def problem_solve(self, verbosity=0):
        #
        # Load variables from conv_react problem
        #
        conv_react = self

        #
        # PETSc options
        #
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        # petsc_options = {"ksp_type": "gmres", "pc_type": "ilu"}

        #
        # Define problem
        #
        problem_u = LinearProblem(conv_react.a_u, conv_react.L_u, petsc_options=petsc_options)

        # Solve u
        conv_react.u.x.array[:] = problem_u.solve().x.array
        conv_react.u.x.scatter_forward()


# ---------------------------

def print_info(u_data):
# energy, dynamics = 0):
    u_max, u_min = u_data
    s = f"{u_max:.4e}"
    s += f" {u_min:.4e}"
    printMPI(s)


def define_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--A', default=10)
    parser.add_argument('--exp', default=1000)
    parser.add_argument('--conv', default=1e6)

    # Define remaining parameters
    parser.add_argument('--mesh', default="circle")

    # Params for the discrete scheme
    parser.add_argument('--nx', default=2**7)
    
    # Other parameters
    parser.add_argument('--verbosity', default=0, help="No extra information shown")
    parser.add_argument('--plot', default=10, help="Plot shown every number of time steps")
    parser.add_argument('--plot_mesh', default=0, help="Plot mesh")
    parser.add_argument('--save', default=1, help="No figures and output saved")
    parser.add_argument('--savefile', default="conv_react", help="Name of output file")
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

    #
    # Init problem
    #
    conv_react_FE = problem_FE(parameters)
    conv_react_FE.variational_problem_u()

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
        topology, cell_types, geometry = plot.create_vtk_mesh(conv_react_FE.mesh, conv_react_FE.mesh.topology.dim)
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
    tdim = conv_react_FE.mesh.topology.dim
    num_cells = conv_react_FE.mesh.topology.index_map(tdim).size_local
    h = dolfinx.cpp.mesh.h(conv_react_FE.mesh, tdim, range(num_cells))

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
    conv_react_iterations = conv_react_FE.problem_solve(verbosity=int(p.verbosity))

    u = conv_react_FE.u
    u_ex = conv_react_FE.u_exact

    # Print error
    printMPI(f"Error L2: {np.sqrt(assemble_scalar(form((u-u_ex)**2*dx))): .2e}")
    printMPI(f"Error V: {np.sqrt(assemble_scalar(form((u-u_ex)**2*dx + inner(conv_react_FE.beta, grad(u-u_ex))**2*dx))): .2e}")

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
    cells, types, x = plot.create_vtk_mesh(conv_react_FE.Vh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    grid.point_data["u_ex"] = u_ex.x.array[:]
    grid.set_active_scalars("u_ex")

    plotter = pyvista.Plotter()

    # warped = grid.warp_by_scalar()
    warped = grid.warp_by_scalar(factor=1.75*1/(np.max([u_ex_max, 1])))
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalar_bar_args=sargs,  cmap=mpl.colormaps["plasma"])
    plotter.view_yz()

    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        figure = plotter.screenshot(f"./{base_name_save}_u_ex_nx-{int(p.nx)}.png", transparent_background=True)
                
    comm.Barrier()
    if rank == 0:
        img = Image.open(f"./{base_name_save}_u_ex_nx-{int(p.nx)}.png")
        width, height = img.size
        # Setting the points for cropped image
        left = 0.2 * width
        top = 0.15 * height
        right = 0.93 * width
        bottom = 0.85 * height
        im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
        im_cropped.save(f"./{base_name_save}_u_ex_nx-{int(p.nx)}.png")
    else:
        plotter.show()
    plotter.close()

    # Plot approximation
    grid.point_data["u"] = u.x.array[:]
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()

    # warped = grid.warp_by_scalar()
    warped = grid.warp_by_scalar(factor=1.75*1/(np.max([u_max, 1])))
    plotter.add_mesh(warped, show_edges=False, show_scalar_bar=True, scalar_bar_args=sargs,  cmap=mpl.colormaps["plasma"])
    plotter.view_yz()

    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    if pyvista.OFF_SCREEN:
        figure = plotter.screenshot(f"./{base_name_save}_u_FE_nx-{int(p.nx)}.png", transparent_background=True)
                
    comm.Barrier()
    if rank == 0:
        img = Image.open(f"./{base_name_save}_u_FE_nx-{int(p.nx)}.png")
        width, height = img.size
        # Setting the points for cropped image
        # left = 0.3 * width
        # top = 0.15 * height
        # right = 0.93 * width
        # bottom = 0.9 * height
        left = 0.2 * width
        top = 0.15 * height
        right = 0.93 * width
        bottom = 0.85 * height
        im_cropped = img.crop((left, top, right, bottom)) # default window size is 1024x768
        im_cropped.save(f"./{base_name_save}_u_FE_nx-{int(p.nx)}.png")
    else:
        plotter.show()
    plotter.close()