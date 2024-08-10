# create 2D mesh with fenics

import fenics as f
import numpy as np
import matplotlib.pyplot as plt


# Create mesh and define function space
def make_model(n):
    mesh = f.UnitSquareMesh(n, n)

    # mark the boundary top bottom and right
    top = f.CompiledSubDomain("near(x[1], 1.0) && on_boundary")
    bottom = f.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
    right = f.CompiledSubDomain("near(x[0], 1.0) && on_boundary")

    # create volume markers
    sm = f.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sm.set_all(0)
    top.mark(sm, 1)
    bottom.mark(sm, 2)
    right.mark(sm, 3)

    vm = f.MeshFunction("size_t", mesh, 1)
    vm.set_all(1)

    import festim as F

    model = F.Simulation()

    model.mesh = F.Mesh(
        mesh=mesh, surface_markers=sm, volume_markers=vm, type="cylindrical"
    )

    model.T = 300

    model.materials = F.Material(id=1, D_0=1.0, E_D=0)

    model.boundary_conditions = [
        F.DirichletBC(field="solute", value=2, surfaces=[1]),
        F.DirichletBC(field="solute", value=0, surfaces=[2, 3]),
    ]

    model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, transient=False
    )

    flux_1 = F.SurfaceFluxCylindrical(field="solute", surface=1)
    flux_2 = F.SurfaceFluxCylindrical(field="solute", surface=2)
    flux_3 = F.SurfaceFluxCylindrical(field="solute", surface=3)
    model.exports = [F.DerivedQuantities([flux_1, flux_2, flux_3])]
    return model, flux_1, flux_2, flux_3


total_fluxes = []
ns = np.geomspace(10, 300, 10, dtype=int)
for n in ns:
    model, flux_1, flux_2, flux_3 = make_model(n)
    model.initialise()
    model.run()

    # same with .2e
    print("Flux 1", "{:.2e}".format(flux_1.data[-1]))
    print("Flux 2", "{:.2e}".format(flux_2.data[-1]))
    print("Flux 3", "{:.2e}".format(flux_3.data[-1]))
    total_flux = flux_1.data[-1] + flux_2.data[-1] + flux_3.data[-1]
    print(
        "Total flux",
        "{:.2e}".format(total_flux),
    )
    # show total flux as a percent of flux 1
    print(
        "Total flux as a percent of flux 1",
        "{:.2%}".format(total_flux / flux_1.data[-1]),
    )
    total_fluxes.append(total_flux)

# plot total fluxes
plt.plot(ns, total_fluxes)
plt.xlabel("Number of discretization points")
plt.ylabel("Total flux")
plt.ylim(bottom=0)
plt.show()
