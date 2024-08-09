import festim as F
import numpy as np

my_model = F.Simulation()

inner_radius = 0.1e-2
thickness = 0.2e-2
my_model.mesh = F.MeshFromVertices(
    np.linspace(inner_radius, inner_radius + thickness, num=100),
    type="cylindrical",
)

import h_transport_materials as htm

steel_D = htm.diffusivities.filter(material=htm.Steel).mean()
steel_S = htm.solubilities.filter(material=htm.Steel).mean()

steel = F.Material(
    id=1,
    D_0=steel_D.pre_exp.magnitude,
    E_D=steel_D.act_energy.magnitude,
    S_0=steel_S.pre_exp.magnitude,
    E_S=steel_S.act_energy.magnitude,
)

my_model.materials = steel

my_model.boundary_conditions = [
    F.SievertsBC(surfaces=[1], S_0=steel.S_0, E_S=steel.E_S, pressure=1e5),
    F.DirichletBC(surfaces=[2], value=0, field="solute"),
]

my_model.T = 500

my_model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-10,
    final_time=10000,
)

my_model.dt = F.Stepsize(10, stepsize_change_ratio=1.1)

derived_quantities = F.DerivedQuantities(
    [
        F.SurfaceFluxCylindrical(field="solute", surface=1),
        F.SurfaceFluxCylindrical(field="solute", surface=2),
    ],
    show_units=True,
)

txt_export = F.TXTExport(field="solute", filename="task14/solute.txt")

my_model.exports = [derived_quantities, txt_export]

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

cmap = plt.get_cmap("viridis")

data = np.genfromtxt("task14/solute.txt", skip_header=1, delimiter=",")
for i in range(1, data.shape[1], 3):
    plt.plot(data[:, 0] / 1e-3, data[:, i], color=cmap(i/data.shape[1]))

plt.xlabel("x [mm]")
plt.ylabel("c [H/m³]")
plt.show()

plt.plot(
    derived_quantities[0].t,
    np.abs(derived_quantities[0].data),
    linewidth=2,
    label="Inner surface",
)

plt.plot(
    derived_quantities[1].t,
    np.abs(derived_quantities[1].data),
    linewidth=2,
    label="Outer surface",
)

plt.xlabel("Time [s]")
plt.ylabel("Absolute flux [H/m s]")
plt.legend()
plt.show()