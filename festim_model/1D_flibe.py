import festim as F
import fenics as f
import h_transport_materials as htm

# I copy so that I can adjust things without messing up the original file
from cylindrical_integrator import CylindricalFlux

print(f"HTM version {htm.__version__}")

# Moving flibe material stuff outside of make_model definition
flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE).mean()
# adding flibe solubility variable and using that for the boundary conditions
flibe_solubility = htm.solubilities.filter(material=htm.FLIBE).mean()
flibe_permeability = htm.permeabilities.filter(material=htm.FLIBE).mean()

flibe = F.Material(
    id=1,
    D_0=flibe_diffusivity.pre_exp.magnitude,
    E_D=flibe_diffusivity.act_energy.magnitude,
    S_0 = flibe_solubility.pre_exp.magnitude,
    E_S = flibe_solubility.act_energy.magnitude
)


# changed folder destination
def make_model(thickness, diameter, nx, ny, two_dimensional: bool, folder="1D_flibe_results"):
    fenics_mesh = f.RectangleMesh(
        f.Point(0, 0), f.Point(diameter / 2, thickness), nx, ny
    )

    # marking physical groups (volumes and surfaces)
    volume_markers = f.MeshFunction("size_t", fenics_mesh, fenics_mesh.topology().dim())
    volume_markers.set_all(1)

    tol = 1e-14
    bottom_surface = f.CompiledSubDomain("on_boundary && near(x[1], 0, tol)", tol=tol)
    right_surface = f.CompiledSubDomain(
        "on_boundary && near(x[0], L, tol)", L=diameter / 2, tol=tol
    )
    top_surface = f.CompiledSubDomain(
        "on_boundary && near(x[1], H, tol)", H=thickness, tol=tol
    )
    surface_markers = f.MeshFunction(
        "size_t", fenics_mesh, fenics_mesh.topology().dim() - 1
    )
    surface_markers.set_all(0)
    bottom_surface.mark(surface_markers, 1)
    right_surface.mark(surface_markers, 2)
    top_surface.mark(surface_markers, 3)

    # f.XDMFFile("volume_markers.xdmf").write(volume_markers)
    # f.XDMFFile("surface_markers.xdmf").write(surface_markers)

    model = F.Simulation()

    my_mesh = F.Mesh(
        fenics_mesh,
        volume_markers=volume_markers,
        surface_markers=surface_markers,
        type="cylindrical",
    )
    model.mesh = my_mesh 

    model.materials = flibe

    # adjusting the henrybcs for the actual values of flibe

    upstream_pressure = F.HenrysBC(surfaces=[1], H_0=flibe.S_0, E_H=flibe.E_S, pressure=101325) # pressure in Pa
    recombination_lateral = F.DirichletBC(surfaces=[2], value=0, field="solute")
    recombination_top = F.DirichletBC(surfaces=[3], value=0, field="solute")
    model.boundary_conditions = [
        upstream_pressure,
        recombination_top,
    ]

    if two_dimensional:
        model.boundary_conditions.append(recombination_lateral)

    # Adjust temperature from 760K to 973.15K
    model.T = F.Temperature(973.15)

    model.settings = F.Settings(
        absolute_tolerance=1e4, relative_tolerance=1e-10, final_time=10 * 3600
    )
    model.dt = F.Stepsize(initial_value=10, stepsize_change_ratio=1.1)


    # Unsure about the exact values of the surfaces, the "right" is 2 but the cylinderical flux is on 3
    # Also made derived_quantities a model attribute
    model.derived_quantities = F.DerivedQuantities(
        [F.HydrogenFlux(surface=2), CylindricalFlux(surface=2), F.HydrogenFlux(surface=3)],
        filename=f"{folder}/derived_quantities.csv",
    )

    model.exports = [
        F.XDMFExport(field="solute", filename=f"{folder}/concentration.xdmf"),
        model.derived_quantities
    ]

    # model.log_level = 20

    return model

# defining downstream flux function here
def downstream_flux(t, P_up, permeability, L, D):
    """calculates the downstream H flux at a given time t

    Args:
        t (float, np.array): the time
        P_up (float): upstream partial pressure of H
        permeability (float): salt permeability
        L (float): salt thickness
        D (float): diffusivity of H in the salt

    Returns:
        float, np.array: the downstream flux of H
    """
    n_array = np.arange(1, 10000)[:, np.newaxis]
    summation = np.sum((-1)**n_array * np.exp(-(np.pi * n_array)**2 * D/L**2 * t), axis=0)
    return P_up * permeability / L * (2*summation + 1)


if __name__ == "__main__":

    # Dimensions of the flibe based on the experiment
    salt_thickness = 20e-3 # in m
    salt_diameter = 77.927e-3 # in m

    # Creating a 1D model of the cylinder of flibe
    model = make_model(
        salt_thickness, salt_diameter, nx=40, ny=40, two_dimensional=False
    )
    model.initialise()
    model.run()

    # Plotting the flux on the top surface #########################################

    times = model.derived_quantities.t
    computed_flux = model.derived_quantities.filter(surfaces=3).data

    import numpy as np
    import matplotlib.pyplot as plt

    P_up = 101325 * htm.ureg.Pa
    T = 973.15 * htm.ureg.K
    plt.scatter(times, np.abs(computed_flux), alpha=0.2, label="computed")
    plt.plot(times, downstream_flux(times * htm.ureg.s, P_up, permeability=flibe_permeability.value(T),
                                    L=salt_thickness * htm.ureg.m, D=flibe_diffusivity.value(T)),
            color="tab:orange", label="analytical"
            )
    plt.ylim(bottom=0)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Downstream flux (H/m2/s)")
    plt.show()
