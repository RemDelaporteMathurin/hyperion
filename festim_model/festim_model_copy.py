import festim as F
import fenics as f
import h_transport_materials as htm
import numpy as np

#from cylindrical_integrator import CylindricalFlux

print(f"HTM version {htm.__version__}")

P_up = 100

def make_model(thickness, diameter, nx, ny, two_dimensional: bool, folder="results", pressure = P_up):
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

    # Using calderoni's numbers instead of the mean
    flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    # Adding flibe solubility properties
    flibe_solubility = htm.solubilities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    flibe = F.Material(
        id=1,
        D_0=flibe_diffusivity.pre_exp.magnitude,
        E_D=flibe_diffusivity.act_energy.magnitude,
        S_0 = flibe_solubility.pre_exp.magnitude,
        E_S = flibe_solubility.act_energy.magnitude
    )



    model.materials = flibe

    # Using the actual flibe properties and pressure (~100s of Pa)
    upstream_pressure = F.HenrysBC(surfaces=[1], H_0=flibe.S_0, E_H=flibe.E_S, pressure=P_up)
    recombination_lateral = F.DirichletBC(surfaces=[2], value=0, field="solute")
    recombination_top = F.DirichletBC(surfaces=[3], value=0, field="solute")
    model.boundary_conditions = [
        upstream_pressure,
        recombination_top,
    ]

    if two_dimensional:
        model.boundary_conditions.append(recombination_lateral)

    model.T = F.Temperature(760)

    model.settings = F.Settings(
        absolute_tolerance=1e4, relative_tolerance=1e-10, final_time=8 * 3600
    )
    model.dt = F.Stepsize(initial_value=10, stepsize_change_ratio=1.1)

    # Changing both of the derived quantities to be cylindrical surface flux
    derived_quantities = F.DerivedQuantities(
        [F.SurfaceFluxCylindrical(field = "solute", surface=2), F.SurfaceFluxCylindrical(field = "solute", surface=3)],
        filename=f"{folder}/derived_quantities.csv",
    )

    model.exports = [
        F.XDMFExport(field="solute", filename=f"{folder}/concentration.xdmf"),
        derived_quantities,
    ]

    # model.log_level = 20

    return model




# Function for computing analytical solution to flux
def downstream_flux_salt(t, P_up, L, permeability, D):
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
    salt_thickness = 40e-3
    salt_diameter = 50e-3

    model = make_model(
        salt_thickness, salt_diameter, nx=20, ny=40, two_dimensional=True
    )
    model.initialise()
    model.run()
