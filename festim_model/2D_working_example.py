import festim as F
import fenics as f
import h_transport_materials as htm
import pint
import numpy as np
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()
ureg.setup_matplotlib()

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
        E_S = flibe_solubility.act_energy.magnitude,
        solubility_law = "henry"
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

    model.T = F.Temperature(973)

    model.settings = F.Settings(
        absolute_tolerance=1e4, relative_tolerance=1e-10, final_time=8 * 3600
    )
    model.dt = F.Stepsize(initial_value=10, stepsize_change_ratio=1.1)

    # Changing both of the derived quantities to be cylindrical surface flux
    derived_quantities = F.DerivedQuantities(
        [F.SurfaceFluxCylindrical(field = "solute", surface=2), F.SurfaceFluxCylindrical(field = "solute", surface=3), F.SurfaceFluxCylindrical(field = 'solute', surface = 1)],
        filename=f"{folder}/derived_quantities.csv",
    )

    model.exports = [
        F.XDMFExport(field="solute", filename=f"{folder}/concentration.xdmf"),
        derived_quantities,
    ]

    #model.log_level = 20

    return model

if __name__ == "__main__":
    salt_thickness = 2e-3
    salt_diameter = 80e-3

    model = make_model(
        salt_thickness, salt_diameter, nx=500, ny=500, two_dimensional=True
    )
    model.initialise()
    model.run()

    data = np.genfromtxt(
            "results/derived_quantities.csv", delimiter=",", names=True
        ) 
    
    ts = data["ts"] * ureg.s
    # Salt surface flux
    flux_top = data["solute_flux_surface_3"] 
    # Salt lateral flux
    flux_lateral = data["solute_flux_surface_2"] 
    # Salt bottom flux
    flux_bottom = data["solute_flux_surface_1"]

    plt.plot(ts, np.abs(flux_top), label = 'top')
    plt.plot(ts, np.abs(flux_lateral), label = 'side')
    plt.legend()
    plt.show()

    total_flux = flux_top[-1] + flux_bottom[-1] + flux_lateral[-1]
    print(total_flux)


    