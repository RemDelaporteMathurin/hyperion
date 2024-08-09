import festim as F
import fenics as f
import h_transport_materials as htm
import numpy as np
import matplotlib.pyplot as plt


P_up = 100

def make_model(folder):
    model = F.Simulation()

    model.mesh = F.MeshFromXDMF(
        volume_file=f"{folder}/mesh_domains.xdmf", boundary_file=f"{folder}/mesh_boundaries.xdmf", type = 'cylindrical'
    )

    # Material properties
    # Using calderoni's numbers instead of the mean
    flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    # Adding flibe solubility properties
    flibe_solubility = htm.solubilities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    flibe = F.Material(
        id=6, # Given by the meshing file
        D_0=flibe_diffusivity.pre_exp.magnitude,
        E_D=flibe_diffusivity.act_energy.magnitude,
        S_0 = flibe_solubility.pre_exp.magnitude,
        E_S = flibe_solubility.act_energy.magnitude,
        solubility_law = "henry"
    )

    nickel_diffusivity = htm.diffusivities.filter(material=htm.NICKEL).mean()
    nickel_solubility = htm.solubilities.filter(material=htm.NICKEL).mean()
    nickel = F.Material(
            id = 7,
            D_0 = nickel_diffusivity.pre_exp.magnitude,
            E_D = nickel_diffusivity.act_energy.magnitude,
            S_0 = nickel_solubility.pre_exp.magnitude,
            E_S = nickel_solubility.act_energy.magnitude
    )

    model.materials = [nickel, flibe]

    model.boundary_conditions = [
        F.SievertsBC(surfaces=[8], S_0=nickel.S_0, E_S=nickel.E_S, pressure = P_up),
        # Dirichlet BCs on salt top and outside of side wall
        F.DirichletBC(field="solute", value = 0, surfaces=[9]),
        F.DirichletBC(field="solute", value = 0, surfaces = [10])
    ]

    model.T = F.Temperature(973)

    derived_quantities = F.DerivedQuantities(
        [F.SurfaceFluxCylindrical(field = "solute", surface=9), 
         F.SurfaceFluxCylindrical(field = "solute", surface=10),
         F.SurfaceFluxCylindrical(field = "solute", surface=8)],
        filename=f"{folder}/derived_quantities.csv",
    )

    model.exports = [F.XDMFExport("solute", folder = folder),
                     derived_quantities
                    ]

    model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        chemical_pot=True,
        final_time = 30*3600,
        maximum_iterations = 100
    )

    model.dt = F.Stepsize(initial_value = 10, stepsize_change_ratio = 1.1)


    model.log_level = 20

    return model

if __name__ == "__main__":


    folder = '3D_cad'
    model = make_model(folder)

    model.initialise()
    model.run()

    data = np.genfromtxt("3D_cad/derived_quantities.csv", delimiter = ',', names=True)
    ts = data['ts']
    flux = data['solute_flux_surface_9']

    plt.plot(ts, flux)
    plt.show()
