import festim as F
import fenics as f
import h_transport_materials as htm
import numpy as np
import matplotlib.pyplot as plt

from convert_mesh import convert_med_to_xdmf


P_up = 100
temp = 700

def make_model(folder):
    model = F.Simulation()

    model.mesh = F.MeshFromXDMF(
        volume_file=f"{folder}/mesh_domains.xdmf", boundary_file=f"{folder}/mesh_boundaries.xdmf", type = 'cylindrical'
    )

    # Material properties
    # Using calderoni's numbers instead of the mean
    flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    print('diffusivity',flibe_diffusivity.value(temp))
    print('pre_exp',flibe_diffusivity.pre_exp)
    print('act_eng',flibe_diffusivity.act_energy)
    # Adding flibe solubility properties
    flibe_solubility = htm.solubilities.filter(material=htm.FLIBE, author = 'calderoni')[0]
    flibe = F.Material(
        id=6, # Given by the meshing file
        D_0=flibe_diffusivity.pre_exp.magnitude*2.268,
        E_D=flibe_diffusivity.act_energy.magnitude*1.04985,
        S_0 = flibe_solubility.pre_exp.magnitude*0.96279,
        E_S = flibe_solubility.act_energy.magnitude*0.987,
        solubility_law = "henry"
    )
    R = 8.63e-5
    diff = flibe.D_0*np.exp(-flibe.E_D / temp / R)
    sol = flibe.S_0*np.exp(-flibe.E_S/R/temp)
    print('diff', diff)
    print('solubility', flibe_solubility.value(temp))
    print('solubility pre_exp', flibe_solubility.pre_exp)
    print('solubility act_energy', flibe_solubility.act_energy)
    print('sol', sol)

    print('raaahhhhh', (flibe_diffusivity.value(temp).magnitude-diff)/flibe_diffusivity.value(temp).magnitude)
    print('grrr', (flibe_solubility.value(temp).magnitude-sol)/flibe_solubility.value(temp).magnitude)
    

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

    model.T = F.Temperature(temp)

    derived_quantities = F.DerivedQuantities(
        [F.SurfaceFluxCylindrical(field = "solute", surface=9), 
         F.SurfaceFluxCylindrical(field = "solute", surface=10),
         F.SurfaceFluxCylindrical(field = "solute", surface=8)],
        filename=f"{folder}/derived_quantities.csv",
        #show_units = True
    )

    model.exports = [F.XDMFExport("solute", folder = folder),
                     derived_quantities
                    ]

    model.settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-10,
        chemical_pot=True,
        final_time = 20*3600,
        maximum_iterations = 100,
        #transient = False
    )

    model.dt = F.Stepsize(initial_value = 100, stepsize_change_ratio = 1.1)


    #model.log_level = 20

    return model

if __name__ == "__main__":

    '''

    flibe_thicknesses = [5]
    nickel_thicknesses = [1,2,3]

    for flibe_thickness in flibe_thicknesses:
        for nickel_thickness in nickel_thicknesses:
            folder = f'3D_cad/{flibe_thickness}mm_{nickel_thickness}mm_nickel'
            filename = folder + f'/{flibe_thickness}mm_{nickel_thickness}mm_nickel.med'
            correspondance_dict, cell_data_types = convert_med_to_xdmf(filename, 
                                                                    cell_file = f"{folder}/mesh_domains.xdmf",  
                                                                    facet_file=f"{folder}/mesh_boundaries.xdmf",
                                                                    cell_type="triangle",
                                                                    facet_type="line")
            model = make_model(folder)

            model.initialise()
            model.run()
            '''
    
    folder = '3D_cad/calderoni'
    filename = '3D_cad/calderoni/calderoni.med'
    correspondance_dict, cell_data_types = convert_med_to_xdmf(filename, 
                                                                    cell_file = f"{folder}/mesh_domains.xdmf",  
                                                                    facet_file=f"{folder}/mesh_boundaries.xdmf",
                                                                    cell_type="triangle",
                                                                    facet_type="line") 
    model = make_model(folder)

    model.initialise()
    model.run()
