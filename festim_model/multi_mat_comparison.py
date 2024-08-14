
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pint
import h_transport_materials as htm
from matplotlib.colors import Normalize



from convert_mesh import convert_med_to_xdmf
from festim_model_copy import downstream_flux_salt
from make_multi_mat import P_up, temp
from scipy.optimize import curve_fit
from compare_1d_2d import prop_errors, ureg

#ureg = pint.UnitRegistry()
ureg.setup_matplotlib()

thickness = 8e-3
diameter = 80e-3
radius = 40e-3


# flibe material properties
flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE, author='calderoni')[0]
flibe_solubility = htm.solubilities.filter(material=htm.FLIBE, author='calderoni')[0]
flibe_permeability = htm.permeabilities.filter(material=htm.FLIBE).mean()

flibe_thicknesses = [2e-3, 5e-3, 8e-3, 14e-3]
nickel_thicknesses = [1e-3, 2e-3, 3e-3]

# For error calculations


cmap = plt.get_cmap("Blues")
norm = Normalize(vmin = flibe_thicknesses[0]-12e-3, vmax = flibe_thicknesses[-1])
plt.gca().xaxis.set_units(ureg.hour)


if __name__ == "__main__":

    overall_error = {
    "diffusivity error": [],
    "solubility error": [],
    "permeability error": [] 
    }

    for nickel_thickness in nickel_thicknesses:

        errors = {
        "diffusivity error": [],
        "solubility error": [],
        "permeability error": []
        }

        for flibe_thickness in flibe_thicknesses:

            folder = f"3D_cad/{flibe_thickness*1000:.0f}mm_{nickel_thickness*1000:.0f}mm_nickel" 
            data = np.genfromtxt(f"{folder}/derived_quantities.csv", delimiter = ',', names=True)

            # 2D one material data
            folder_1d= f"2D_model/{flibe_thickness*1000:.2f}mm_thick_80.00mm_wide/1d"
            folder_2d = f"2D_model/{flibe_thickness*1000:.2f}mm_thick_80.00mm_wide/2d"
            data_1d = np.genfromtxt(f"{folder_1d}/derived_quantities.csv", delimiter=',', names=True)
            data_2d = np.genfromtxt(f"{folder_2d}/derived_quantities.csv", delimiter = ',', names = True)
            ts = data['ts'] * ureg.s
            ts_1d = data_1d['ts'] * ureg.s
            ts_2d = data_2d['ts'] * ureg.s
            flux_1d = np.abs(data_1d["solute_flux_surface_3"]) * ureg.particle * ureg.s**-1 / np.pi / (diameter/2)**2 * ureg.m**-2
            flux_2d = np.abs(data_2d["solute_flux_surface_3"]) * ureg.particle * ureg.s**-1 / np.pi / (diameter/2)**2 * ureg.m**-2
            top_flux = np.abs(data['solute_flux_surface_10']) / (np.pi * radius**2) * ureg.particle * ureg.s**-1 * ureg.m**-2
            bottom_flux = data['solute_flux_surface_8'] / (np.pi * radius**2)
            side_flux = data['solute_flux_surface_9'] / (np.pi * diameter * thickness)



            plt.plot(ts.to(ureg.hour), np.abs(top_flux), label = 'multi material', color = cmap(norm(flibe_thickness)))
            plt.plot(ts_1d.to(ureg.hour), flux_1d, label = 'one dimensional one material', color = cmap(norm(flibe_thickness)), linestyle = 'dashed')
            plt.plot(ts_2d.to(ureg.hour), flux_2d, color = cmap(norm(flibe_thickness)), linestyle = 'dotted')
            
            plt.fill(
                np.append(ts_1d, ts[::-1]),
                np.append(flux_1d, top_flux[::-1]),
                alpha = 0.5,
                color = cmap(norm(flibe_thickness))
            )

            plt.annotate(
                f"  {flibe_thickness*1000:.0f}mm FLiBe", (ts[-1] + 0.3*ureg.hour, (flux_1d[-1] + top_flux[-1]) / 2), color=cmap(norm(flibe_thickness))
            )

            errors["diffusivity error"].append(prop_errors(top_flux.magnitude, ts.magnitude, flibe_thickness, temp, P_up, plot = False)['diffusivity error'])
            errors["solubility error"].append(prop_errors(top_flux.magnitude, ts.magnitude, flibe_thickness, temp, P_up)['solubility error'])
            errors["permeability error"].append(prop_errors(top_flux.magnitude, ts.magnitude, flibe_thickness, temp, P_up)['permeability error'])
            
        overall_error['diffusivity error'].append(errors['diffusivity error'])
        overall_error['permeability error'].append(errors['permeability error'])
        overall_error['solubility error'].append(errors['solubility error'])

        plt.title(f"Permeation fluxes at nickel thickness: {nickel_thickness*1000:.0f}mm")
        plt.xlabel(f"Time ({plt.gca().xaxis.get_units()})")
        plt.ylabel(f"Permeation flux ({plt.gca().yaxis.get_units():~P})")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.savefig(rf"multi_mat_{nickel_thickness*1000:.0f}mm.svg")
        plt.show()
        # Creating an error contour like in 1D_model.ipynb

    XX, YY = np.meshgrid(np.array(flibe_thicknesses)*1e3, np.array(nickel_thicknesses)*1e3)
    ZZ = overall_error['permeability error']

    CF = plt.contourf(XX, YY, ZZ, levels = 100)
    CS = plt.contour(XX,YY,ZZ, levels = 15, colors = 'white')
    plt.clabel(CS, fmt="%.2f")
    plt.xlabel(f"Flibe thickness (mm)")
    plt.ylabel(f"Nickel thickness (mm)")
    from matplotlib.cm import ScalarMappable
    plt.colorbar(CF, label = 'Difference (%)')
    plt.title("Permeability error by varying FLiBe thickness and nickel thickness")
    plt.savefig('multi_mat_contour.svg')
    plt.show()