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
from compare_1d_2d import prop_errors

ureg = pint.UnitRegistry()
ureg.setup_matplotlib()

thickness = 8e-3
diameter = 56e-3
radius = diameter/2

flibe_diffusivity = htm.diffusivities.filter(material=htm.FLIBE, author='calderoni')[0]
flibe_solubility = htm.solubilities.filter(material=htm.FLIBE, author='calderoni')[0]
flibe_permeability = htm.permeabilities.filter(material=htm.FLIBE).mean()

data = np.genfromtxt("3D_cad/calderoni/derived_quantities.csv", delimiter=',', names = True)
ts = data['ts'] * htm.ureg.s
top_flux = np.abs(data['solute_flux_surface_10']) / (np.pi * radius**2) * htm.ureg.particle * htm.ureg.s**-1 * htm.ureg.m**-2
flux_1d = downstream_flux_salt(ts, P_up * htm.ureg.Pa, thickness * htm.ureg.m, flibe_diffusivity.value(temp)*flibe_solubility.value(temp), flibe_diffusivity.value(temp))

plt.plot(ts, top_flux, label = '2D two mat')
plt.plot(ts, flux_1d, label = '1D one mat')
plt.legend()
plt.title("Calderoni model comparison")
plt.show()
print('temp: ',temp)
print("already diff", flibe_diffusivity.value(temp))
error = prop_errors(top_flux.magnitude, ts.magnitude, thickness, temp, P_up, plot = True)
print(error)
folder = f"3D_cad/calderoni" 