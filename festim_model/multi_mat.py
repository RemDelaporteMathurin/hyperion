
import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("3D_cad/derived_quantities.csv", delimiter = ',', names=True)
ts = data['ts']
flux = data['solute_flux_surface_10']

plt.plot(ts, flux)
plt.show()