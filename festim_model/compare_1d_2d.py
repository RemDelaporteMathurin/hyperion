import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pint

from run_comparison import T_values

ureg = pint.UnitRegistry()
ureg.setup_matplotlib()

plt.gca().xaxis.set_units(ureg.hour)

cmap = plt.get_cmap("Reds")

norm = Normalize(vmin=700, vmax=900)

for T in T_values:
    data_1d = np.genfromtxt(
        f"results/{T:.0f}K/1d/derived_quantities.csv", delimiter=",", names=True
    )
    data_2d = np.genfromtxt(
        f"results/{T:.0f}K/2d/derived_quantities.csv", delimiter=",", names=True
    )

    t_1d = data_1d["ts"] * ureg.s
    t_2d = data_2d["ts"] * ureg.s
    flux_1d = np.abs(data_1d["Flux_surface_3_solute"]) * ureg.particle * ureg.s**-1
    flux_2d = np.abs(data_2d["Flux_surface_3_solute"]) * ureg.particle * ureg.s**-1
    plt.plot(t_1d, flux_1d, color=cmap(norm(T)))
    plt.plot(t_2d, flux_2d, color=cmap(norm(T)))

    plt.fill(
        np.append(t_1d, t_2d[::-1]),
        np.append(flux_1d, flux_2d[::-1]),
        alpha=0.5,
        color=cmap(norm(T)),
    )

    plt.annotate(
        f"  {T:.0f} K", (t_1d[-1], (flux_1d[-1] + flux_2d[-1]) / 2), color=cmap(norm(T))
    )

plt.xlabel(f"Time ({plt.gca().xaxis.get_units()})")
plt.ylabel(f"Permeation flux ({plt.gca().yaxis.get_units():~P})")

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.gca().spines[["right", "top"]].set_visible(False)

plt.show()
