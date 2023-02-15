from festim_model import make_model
import numpy as np

salt_thickness = 10e-3
salt_diameter = 100e-3

T_values = np.linspace(730, 900, num=4)

if __name__ == "__main__":

    for T in T_values:
        model_2d = make_model(
            salt_thickness,
            salt_diameter,
            nx=40,
            ny=20,
            folder=f"results/{T:.0f}K/2d",
            two_dimensional=True,
        )

        model_2d.T.value = T

        model_2d.initialise()
        model_2d.run()

        model_1d = make_model(
            salt_thickness,
            salt_diameter,
            nx=40,
            ny=20,
            folder=f"results/{T:.0f}K/1d",
            two_dimensional=False,
        )

        model_1d.T.value = T

        model_1d.initialise()
        model_1d.run()
