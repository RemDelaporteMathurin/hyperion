#from festim_model import make_model
# Created a copy of festim model to preserve it
from festim_model_copy import make_model
import numpy as np
import h_transport_materials as htm

salt_thickness = 8e-3
salt_diameter = 80e-3

T_values = np.linspace(700, 900, num=5)
T_val = 773

thick_plot = False
T_plot = True


thicknesses = np.linspace(2e-3, 10e-3, num = 5)
diameters = np.linspace(20e-3, 100e-3, num = 5)

if __name__ == "__main__":

    if thick_plot:
        for diameter in diameters:
            for thickness in thicknesses:
                model_2d = make_model(
                    thickness,
                    diameter,
                    nx=40,
                    ny=20,
                    folder = f"results/{thickness*1000:.2f}mm_thick_{diameter*1000:.2f}mm_wide/2d",
                    two_dimensional = True
                )
                
                model_2d.T.value = T_val

                model_2d.initialise()
                model_2d.run()

                model_1d = make_model(
                    thickness,
                    diameter,
                    nx=40,
                    ny=20,
                    folder = f"results/{thickness*1000:.2f}mm_thick_{diameter*1000:.2f}mm_wide/1d",
                    two_dimensional = False
                )

                model_1d.T.value = T_val

                model_1d.initialise()
                model_1d.run()

            

    if T_plot:
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


