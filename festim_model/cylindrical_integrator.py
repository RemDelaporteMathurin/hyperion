import festim as F
import fenics as f


class CylindricalFlux(F.HydrogenFlux):
    def __init__(self, surface, coverage=2 * f.pi) -> None:
        super().__init__(surface)
        self.coverage = coverage

    def compute(self, soret=False):
        r = f.Expression("x[0]", degree=1)
        field_to_prop = {
            "0": self.D,
            "solute": self.D,
            0: self.D,
            "T": self.thermal_cond,
        }
        self.prop = field_to_prop[self.field]
        flux = self.coverage * f.assemble(
            r * self.prop * f.dot(f.grad(self.function), self.n) * self.ds(self.surface)
        )
        return flux
